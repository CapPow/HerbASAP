#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU General Public License
#    as published by the Free Software Foundation; either version 3
#    of the License, or (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""

    AYUP (as of yet unnamed program) performs post processing steps on raw 
    format images of natural history specimens. Specifically designed for 
    Herbarium sheet images.

"""
# imports here
import numpy as np

from keras.models import load_model
import keras.backend as K
from PIL import Image


class ccRead():
    def __init__(self, parent=None, *args):
        super(ccRead, self).__init__()
        self.parent = parent
        self.position_model = load_model("models/mlp_histogram.hdf5")
        self.discriminator_model = load_model("models/discriminator.hdf5")
        self.size_model = load_model("models/size_model.hdf5")

    def predict_uncertainty_position(self, model_function, x, n_iter=10):
        result = []

        for i in range(n_iter):
            result.append(model_function([x[0], x[1], 1]))

        result = np.array(result)
        uncertainty = result.var(axis=0)
        prediction = result.mean(axis=0)
        return prediction, uncertainty

    def predict_uncertainty_discriminator(self, model_function, x, n_iter=10):
        result = []

        for i in range(n_iter):
            result.append(model_function([x, 1]))

        result = np.array(result)
        uncertainty = result.var(axis=0)
        prediction = result.mean(axis=0)
        return prediction, uncertainty

    def testFeature(self, img_filepath, stride=25, partition_size=100, buffer_size=20):
        """Returns bool condition, if this module functions on a test input."""

        position_function = K.function(
            [self.position_model.layers[0].input, self.position_model.layers[1].input, K.learning_phase()],
            [self.position_model.layers[-1].output])

        discriminator_function = K.function([self.discriminator_model.layers[0].input, K.learning_phase()],
                                            [self.discriminator_model.layers[-1].output])


        try:
            im = Image.open(img_filepath)
            image_width, image_height = im.size
            possible_positions = []

            hists_rgb = []
            hists_hsv = []
            for r in range((image_height - partition_size) // stride + 2):
                for c in range((image_width - partition_size) // stride + 2):
                    x1, y1 = c * stride, r * stride
                    x2, y2 = x1 + partition_size, y1 + partition_size
                    partitioned_im = im.crop((x1, y1, x2, y2))
                    possible_positions.append((x1, y1, x2, y2))
                    partitioned_im_hsv = partitioned_im.convert("HSV")

                    hists_rgb.append(partitioned_im.histogram())
                    hists_hsv.append(partitioned_im_hsv.histogram())

            hists_rgb = np.array(hists_rgb).astype('float16') / 255
            hists_hsv = np.array(hists_hsv).astype('float16') / 255

            position_prediction, position_uncertainty = self.predict_uncertainty_position(position_function,
                                                                                     [hists_rgb, hists_hsv],
                                                                                     len([hists_rgb, hists_hsv]))

            only_cc_position_uncertainty = position_prediction[0][:, 1]
            only_cc_position_prediction = position_prediction[0][:, 1]

            lowest_uncertainties = []
            most_certain_images = {}
            for idx, prediction_value in enumerate(only_cc_position_prediction):
                if len(list(most_certain_images)) >= buffer_size:
                    del most_certain_images[max(most_certain_images.keys())]

                if prediction_value > 0.9:
                    most_certain_images[only_cc_position_uncertainty[idx]] = possible_positions[idx]

            highest_prob_images = []
            for position in list(most_certain_images.values()):
                highest_prob_images.append(np.array(im.crop(position)))

            highest_prob_images_pred = np.array(highest_prob_images)
            discriminator_prediction, discriminator_uncertainty = self.predict_uncertainty_discriminator(
                discriminator_function,
                highest_prob_images_pred,
                len(highest_prob_images_pred))

            only_cc_uncertainty_column = []
            only_cc_probability_column = []
            try:
                only_cc_uncertainty_column = discriminator_uncertainty[0][:, 1]
                only_cc_probability_column = discriminator_prediction[0][:, 1]
            except IndexError:
                print("Discriminator could not find best image.")

            lowest_uncertainty = 1
            best_image = None
            for idx, prediction_value in enumerate(only_cc_probability_column):
                if prediction_value == max(only_cc_probability_column) and \
                        prediction_value > 0.9 and \
                        only_cc_uncertainty_column[idx] < lowest_uncertainty:
                    lowest_uncertainty = only_cc_uncertainty_column[idx]
                    best_image = im.crop(possible_positions[idx])

            try:
                best_image.save(_)
            except ValueError:
                print(f"Did not find a colorchip due to value error.")
            except AttributeError:
                print(f"Did not find a colorchip due to attribute error.")
            except IndexError:
                print(f"Did not find a colorchip due to index error.")
            except SystemError:
                # best_image = im.crop(best_image_crop)
                # best_image.save(F"Dataset/Images/ml_dual_results/{base_name}.jpg", 'JPEG', quality=100)
                pass
        except:
            # some unknown error, assume test failed
            return False
