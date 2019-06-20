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


try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K
except ImportError:
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

        self.position_function = K.function(
            [self.position_model.layers[0].input, self.position_model.layers[1].input, K.learning_phase()],
            [self.position_model.layers[-1].output])

        self.discriminator_function = K.function([self.discriminator_model.layers[0].input, K.learning_phase()],
                                                 [self.discriminator_model.layers[-1].output])

    def _predict_uncertainty_position(self, x, n_iter=10):
        """
        Predicts with uncertainty the position of a color chip using the mean and variance.

        :param x: A list that contains both the RGB histogram and the HSV histogram of a given partition. Format of the
        list should be [rgb_histogram, hsv_histogram]
        :type x: list, ndarray
        :param n_iter: Number of iterations for the results. Most often would be the length of the list of histograms,
        though in a sorted list could be a value between 0 and length of list.
        :type n_iter: int
        :return: Returns the prediction and uncertainty. The prediction is a category exclusive probability.
        :rtype: list
        """
        result = []

        for i in range(n_iter):
            result.append(self.position_function([x[0], x[1], 1]))

        result = np.array(result)
        uncertainty = result.var(axis=0)
        prediction = result.mean(axis=0)
        return prediction, uncertainty

    def _predict_uncertainty_discriminator(self, x, n_iter=10):
        """
        Predicts with uncertainty the probability that the given partition contains a color chip using the mean and
        variance.

        :param x: A list that contains both the RGB histogram and the HSV histogram of a given partition. Format of the
        list should be [rgb_histogram, hsv_histogram]
        :type x: list, ndarray
        :param n_iter: Number of iterations for the results. Most often would be the length of the list of histograms,
        though in a sorted list could be a value between 0 and length of list.
        :type n_iter: int
        :return: Returns the prediction and uncertainty. The prediction is a category exclusive probability.
        :rtype: list
        """
        result = []

        for i in range(n_iter):
            result.append(self.discriminator_function([x, 1]))

        result = np.array(result)
        uncertainty = result.var(axis=0)
        prediction = result.mean(axis=0)
        return prediction, uncertainty

    def testFeature(self, img_filepath, stride=25, partition_size=100, buffer_size=20):
        """
        Tests whether the given image (and its color chip) is compatible with the neural network.

        :param img_filepath: Path to the image.
        :type img_filepath: str
        :param stride: The amount of pixels that the partition window will move.
        :type stride: int
        :param partition_size: The size of the partition window.
        :type partition_size: int
        :param buffer_size: The amount of images the region proposal network will keep for the discriminator. In
        general, the higher the number of this buffer size, the more likely that the true color chip will reside in the
        buffer. However, this also decreases (linearly) how many images can be processed within a given time.
        :type buffer_size: int
        :return: Returns true if the neural networks are able to detect a color chip within an image. Returns false if
        it cannot find a color chip.
        :rtype: bool
        """

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

        position_prediction, position_uncertainty = self._predict_uncertainty_position([hists_rgb, hists_hsv],
                                                                                       len([hists_rgb, hists_hsv]))

        only_cc_position_uncertainty = position_prediction[0][:, 1]
        only_cc_position_prediction = position_prediction[0][:, 1]

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
        discriminator_prediction, discriminator_uncertainty = self._predict_uncertainty_discriminator(
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
            return True
        except ValueError:
            return False
        except AttributeError:
            return False
        except IndexError:
            return False
        except SystemError:
            return False
