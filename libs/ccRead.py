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
import cv2


class ColorchipRead():
    def __init__(self, parent=None, *args):
        super(ColorchipRead, self).__init__()
        self.parent = parent
        self.position_model = load_model("libs/models/mlp_proposal.hdf5")
        self.discriminator_model = load_model("libs/models/discriminator.hdf5")
        self.high_precision_model = load_model("libs/models/highprecision_discriminator.hdf5")
        # self.size_model = load_model("models/size_model.hdf5")

        self.position_function = K.function(
            [self.position_model.layers[0].input, self.position_model.layers[1].input, K.learning_phase()],
            [self.position_model.layers[-1].output])

        self.discriminator_function = K.function([self.discriminator_model.layers[0].input, K.learning_phase()],
                                                 [self.discriminator_model.layers[-1].output])

        self.high_precision_discriminator_function = K.function([self.high_precision_model.layers[10].input,
                                                                 self.high_precision_model.layers[11].input,
                                                                 self.high_precision_model.layers[0].input,
                                                                 K.learning_phase()],
                                                                [self.high_precision_model.layers[-1].output])

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

    def _predict_uncertainty_hp_discriminator(self, x, n_iter=10):
        """
        Predicts with uncertainty the probability that the given partition contains a color chip using the mean and
        variance. This uses the high precision model, which is more robust against false positives.

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
            result.append(self.high_precision_discriminator_function([x[0], x[1], x[2], 1]))

        result = np.array(result)
        uncertainty = result.var(axis=0)
        prediction = result.mean(axis=0)
        return prediction, uncertainty

    def ocv_to_pil(self, im):
        """
        Converts an OCV image into PIL format. From
        https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format?noredirect=1&lq=1
        :param im: OpenCV image.
        :type im:
        :return:
        """
        #pil_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pil_image = np.array(im)
        pil_image = Image.fromarray(pil_image)
        return pil_image

    def predict_color_chip_location(self, im, stride=50, partition_size=125, buffer_size=20, high_precision=False):
        
        im = self.ocv_to_pil(im)
        image_width, image_height = im.size
        possible_positions = []
        hists_rgb = []
        hists_hsv = []
        for r in range(-2, (image_height - partition_size) // stride + 2):
            for c in range(-2, (image_width - partition_size) // stride + 2):
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

        only_cc_position_uncertainty = position_uncertainty[0][:, 1]
        only_cc_position_prediction = position_prediction[0][:, 1]

        most_certain_images = {}

        for idx, prediction_value in enumerate(only_cc_position_prediction):
            if len(list(most_certain_images)) >= buffer_size:
                del most_certain_images[max(most_certain_images.keys())]

            if prediction_value > 0.98:
                most_certain_images[only_cc_position_uncertainty[idx]] = possible_positions[idx]

        only_cc_uncertainty_column = []
        only_cc_probability_column = []
        highest_prob_images = []
        if not high_precision:
            for position in list(most_certain_images.values()):
                highest_prob_images.append(np.array(im.crop(position)))

            highest_prob_images_pred = np.array(highest_prob_images)
            discriminator_prediction, discriminator_uncertainty = self._predict_uncertainty_discriminator(
                highest_prob_images_pred,
                len(highest_prob_images_pred))

            try:
                only_cc_uncertainty_column = discriminator_uncertainty[0][:, 1]
                only_cc_probability_column = discriminator_prediction[0][:, 1]
            except IndexError:
                print("Discriminator could not find best image.")
        else:
            highest_prob_rgb_hists = []
            highest_prob_hsv_hists = []
            for position in list(most_certain_images.values()):
                h_image = im.crop(position)
                highest_prob_rgb_hists.append(h_image.histogram())
                highest_prob_hsv_hists.append(h_image.histogram())
                highest_prob_images.append(np.array(h_image))

            highest_prob_images_pred = np.array(highest_prob_images)
            discriminator_prediction, discriminator_uncertainty = self._predict_uncertainty_hp_discriminator(
                [highest_prob_rgb_hists,
                 highest_prob_hsv_hists,
                 highest_prob_images_pred],
                len(highest_prob_rgb_hists))

            try:
                only_cc_uncertainty_column = discriminator_uncertainty[0][:, 1]
                only_cc_probability_column = discriminator_prediction[0][:, 1]
            except IndexError:
                print("Discriminator could not find best image.")

        lowest_uncertainty = 1
        best_location = None
        for idx, prediction_value in enumerate(only_cc_probability_column):
            if prediction_value == max(only_cc_probability_column) and \
                    prediction_value > 0.9 and \
                    only_cc_uncertainty_column[idx] < lowest_uncertainty:
                lowest_uncertainty = only_cc_uncertainty_column[idx]
                best_location = list(most_certain_images.values())[idx]

        try:
            return best_location
        except ValueError as e:
            print(f"ccRead had a value error: {e}")
            return None
        except AttributeError as e:
            print(f"ccRead had an attribute error: {e}")
            return None
        except IndexError as e:
            print(f"ccRead had an index error: {e}")
            return None
        except SystemError as e:
            print(f"ccRead had a system error: {e}")
            return None

    def test_feature(self, im, stride=50, partition_size=125, buffer_size=20, high_precision=False):
        """
        Tests whether the given image (and its color chip) is compatible with the neural network. This function does not
        spit out values. Compatibility means that the neural network found a color-chip like object, but does not
        ensure that the object found is truly a color chip.

        :param im: Image to be tested. Must be in PIL format.
        :type im: Image
        :param stride: The amount of pixels that the partition window will move.
        :type stride: int
        :param partition_size: The size of the partition window.
        :type partition_size: int
        :param buffer_size: The amount of images the region proposal network will keep for the discriminator. In
        general, the higher the number of this buffer size, the more likely that the true color chip will reside in the
        buffer. However, this also decreases (linearly) how many images can be processed within a given time.
        :type buffer_size: int
        :param high_precision: Boolean to control whether or not to use the high precision discriminator model. The
        high precision is slightly less accurate in terms of centering the color chip, but should have less false
        positives. It is also slower than the regular discriminator model.
        :return: Returns true if the neural networks are able to detect a color chip within an image. Returns false if
        it cannot find a color chip.
        :rtype: bool
        """

        image_width, image_height = im.size
        possible_positions = []

        hists_rgb = []
        hists_hsv = []
        for r in range(-2, (image_height - partition_size) // stride + 2):
            for c in range(-2, (image_width - partition_size) // stride + 2):
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

        only_cc_position_uncertainty = position_uncertainty[0][:, 1]
        only_cc_position_prediction = position_prediction[0][:, 1]

        most_certain_images = {}

        for idx, prediction_value in enumerate(only_cc_position_prediction):
            if len(list(most_certain_images)) >= buffer_size:
                del most_certain_images[max(most_certain_images.keys())]

            if prediction_value > 0.98:
                most_certain_images[only_cc_position_uncertainty[idx]] = possible_positions[idx]

        only_cc_uncertainty_column = []
        only_cc_probability_column = []
        highest_prob_images = []
        if not high_precision:
            for position in list(most_certain_images.values()):
                highest_prob_images.append(np.array(im.crop(position)))

            highest_prob_images_pred = np.array(highest_prob_images)
            discriminator_prediction, discriminator_uncertainty = self._predict_uncertainty_discriminator(
                highest_prob_images_pred,
                len(highest_prob_images_pred))

            try:
                only_cc_uncertainty_column = discriminator_uncertainty[0][:, 1]
                only_cc_probability_column = discriminator_prediction[0][:, 1]
            except IndexError:
                print("Discriminator could not find best image.")
        else:
            highest_prob_rgb_hists = []
            highest_prob_hsv_hists = []
            for position in list(most_certain_images.values()):
                h_image = im.crop(position)
                highest_prob_rgb_hists.append(h_image.histogram())
                highest_prob_hsv_hists.append(h_image.histogram())
                highest_prob_images.append(np.array(h_image))

            highest_prob_images_pred = np.array(highest_prob_images)
            discriminator_prediction, discriminator_uncertainty = self._predict_uncertainty_hp_discriminator(
                [highest_prob_rgb_hists,
                 highest_prob_hsv_hists,
                 highest_prob_images_pred],
                len(highest_prob_rgb_hists))

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
                best_image = Image.fromarray(highest_prob_images[idx])

        try:
            best_image.show()
            return True
        except ValueError as e:
            print(f"ccRead had a value error: {e}")
            return False
        except AttributeError as e:
            print(f"ccRead had an attribute error: {e}")
            return False
        except IndexError as e:
            print(f"ccRead had an index error: {e}")
            return False
        except SystemError as e:
            print(f"ccRead had a system error: {e}")
            return False
