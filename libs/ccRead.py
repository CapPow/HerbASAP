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
from numpy.lib.stride_tricks import as_strided
import numbers

import os

try:
    import plaidml.keras
    plaidml.keras.install_backend()

    from keras.models import load_model
    import keras.backend as K
except ImportError:
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras import backend as K
    except ImportError:
        from keras.models import load_model
        import keras.backend as K
    finally:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from PIL import Image, ImageCms
import cv2
import time
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(f"[INFO] Forcing use of CPU for neural network prediction (TensorFlow)")


class ColorchipRead:
    def __init__(self, parent=None, *args):
        super(ColorchipRead, self).__init__()
        self.parent = parent
        # self.position_model = load_model("libs/models/mlp_proposal.hdf5")
        # self.discriminator_model = load_model("libs/models/discriminator.hdf5")
        # self.high_precision_model = load_model("libs/models/highprecision_discriminator.hdf5")
        # self.size_det_model = load_model("libs/models/size_model.hdf5")
        # self.large_colorchip_regressor_model = load_model("libs/models/lcc_regressor.hdf5")
        # self.size_model = load_model("libs/models/size_model.hdf5")

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
#        init_im = np.zeros((250, 250, 3)).astype('uint8')
#        init_im = Image.fromarray(init_im)
#        print("[INFO] Initializing neural networks")
#        self.predict_colorchip_size(init_im)
#        self.process_colorchip_big(init_im)
#        self.process_colorchip_small(init_im, (250, 250))
#        print("[INFO] Finished initializing neural networks")

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
        variance. This uses the high precision model, which is more robust against false positives if

        Note: Currently, this model performs very poorly (relative to the normal discriminator model) against images
        with drastic white balance shift.

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

        pil_image = np.array(im)
        pil_image = Image.fromarray(pil_image)
        return pil_image

    def predict_colorchip_size(self, im):
        """
        Predicts the size of the color chip through color histograms and a dense neural network. This is essential for
        knowing the correct neural network model to use for determining the color chip values.
        :param im: The image to be predicted on.
        :type im: OCV Image
        :return: Returns 'big' for big colorchips, and 'small' for small colorchips.
        :rtype: str
        """

        start = time.time()
        im = self.ocv_to_pil(im)
        im_hsv = im.convert("HSV")

        hist_rgb = im.histogram()
        hist_hsv = im_hsv.histogram()

        X_rgb = np.array(hist_rgb) / 8196
        X_hsv = np.array(hist_hsv) / 8196

        size_det_result = self.size_det_model.predict([[X_rgb], [X_hsv]])[0]

        end = time.time()
        print(f"Predict color chip size took: {end - start} seconds.")
        if size_det_result[0] > size_det_result[1]:
            return 'big'
        else:
            return 'small'

    def process_colorchip_big(self, im):
        """
        Processes big color chips using neural networks
        :param im: The image to be processed.
        :type im: OCV Image
        :return: TBD
        """
        start = time.time()
        im = self.ocv_to_pil(im)
        original_image_width, original_image_height = im.size

        resized_im = im.resize((256, 256))
        im_np = np.array(resized_im) / 255

        crop_vals = self.large_colorchip_regressor_model.predict(np.array([im_np]))[0]

        prop_crop_vals = np.array(crop_vals) / 256
        prop_x1, prop_y1, prop_x2, prop_y2 = prop_crop_vals[0], prop_crop_vals[1], prop_crop_vals[2], prop_crop_vals[3]
        scaled_x1, scaled_x2 = prop_x1 * original_image_width, prop_x2 * original_image_width
        scaled_y1, scaled_y2 = prop_y1 * original_image_height, prop_x2 * original_image_height

        cropped_im = im.crop((scaled_x1, scaled_y1, scaled_x2, scaled_y2))
        end = time.time()
        cc_crop_time = round(end - start, 3)
        try:
            return (scaled_x1, scaled_y1, scaled_x2, scaled_y2), cropped_im, cc_crop_time
        except SystemError as e:
            print(f"System error: {e}")

    def process_colorchip_small(self, im, original_size, stride_style='quick',
                                stride=25, partition_size=125, buffer_size=10,
                                over_crop=0, high_precision=False):
        """

        :param im:
        :param original_size:
        :param stride_style:
        :param stride:
        :param partition_size:
        :param buffer_size:
        :param over_crop:
        :param high_precision:
        :return:
        """
        im = self.ocv_to_pil(im)
        im_hsv = im.convert("HSV")
        start = time.time()
        image_width, image_height = im.size
        original_width, original_height = original_size
        possible_positions = []
        hists_rgb = []
        hists_hsv = []

        if stride_style == 'whole':
            nstart = time.time()
            im = np.array(im)
            im_hsv = np.array(im_hsv)
            partitions = self.extract_patches(im,
                                              (partition_size, partition_size, 3),
                                              stride)
            partitions_hsv = self.extract_patches(im_hsv,
                                                  (partition_size, partition_size, 3),
                                                  stride)
            partitions = np.reshape(partitions, (-1, 125, 125, 3))
            partitions_hsv = np.reshape(partitions_hsv, (-1, 125, 125, 3))
            for r in range(-over_crop, (image_height - partition_size) // stride + over_crop):
                for c in range(-over_crop, (image_width - partition_size) // stride + over_crop):
                    x1, y1 = c * stride, r * stride
                    x2, y2 = x1 + partition_size, y1 + partition_size
                    possible_positions.append((x1, y1, x2, y2))
            #         partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
            #         hists_rgb.append(partitioned_im.histogram())
            #         hists_hsv.append(partitioned_im_hsv.histogram())

            for index in range(len(partitions)):
                partitioned_im = Image.fromarray(partitions[index])
                partitioned_im_hsv = Image.fromarray(partitions_hsv[index])
                hists_rgb.append(partitioned_im.histogram())
                hists_hsv.append(partitioned_im_hsv.histogram())

            print(f"New striding took {time.time() - nstart}")

        elif stride_style == 'quick':
            for c in range(-over_crop, (image_width - partition_size) // stride + over_crop):
                x1, y1 = c * stride, 0
                x2, y2 = x1 + partition_size, y1 + partition_size
                partitioned_im = im.crop((x1, y1, x2, y2))
                possible_positions.append((x1, y1, x2, y2))
                partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))

                hist = partitioned_im.histogram()
                hist_hsv = partitioned_im_hsv.histogram()

                hists_rgb.append(hist)
                hists_hsv.append(hist_hsv)

                x1, y1 = c * stride, image_height - partition_size
                x2, y2 = x1 + partition_size, y1 + partition_size
                partitioned_im = im.crop((x1, y1, x2, y2))
                possible_positions.append((x1, y1, x2, y2))
                partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
                hist = partitioned_im.histogram()
                hist_hsv = partitioned_im_hsv.histogram()

                hists_rgb.append(hist)
                hists_hsv.append(hist_hsv)

            for r in range(-over_crop, (image_height - partition_size) // stride + over_crop):
                x1, y1 = 0, r * stride
                x2, y2 = x1 + partition_size, y1 + partition_size
                partitioned_im = im.crop((x1, y1, x2, y2))
                possible_positions.append((x1, y1, x2, y2))
                partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
                hist = partitioned_im.histogram()
                hist_hsv = partitioned_im_hsv.histogram()

                hists_rgb.append(hist)
                hists_hsv.append(hist_hsv)

                x1, y1 = image_width - partition_size, r * stride
                x2, y2 = x1 + partition_size, y1 + partition_size
                partitioned_im = im.crop((x1, y1, x2, y2))
                possible_positions.append((x1, y1, x2, y2))
                partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
                hist = partitioned_im.histogram()
                hist_hsv = partitioned_im_hsv.histogram()

                hists_rgb.append(hist)
                hists_hsv.append(hist_hsv)
        elif stride_style == 'ultraquick':
            positions = [(0, 0, partition_size, partition_size),
                         (0, image_height - partition_size, partition_size, image_height),
                         (image_width - partition_size, 0, image_width, partition_size),
                         (image_width - partition_size, image_height - partition_size, image_width, image_height)]

            for position in positions:
                partitioned_im = im.crop(position)
                possible_positions.append(position)
                partitioned_im_hsv = im_hsv.crop(position)
                hists_rgb.append(partitioned_im.histogram())
                hists_hsv.append(partitioned_im_hsv.histogram())
        else:
            raise RuntimeError('Invalid stride_style was given, must be "quick" or "whole"')

        hists_rgb = np.array(hists_rgb) / 255
        hists_hsv = np.array(hists_hsv) / 255

        position_prediction, position_uncertainty = self._predict_uncertainty_position([hists_rgb, hists_hsv],
                                                                                       len([hists_rgb, hists_hsv]))

        only_cc_position_uncertainty = position_uncertainty[0][:, 1]
        only_cc_position_prediction = position_prediction[0][:, 1]

        most_certain_images = {}
        position_prediction_dict = dict(zip(only_cc_position_prediction,
                                            [i for i in range(len(only_cc_position_prediction))]))

        for prediction in sorted(position_prediction_dict)[-buffer_size:]:
            most_certain_images[only_cc_position_uncertainty[position_prediction_dict[prediction]]] = \
                possible_positions[position_prediction_dict[prediction]]

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
            print("Using high precision discriminator")
            highest_prob_rgb_hists = []
            highest_prob_hsv_hists = []
            for position in list(most_certain_images.values()):
                h_image = im.crop(position)
                h_image_hsv = im_hsv.crop(position)
                highest_prob_rgb_hists.append(h_image.histogram())
                highest_prob_hsv_hists.append(h_image_hsv.histogram())
                highest_prob_images.append(np.array(h_image))

            highest_prob_images_pred = np.array(highest_prob_images)
            highest_prob_rgb_hists = np.array(highest_prob_rgb_hists)
            highest_prob_hsv_hists = np.array(highest_prob_hsv_hists)
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
        best_location = None

        # for i in range(buffer_size):
        #     if only_cc_probability_column[i] == float('inf'): # Catching inf prediction, which is not permissible.
        #         only_cc_probability_column[i] = 0
        #         only_cc_uncertainty_column[i] = 1

        max_discriminator_pred = max(only_cc_probability_column)
        if max_discriminator_pred > 0:
            for idx, prediction_value in enumerate(only_cc_probability_column):
                if prediction_value > max_discriminator_pred - 0.05 and \
                        only_cc_uncertainty_column[idx] < lowest_uncertainty:
                    lowest_uncertainty = only_cc_uncertainty_column[idx]
                    best_location = list(most_certain_images.values())[idx]
                    best_image = im.crop(tuple(best_location))

        x1, y1, x2, y2 = best_location[0], best_location[1], best_location[2], best_location[3]
        prop_x1, prop_y1, prop_x2, prop_y2 = x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height

        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = prop_x1 * original_width, \
                                                     prop_y1 * original_height, \
                                                     prop_x2 * original_width, \
                                                     prop_y2 * original_height

        end = time.time()
        try:
            # best_image.show()
            print(f"Color chip cropping took: {end - start} seconds.")
            cc_crop_time = round(end - start, 3)
            return (scaled_x1, scaled_y1, scaled_x2, scaled_y2), best_image, cc_crop_time
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

    def predict_color_chip_quadrant(self, original_size, scaled_crop_location):
        """
        Returns the quadrant of the color chip location
        :param original_size: Size of the original image, in the format (width, height)
        :type original_size: tuple
        :param scaled_crop_location: Tuple of the scaled location of the scaled colorchip crop location, in the format
        (x1, y1, x2, y2)
        :type scaled_crop_location: tuple
        :return: Returns the quadrant where the color chip lies.
        :rtype: int
        """

        original_width, original_height = original_size
        x1, y1, x2, y2 = scaled_crop_location
        half_width = original_width / 2
        half_height = original_height / 2

        if x1 < half_width and y1 < half_height:
            return 4
        elif x1 > half_width and y1 < half_height:
            return 1
        elif x1 > half_width and y1 > half_height:
            return 2
        elif x1 < half_width and y1 > half_height:
            return 3
        else:
            return None

    def predict_color_chip_whitevals(self, color_chip_image):
        """
        Takes the white values within the cropped CC image and averages them in RGB. The whitest values in the image is
        determined in the L*a*b color space, wherein only lightness values higher than (max lightness value - 1) is
        considered
        
        :param color_chip_image: The cropped color chip image.
        :type color_chip_image: Image
        :return: Returns a list of the averaged whitest values
        :rtype: list
        """

        cci_array = np.array(color_chip_image,  dtype=np.uint8)
        ccil_array = cv2.cvtColor(color_chip_image, cv2.COLOR_RGB2Lab)
        width, height = ccil_array.shape[0], ccil_array.shape[1]
        # id max_lightness in ccil (lab space)
        max_lightness = ccil_array[...,0].max()
        # index cci_array based on conditions met in ccil_array
        white_pixels_rgbvals = cci_array[ ccil_array[...,0]> max_lightness-1]
        white_pixels_average = np.average(white_pixels_rgbvals, axis=0)

        return list(white_pixels_average)

    def test_feature(self, im, original_size, cc_size='predict'):
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
        if cc_size == 'predict':
            cc_size = self.predict_colorchip_size(im)
        if cc_size == 'big':
            cc_position, cropped_cc, cc_crop_time = self.process_colorchip_big(im)
        else:
            cc_position, cropped_cc, cc_crop_time= self.process_colorchip_small(im, original_size)
        
        if isinstance(cc_position, tuple):
            ccStatus = True
        else:
            ccStatus = False
        return ccStatus, cropped_cc
            
            