#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import time
from os import path, environ
import sys

# Importing TensorFlow within debug messages
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Importing Keras and making sure that Keras uses TensorFlow instead of some other backend.
import keras
from keras.models import load_model
from keras import backend as K
if K.backend() != 'tensorflow':
    raise RuntimeError(f"Please set your keras.json to use TensorFlow. It is currently using {keras.backend.backend()}")
# from libs.settingsWizard import ImageDialog
from libs.models.keras_frcnn.useFRCNN import process_image_frcnn


class ColorChipError(Exception):
    def __init__(self, msg='ColorChipError', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
    pass


class InvalidStride(ColorChipError):
    def __init__(self):
        default_message = 'Invalid stride_style was given, must be "quick" or "whole"'
        super().__init__(default_message)


class EmptyDiscriminatorArray(ColorChipError):
    def __init__(self):
        default_message = 'Unable to properly slice discriminator prediction array. It is likely empty.'
        super().__init__(default_message)


class DiscriminatorFailed(ColorChipError):
    def __init__(self):
        default_message = 'Could not find a color reference chart. Altering the window size set in the settings wizard may help. Check that it is unobstructed and parallell to the focal plane.'
        super().__init__(default_message)


class FRCNNLCCFailed(ColorChipError):
    def __init__(self):
        default_message = 'F-RCNN could not find the colorchip within this image.'
        super().__init__(default_message)

class SquareFindingFailed(ColorChipError):
    def __init__(self):
        default_message = 'We could not find the appropriate square with the white internal color patch.'
        super().__init__(default_message)


class ColorchipRead:

    def __init__(self, parent=None, *args):
        super(ColorchipRead, self).__init__()
        self.parent = parent
        try: # try to load models from relative paths
            self.position_model = tf.lite.Interpreter(model_path='libs/models/mlp_proposal.tflite')
            self.K_position_model = load_model("libs/models/mlp_proposal_k.hdf5")
            self.discriminator_model = tf.lite.Interpreter(
                model_path='libs/models/discriminator.tflite')
        except (ValueError, OSError):
            # alternative loading to accomodate pyinstaller
            bundle_dir = getattr(sys, '_MEIPASS', path.abspath(path.dirname(__file__)))
            self.position_model = tf.lite.Interpreter(
                model_path=path.join(bundle_dir, 'libs/models/mlp_proposal.tflite'))

            self.K_position_model = load_model(path.join(bundle_dir, "libs/models/mlp_proposal_k.hdf5"))
            self.discriminator_model = tf.lite.Interpreter(
                model_path=path.join(bundle_dir, 'libs/models/discriminator.tflite'))

        self.position_model.allocate_tensors()
        self.position_input_details = self.position_model.get_input_details()
        self.position_output_details = self.position_model.get_output_details()
        self.position_function = K.function([self.K_position_model.layers[0].input,
                                             self.K_position_model.layers[1].input,
                                             K.learning_phase()],
                                            [self.K_position_model.layers[-1].output])
        self.discriminator_model.allocate_tensors()
        self.discriminator_input_details = self.discriminator_model.get_input_details()
        self.discriminator_output_details = self.discriminator_model.get_output_details()

    def ocv_to_pil(self, im):
        """
        Converts an OCV image into PIL format. From
        https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format?noredirect=1&lq=1
        :param im: OpenCV image.
        :type im: numpy.ndarray
        :return: Returns an image converted to PIL format.
        :rtype: PIL.Image
        """

        pil_image = np.array(im)
        pil_image = Image.fromarray(pil_image)
        return pil_image

    def _position_with_uncertainty(self, x, n_iter=20):
        result = []

        for i in range(n_iter):
            result.append(self.position_function([x[0], x[1], 1]))

        result = np.array(result)
        uncertainty = result.var(axis=0)
        prediction = result.mean(axis=0)
        return prediction, uncertainty

    def process_colorchip_big(self, im, pp_fix=0):
        """
        Processes big colorchips using a minimally-modified Google MobileNetV2 neural network model. The model predicts
        the bounding box within a shrunken 256x256 image. Most large colorchips should be accurately predicted, as long
        as enough colorchip information is retained after this resizing.
        :param im: The image to be processed. Should have a large colorchip.
        :type im: Image
        :return: Returns a tuple containing the bounding box (x1, y1, x2, y2) and the cropped image of the colorchip.
        :rtype: tuple
        """
        start = time.time()

        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = process_image_frcnn(im, pp_fix=pp_fix)
        if max(scaled_x1, scaled_y1, scaled_x2, scaled_y2) == 0:
            raise FRCNNLCCFailed

        cropped_im = im[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
        try:
            cc_crop_time = round(time.time() - start, 3)
            return (scaled_x1, scaled_y1, scaled_x2, scaled_y2), cropped_im, cc_crop_time
        except SystemError as e:
            raise ColorChipError(f"System error: {e}")

    @staticmethod
    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    @staticmethod
    def find_squares(img, contour_area_floor=850, contour_area_ceiling=100000, leap=6, k_size=5):
        # taken from: opencv samples
        if k_size > 0:
            img = cv2.GaussianBlur(img, (k_size, k_size), 0)
        squares = []
        for gray in cv2.split(img):
            for thrs in range(0, 255, leap):
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 50, apertureSize=3)
                    bin = cv2.dilate(bin, None)
                else:
                    _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_len = cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                    contourArea = cv2.contourArea(cnt)
                    if len(cnt) == 4 and contour_area_floor < contourArea < contour_area_ceiling \
                            and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([ColorchipRead.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                                          for i in range(4)])
                        if max_cos < 0.1:
                            squares.append(cnt)
        return squares

    @staticmethod
    def _legacy_regions(im, im_hsv, image_width, image_height, whole_extrema, stride_style='quick', stride=25,
                        partition_size=125, over_crop=0, hard_cut_value=50):
        possible_positions = []
        hists_rgb = []
        hists_hsv = []

        #Always perform the ultraquick method
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


        if stride_style == 'whole':
            for r in range(-over_crop, (image_height - partition_size) // stride + over_crop):
                for c in range(-over_crop, (image_width - partition_size) // stride + over_crop):
                    x1, y1 = c * stride, r * stride
                    x2, y2 = x1 + partition_size, y1 + partition_size
                    partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
                    partitioned_im_hsv = partitioned_im_hsv.resize((125, 125))
                    extrema = partitioned_im_hsv.getextrema()
                    extrema = extrema[1][1]
                    if whole_extrema - hard_cut_value < extrema:
                        possible_positions.append((x1, y1, x2, y2))
                        partitioned_im = im.crop((x1, y1, x2, y2))
                        partitioned_im = partitioned_im.resize((125, 125))
                        hists_rgb.append(partitioned_im.histogram())
                        hists_hsv.append(partitioned_im_hsv.histogram())

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
        else:
            raise InvalidStride

        return hists_rgb, hists_hsv, possible_positions

    def _square_first_regions(self):
        pass

    def process_colorchip_small(self, im, original_size, stride_style='quick',
                                partition_size=125, discriminator_floor=0.99,
                                over_crop=1, hard_cut_value=50, high_precision=True,
                                try_hard=True):
        """
        Finds small colorchips using the quickCC model. This model is specifically trained on tiny colorchips found in
        many herbarium collections. If the colorchip is similar in size to those, and is in the same proportion to the
        image, the model should be able to find it.

        :param im: The image containing the small colorchip.
        :type im: PIL.Image
        :param original_size: The original size of the image as a tuple containing (width, height). This is used for
        calculating the final bounding box respective to the original size of the image.
        :type original_size: tuple
        :param stride_style: A string that denotes the stride style, either "whole", "quick", or "ultraquick". In
        whole, the neural network will look at the entirety of the image. Quick will look at only the outside borders
        of the image, and therefore is appropriate for collections wherein the colorchip is found in the outside
        border. Ultraquick will only look at the four outer corners of the image. Using quicker stride styles heavily
        decrease the amount of processing time.
        :type stride_style: str
        :param stride: The amount of pixels the sliding window will move. In general, lower values will allow more
        accurate predictions, but at the cost of significantly higher computation time. If you find that your
        colorchips are mis-cropped, lower this value.
        :type stride: int
        :param partition_size: The partition size of the sliding window. If your colorchip is slightly large (relative
        to other collections), increase this value. If it is slightly too small (relative to other collections)
        decrease this value.
        :type partition_size: int
        :param buffer_size: The amount of partition images to be processed by the classifier network. A higher value
        will increase accuracy (if the proposal network did not correctly determine the colorchip) but also increase
        computation time.
        :type buffer_size: int
        :param over_crop: The amount of pixels the sliding window will go past the original image dimensions. Helpful
        if your colorchips are cropped at the end of the image.
        :type over_crop: int
        :param high_precision: [INFO] Will be changed soon, must rewrite docs for this parameter.
        :type high_precision: bool
        :param try_hard: If the model should make multiple attempts to locate the CRC using various image manipulations.
        :type try_hard: bool
        :return: Returns a tuple containing the bounding box (x1, y1, x2, y2) and the cropped image of the colorchip.
        :rtype: tuple
        """
        im = self.ocv_to_pil(im)
        im_hsv = im.convert("HSV")
        whole_extrema = im_hsv.getextrema()
        whole_extrema = whole_extrema[1][1]
        start = time.time()
        image_width, image_height = im.size
        original_width, original_height = original_size
        discriminator_model = self.discriminator_model
        stride = partition_size // 5
        hists_rgb, hists_hsv, possible_positions = self._legacy_regions(im=im, im_hsv=im_hsv,
                                                                        image_width=image_width,
                                                                        image_height=image_height,
                                                                        whole_extrema=whole_extrema,
                                                                        stride_style=stride_style, 
                                                                        stride=stride,
                                                                        partition_size=partition_size,
                                                                        over_crop=over_crop,
                                                                        hard_cut_value=hard_cut_value)

        hists_rgb = np.array(hists_rgb, dtype=np.uint16)
        hists_hsv = np.array(hists_hsv, dtype=np.uint16)

        position_predictions = []
        position_prediction, position_uncertainty = self._position_with_uncertainty([hists_rgb, hists_hsv], 20)

        only_cc_position_uncertainty = position_uncertainty[0][:, 1]
        only_cc_position_prediction = position_prediction[0][:, 1]

        indices = [index for index in range(len(only_cc_position_prediction))]
        position_uncertainty, position_predictions, indices = \
            (list(t) for t in zip(*sorted(zip(only_cc_position_uncertainty, only_cc_position_prediction, indices))))

        max_pred = max(position_predictions)
        for _j, position_prediction in enumerate(position_predictions):
            try:
                if position_prediction <= (max_pred - 0.001):
                    del position_prediction
                    del position_uncertainty[_j]
                    del indices[_j]
            except IndexError:
                break

        highest_prob_images = []
        highest_prob_positions = []

        for i in indices:
            highest_prob_images.append(np.array(im.crop(possible_positions[i]).resize((125, 125))))
            highest_prob_positions.append(possible_positions[i])

        highest_prob_images_pred = np.array(highest_prob_images, dtype=np.float32) / 255

        for i, highest_prob_image in enumerate(highest_prob_images):
            discriminator_model.set_tensor(discriminator_model.get_input_details()[0]['index'], [highest_prob_images_pred[i]])
            discriminator_model.invoke()
            disc_value = discriminator_model.get_tensor(discriminator_model.get_output_details()[0]['index'])[0][1]
            if disc_value > discriminator_floor:
                best_image = Image.fromarray(highest_prob_image)
                best_location = highest_prob_positions[i]
                break            
        else: 
            if try_hard:
                for i, highest_prob_image in enumerate(highest_prob_images):
                    rot_image = np.rot90(highest_prob_images_pred[i])
                    discriminator_model.set_tensor(discriminator_model.get_input_details()[0]['index'], [rot_image])
                    discriminator_model.invoke()
                    disc_value = discriminator_model.get_tensor(discriminator_model.get_output_details()[0]['index'])[0][1]
                    if disc_value > discriminator_floor - 0.1:
                        best_image = Image.fromarray(highest_prob_image)
                        best_location = highest_prob_positions[i]
                        print("Rerunning disciminator using rotated images was SUCCESSFUL! (1)")
                        break  
                else:
                    for i, highest_prob_image in enumerate(highest_prob_images):
                        rot_image = np.rot90(highest_prob_images_pred[i], k=2)
                        discriminator_model.set_tensor(discriminator_model.get_input_details()[0]['index'], [rot_image])
                        discriminator_model.invoke()
                        disc_value = discriminator_model.get_tensor(discriminator_model.get_output_details()[0]['index'])[0][1]
                        if disc_value > discriminator_floor - 0.2:
                            best_image = Image.fromarray(highest_prob_image)
                            best_location = highest_prob_positions[i]
                            print("Rerunning disciminator using rotated images was SUCCESSFUL! (2)")
                            break  
                    else:
                        for i, highest_prob_image in enumerate(highest_prob_images):
                            rot_image = np.rot90(highest_prob_images_pred[i], k=3)
                            discriminator_model.set_tensor(discriminator_model.get_input_details()[0]['index'], [rot_image])
                            discriminator_model.invoke()
                            disc_value = discriminator_model.get_tensor(discriminator_model.get_output_details()[0]['index'])[0][1]
                            if disc_value > discriminator_floor - 0.3:
                                best_image = Image.fromarray(highest_prob_image)
                                best_location = highest_prob_positions[i]
                                print("Rerunning disciminator using rotated images was SUCCESSFUL! (3)")
                                break
                        else:# otherwise,if quick stride was chosen try agian using whole
                            if stride_style == "quick":
                                print("Trying whole stride")
                                return self.process_colorchip_small(im,
                                                                    original_size,
                                                                    'whole',
                                                                    partition_size,
                                                                    discriminator_floor-0.1,
                                                                    over_crop, hard_cut_value,
                                                                    high_precision)
                            else:
                                # if whole was already tried.. finally.. give up.
                                raise DiscriminatorFailed

        best_image = np.array(best_image, dtype=np.uint8)
        x1, y1, x2, y2 = best_location[0], best_location[1], best_location[2], best_location[3]
        if high_precision:
            best_image, best_square = self.high_precision_cc_crop(best_image)
            ratio = 125 / partition_size
            best_square = [best_square[0] // ratio,
                           best_square[1] // ratio,
                           best_square[2] // ratio,
                           best_square[3] // ratio]
            x1, y1, x2, y2 = best_location[0] + best_square[0], best_location[1] + best_square[1], \
                             best_location[0] + best_square[2], best_location[1] + best_square[3]

        xc = np.array([x1, x2])
        yc = np.array([y1, y2])
        xc = np.clip(xc, 0, original_size[0])
        yc = np.clip(yc, 0, original_size[1])
        x1, y1, x2, y2 = xc[0], yc[0], xc[1], yc[1]

        prop_x1, prop_y1, prop_x2, prop_y2 = x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height

        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = int(prop_x1 * original_width), \
                                                     int(prop_y1 * original_height), \
                                                     int(prop_x2 * original_width), \
                                                     int(prop_y2 * original_height)

        end = time.time()
        cc_crop_time = round(end - start, 3)
        return (scaled_x1, scaled_y1, scaled_x2, scaled_y2), best_image, cc_crop_time

    def high_precision_cc_crop(self, input_img, brighten_value=0):
        """
        Attempts to crop a crc partition with a high degree of precision.
        """
        h, w = input_img.shape[0:2]
        area = h*w
        min_crop_area = area // 250
        max_crop_area = area // 4
        
        
        input_img[input_img < 155] += brighten_value
        
        # identify squares in the crop
        squares = ColorchipRead.find_squares(input_img,
                                             contour_area_floor=min_crop_area,
                                             contour_area_ceiling=max_crop_area,
                                             leap=1,
                                             k_size=0)
        # reduce squares to the unique ones
        squares = [i for i in np.unique(squares, axis=0)]
        # reduce squares to the ... squary ones...
        squery_squares = []
        for square in squares:
            x_arr = square[..., 0]
            y_arr = square[..., 1]
            x1, y1, x2, y2 = np.min(x_arr), np.min(y_arr), np.max(x_arr), np.max(y_arr)
            square_width, square_height = x2 - x1, y2 - y1
            longest_side = max(square_width, square_height)
            shortest_side = min(square_width, square_height)
            ratio = longest_side / shortest_side
            if 0.70 < ratio < 1.30:
                squery_squares.append(square)

        if len(squares) > 5:
            squares =  np.array(squery_squares)
            # generate a list of areas for each square
            areas = [cv2.contourArea(x) for x in squares]
            # modified from: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            d = np.abs(areas - np.median(areas))
            mdev = np.median(d)
            s = d/mdev if mdev else 0.
            # note, the s<2 is square area standard deviations from mean
            squares = squares[s<3]
    
            x1 = np.min(squares[...,0])
            y1 = np.min(squares[...,1])
            x2 = np.max(squares[...,0])
            y2 = np.max(squares[...,1])
            # attempt to expand a touch for future scale det
            x1 = max(x1-5, 0)
            x2 = min(x2+5, w)
            y1 = max(y1-5, 0)
            y2 = min(y2+5, h)
            best_square = [x1, y1, x2, y2]
            best_cropped_img = input_img[y1:y2, x1:x2]
        else:
            # if it failed, try once more but force brightness
            if brighten_value == 0:
                print("trying HP crop with brigher values")
                return self.high_precision_cc_crop(input_img, brighten_value= 80)
            # if no squares were found, just return what was passed in
            print("FAILED high precision crop")
            best_square = [0, 0, w, h]
            best_cropped_img = input_img

        return best_cropped_img, best_square

    @staticmethod
    def predict_color_chip_quadrant(original_size, scaled_crop_location):
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
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_width = original_width / 2
        half_height = original_height / 2

        if cx > half_width and cy < half_height:
            return 1
        elif cx < half_width and cy < half_height:
            return 2
        elif cx < half_width and cy > half_height:
            return 3
        elif cx > half_width and cy > half_height:
            return 4
        else:
            return None

    @staticmethod
    def predict_color_chip_whitevals(cropped_cc, crc_type, seed_pt=None):
        """
        Takes a cropped CC image and determines the average RGB values of the
        whitest portion.
        :param cropped_cc: The cropped color chip image
        :type cropped_cc: Image
        :return: Returns a list of the averaged whitest values
        :rtype: list
        """
        # determine a few parameters which will not change
        h, w, chn = cropped_cc.shape
        area = h * w
        contour_area_floor = area // 200
        contour_area_ceiling = area // 8

        filtered_img = cropped_cc.copy()
        # bump up the dimmer end of the colors
        filtered_img[filtered_img < 155] += 25
        # for small partitions subject to mixed pixels, 
        # we can afford the time to apply a bilateral filter.
        if area < 23000: # partition size of 150 has area of 22500
            filter_sigma = contour_area_floor
            filtered_img = cv2.bilateralFilter(filtered_img,
                                               0,
                                               filter_sigma,
                                               filter_sigma * 2)

        grayImg = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
        cv2.normalize(grayImg, grayImg, 0, 255, cv2.NORM_MINMAX)

        # determine a few parameters which will not change
        minVal = np.min(grayImg)
        maxVal = np.max(grayImg)
        var_threshold = int((maxVal - minVal) * .1)

        if seed_pt:
            # if we have a seed point don't fret over if it is a "good one"
            mask = np.zeros((h+2,w+2),np.uint8)
            floodflags = 8
            floodflags |= cv2.FLOODFILL_FIXED_RANGE
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            floodflags |= (int(maxVal) << 8)
            num,_,mask,rect = cv2.floodFill(grayImg, mask, seed_pt, 0,
                                            (var_threshold,)*3, #low diff
                                            (var_threshold,)*3, #hi diff
                                            floodflags)
            # correct for mask expansion
            mask = mask[1:-1, 1:-1, ...]
        else:
            # when we don't have a seed point find a "good one"
            for i in range(100):
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
                mask = np.zeros((h+2,w+2),np.uint8)
                floodflags = 8
                floodflags |= cv2.FLOODFILL_FIXED_RANGE
                floodflags |= cv2.FLOODFILL_MASK_ONLY
                floodflags |= (int(maxVal) << 8)
                num,_,mask,rect = cv2.floodFill(grayImg,
                                                mask, maxLoc, 0,
                                                (var_threshold,)*3, #low diff
                                                (var_threshold,)*3, #hi diff
                                                floodflags)
                mask = mask[1:-1, 1:-1, ...]
                squares = ColorchipRead.find_squares(mask,
                                                     contour_area_floor=contour_area_floor,
                                                     contour_area_ceiling=contour_area_ceiling,
                                                     leap = 3)
                if len(squares) == 0:
                    # redact brighter values which are not "squarey"
                    grayImg[np.where(mask > 0)] = 0
                    continue

                squares = sorted(squares, key=cv2.contourArea, reverse=True)
                for square in squares:
                    x_arr = square[..., 0]
                    y_arr = square[..., 1]
                    x1, y1, x2, y2 = np.min(x_arr), np.min(y_arr), np.max(x_arr), np.max(y_arr)
                    square_width, square_height = x2 - x1, y2 - y1
                    longest_side = max(square_width, square_height)
                    shortest_side = min(square_width, square_height)
                    ratio = longest_side / shortest_side
    
                    if crc_type == 'Tiffen / Kodak Q-13  (8")':
                        # Longest side is roughly 1.6x longer than shortest side
                        if 1.35 < ratio < 1.85:
                            break
                    else:
                        if 0.75 < ratio < 1.25:
                            break
                else:
                    # redact brighter values which are not "squarey"
                    grayImg[np.where(mask > 0)] = 0
                    continue
                break
            else:
                raise SquareFindingFailed
        extracted = cropped_cc[mask != 0]

        # annotate the detected point for preview window
        squares = ColorchipRead.find_squares(mask,
                                             contour_area_floor=contour_area_floor,
                                             contour_area_ceiling=contour_area_ceiling,
                                             leap = 17)
        square = sorted(squares, key=cv2.contourArea, reverse=True)
        cv2.drawContours(cropped_cc, square, 0, (0,255,0), 1)
        extracted = extracted.reshape(-1,extracted.shape[-1])

        #mode_white = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=extracted)
        # median was selected as preferable over mode
        median_white = np.median(extracted, axis=0)
        return list(median_white), minVal


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
