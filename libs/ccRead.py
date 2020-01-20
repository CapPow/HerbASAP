#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""
import numpy as np
from PIL import Image
import cv2
import time
import os

# Importing TensorFlow within debug messages
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        default_message = 'Discriminator could not find the best image. Try lowering the prediction floor value.'
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
        self.position_model = tf.lite.Interpreter(model_path="libs/models/mlp_proposal.tflite")
        self.position_model.allocate_tensors()
        self.position_input_details = self.position_model.get_input_details()
        self.position_output_details = self.position_model.get_output_details()
        self.K_position_model = load_model("libs/models/mlp_proposal_k.hdf5")
        self.position_function = K.function([self.K_position_model.layers[0].input,
                                             self.K_position_model.layers[1].input,
                                             K.learning_phase()],
        [self.K_position_model.layers[-1].output])

        self.discriminator_model = tf.lite.Interpreter(model_path="libs/models/discriminator.tflite")
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
            raise InvalidStride

        return hists_rgb, hists_hsv, possible_positions

    def _square_first_regions(self):
        pass

    def process_colorchip_small(self, im, original_size, stride_style='whole',
                                stride=25, partition_size=125, discriminator_floor=0.90,
                                over_crop=1, hard_cut_value=50, high_precision=True):
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
        :return: Returns a tuple containing the bounding box (x1, y1, x2, y2) and the cropped image of the colorchip.
        :rtype: tuple
        """

        nim = im
        im = self.ocv_to_pil(im)
        im_hsv = im.convert("HSV")
        whole_extrema = im_hsv.getextrema()
        whole_extrema = whole_extrema[1][1]
        start = time.time()
        image_width, image_height = im.size
        original_width, original_height = original_size
        cv_image = cv2.cvtColor(nim, cv2.COLOR_RGB2HSV)
        partition_area = partition_size * partition_size
        contour_area_floor = partition_area // 10
        contour_area_ceiling = partition_area // 0.5
        squares = self.find_squares(cv_image,
                                    contour_area_floor=contour_area_floor,
                                    contour_area_ceiling=contour_area_ceiling,
                                    leap=17)

        squares = np.array(squares)

        discriminator_model = self.discriminator_model

        part_im = []
        possible_positions = []

        for square in squares:
            M = cv2.moments(square)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            location = (cX - (partition_size // 2), cY - (partition_size // 2), cX + (partition_size // 2), cY + (partition_size // 2))
            part_image = im.crop(location)
            part_image = part_image.resize((125, 125))
            extrema = part_image.convert("HSV").getextrema()
            extrema = extrema[1][1]
            if whole_extrema - hard_cut_value < extrema:
                part_im.append(part_image)
                possible_positions.append(location)

        if len(part_im) != 0:
           inference_type = 'find_squares'
           hists_rgb = []
           hists_hsv = []
           for im in part_im:
               im_hsv = im.convert("HSV")
               hists_rgb.append(im.histogram())
               hists_hsv.append(im_hsv.histogram())

        else:
            inference_type = 'legacy'
            hists_rgb, hists_hsv, possible_positions = self._legacy_regions(im=im, im_hsv=im_hsv,
                                                                            image_width=image_width,
                                                                            image_height=image_height,
                                                                            whole_extrema=whole_extrema,
                                                                            stride_style=stride_style, stride=stride,
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
                if position_prediction < (max_pred - 0.001):
                    del position_prediction
                    del position_uncertainty[_j]
                    del indices[_j]
            except IndexError:
                break

        highest_prob_images = []
        highest_prob_positions = []

        if inference_type == 'find_squares':
            highest_prob_images = [np.array(part_im[k]) for k in indices]
            highest_prob_positions = [possible_positions[k] for k in indices]
        else:
            for i in indices:
                #im.crop(possible_positions[indices[i]]).show()
                highest_prob_images.append(np.array(im.crop(possible_positions[i]).resize((125, 125))))
                highest_prob_positions.append(possible_positions[i])

        highest_prob_images_pred = np.array(highest_prob_images, dtype=np.float32) / 255

        if inference_type == 'find_squares':
            best_image = Image.fromarray(highest_prob_images[0])
            best_location = highest_prob_positions[0]
        else:
            for i, highest_prob_image in enumerate(highest_prob_images):
                discriminator_model.set_tensor(discriminator_model.get_input_details()[0]['index'], [highest_prob_images_pred[i]])
                discriminator_model.invoke()
                disc_value = discriminator_model.get_tensor(discriminator_model.get_output_details()[0]['index'])[0][1]
                #print(disc_value)
                if disc_value > discriminator_floor:
                    # print(f"Discriminator took {i} predictions before finding the colorchip.")
                    best_image = Image.fromarray(highest_prob_image)
                    best_location = highest_prob_positions[i]
                    break
            else:
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
        # print(f"Color chip cropping took: {end - start} seconds using {inference_type} proposal.")

        cc_crop_time = round(end - start, 3)
        print(f"Inference type: {inference_type}")
        return (scaled_x1, scaled_y1, scaled_x2, scaled_y2), best_image, cc_crop_time

    def high_precision_cc_crop(self, input_img):
        """
        Attempts to crop a crc partition with a high degree of precision.
        """
        h, w = input_img.shape[0:2]
        area = h*w
        min_crop_area = area // 3
        max_crop_area = area // 1.01

        # identify squares in the crop
        squares = ColorchipRead.find_squares(input_img,
                                             leap=1,
                                             contour_area_floor=min_crop_area,
                                             contour_area_ceiling=max_crop_area)
        disc_model = self.discriminator_model
        discriminator_thresh = 0.999
        best_cropped_img = input_img
        # identify the largest area among contours
        qualified_squares = []
        for i, current_square_contour in enumerate(squares):
            x_arr = current_square_contour[..., 0]
            y_arr = current_square_contour[..., 1]
            x1, y1, x2, y2 = np.min(x_arr), np.min(y_arr), np.max(x_arr), np.max(y_arr)
            current_square = (x1, y1, x2, y2)
            cropped_img = input_img[y1:y2, x1:x2]
            img = np.zeros((125, 125, 3), dtype=np.float32)
            cropped_img_t = np.array(cropped_img, dtype=np.float32) / 255
            img[0:cropped_img_t.shape[0], 0:cropped_img_t.shape[1]] = cropped_img
            disc_model.set_tensor(disc_model.get_input_details()[0]['index'],
                                                [img])
            disc_model.invoke()
            disc_value = disc_model.get_tensor(disc_model.get_output_details()[0]['index'])[0][1]
            if disc_value > discriminator_thresh:
                qualified_squares.append(current_square)
        try:
            qualified_squares = sorted(qualified_squares, key=lambda x: (x[2] - x[0]) *(x[3] - x[1]), reverse=False)
            best_square = qualified_squares[0]
            # loosen the crops only slightly for future scale det.
            x1, y1, x2, y2 = best_square
            x1 = max(x1-2, 0)
            x2 = min(x2+2, w)
            y1 = max(y1-2, 0)
            y2 = min(y2+2, h)
            best_cropped_img = input_img[y1:y2, x1:x2]
        except IndexError:
            # if failed to find a better crop use the whole image
            best_cropped_img = input_img
            best_square = (0, 0, w, h)

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

        # print(f"x1: {x1} | y1: {y1} | x2: {x2} | y2: {y2}")
        # print(f"cx: {cx} | cy: {cy} | hw: {half_width} | hh: {half_height}")

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

        #cropped_cc = cropped_cc.copy()
        #grayImg = cropped_cc.sum(axis=2)
        #grayImg = cv2.cvtColor(cropped_cc, cv2.COLOR_RGB2GRAY)
        grayImg = cv2.cvtColor(cropped_cc, cv2.COLOR_RGB2LAB)[...,0]
        #cv2.minMaxLoc(cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0])
        for _ in range(300):
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
            var_threshold = int((maxVal - minVal) * .1)
            h, w, chn = cropped_cc.shape
            if seed_pt:
                seed = seed_pt
            else:
                seed = maxLoc
            mask = np.zeros((h+2,w+2),np.uint8)
            floodflags = 8
            floodflags |= cv2.FLOODFILL_FIXED_RANGE
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            floodflags |= (int(maxVal) << 8)
            num,cropped_cc,mask,rect = cv2.floodFill(cropped_cc, mask, seed,
                                                     0,
                                                     (var_threshold,)*3,
                                                     (var_threshold,)*3,
                                                     floodflags)
            mask = mask[1:-1, 1:-1, ...]
            area = h * w
            contour_area_floor = area // 350
            contour_area_ceiling = area // 5
            # use a leap that assures few steps
            squares = ColorchipRead.find_squares(mask,
                                                 contour_area_floor=contour_area_floor,
                                                 contour_area_ceiling=contour_area_ceiling,
                                                 leap = 85)
            if len(squares) == 0:
                # redact brighter values which are not "squarey"
                badMin = np.min(grayImg[np.where(mask != 0)])
                grayImg[np.where(grayImg >= badMin)] = 0
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
                    if 1.40 < ratio < 1.80:
                        break
                else:
                    if 0.80 < ratio < 1.20:
                        break
            else:
                # redact brighter values which are not "squarey"
                #badMin = np.min(grayImg[np.where(mask != 0)])
                #grayImg[np.where(grayImg >= badMin)] = 0
                grayImg[np.where(mask > 0)] = 0
                print('Cleaned up bright points')
                continue
            break
        else:
             raise SquareFindingFailed

        extracted = cropped_cc[mask != 0]
        # annotate the detected point for preview window
        squares = ColorchipRead.find_squares(mask,
                                             contour_area_floor=contour_area_floor,
                                             contour_area_ceiling=contour_area_ceiling,
                                             leap = 85)
        square = sorted(squares, key=cv2.contourArea, reverse=True)
        cv2.drawContours(cropped_cc, square, 0, (0,255,0), 1)
        extracted = extracted.reshape(-1,extracted.shape[-1])
        mode_white = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=extracted)

        return list(mode_white), minVal


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
