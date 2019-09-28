"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""

import cv2
import numpy as np


class ScaleRead:

    def __init__(self, parent=None, *args):
        super(ScaleRead, self).__init__()
        ###
        # setup the scale parameter lookup dict. Organized as color checker
        # names as keys and a tuple of (area in mm, seed det! function)
        ###
        # Tiffen Q-13 scale was manually measured from images
        # X-Rite ColorChecker Passport was manually measured from images
        # X-Rite Colorchecker Classic was manually measured from images
        ###
        self.scale_params = {
                'CameraTrax 24 ColorCard (2" x 3")':(115.5625, self.det_large_crc_seeds, True),
                'ISA ColorGauge Nano':(10.0489, self.det_isa_nano_seeds, False),
                #'ISA ColorGauge Nano':(10.0489, self.det_large_crc_seeds, False),
                'Tiffen / Kodak Q-13  (8")':(386.6379, self.det_large_crc_seeds, False),
                'X-Rite ColorChecker Passport':(164.3652, self.det_large_crc_seeds, True),
                'X-Rite ColorChecker Classic':(361.00, self.det_large_crc_seeds, True)
                }

    @staticmethod
    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    @staticmethod
    def find_squares(img, contour_area_floor=850, contour_area_ceiling=100000, leap=6):
        # taken from: opencv samples
        img = cv2.GaussianBlur(img, (7, 7), 0)
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
                    if len(cnt) == 4 and (contour_area_floor < contourArea < contour_area_ceiling) \
                            and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([ScaleRead.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                                          for i in range(4)])
                        if max_cos < 0.1:
                            squares.append(cnt)
        return squares

    @staticmethod
    def high_precision_cc_crop(input_img):

        h, w = input_img.shape[0:2]
        area = h*w
        min_crop_area = area // 10
        max_crop_area = area // 1.2

        # identify squares in the crop
        squares = ScaleRead.find_squares(input_img,
                                         contour_area_floor=min_crop_area,
                                         contour_area_ceiling=max_crop_area)

        # if, somehow no proper squares were identified, return the entire img
        if len(squares) < 1:
            return input_img
        # identify the largest area among contours
        biggest_square = max(squares, key=cv2.contourArea)
        x_arr = biggest_square[..., 0]
        y_arr = biggest_square[..., 1]
        x1, y1, x2, y2 = np.min(x_arr), np.min(y_arr), np.max(x_arr), np.max(y_arr)
        biggest_square = (x1, y1, x2, y2)
        cropped_img = input_img[y1:y2, x1:x2]
        return cropped_img

    @staticmethod
    def det_large_crc_seeds(h, w, pts=24):
        """
        Generalized seed determination method which choosed randomly positions
        without replacement.
        """
        # determine how to split the points among x / y axes
        h_w_combined = h+w
        if h < w:
            h_ratio = h / h_w_combined
            h_pts = max(int(pts * h_ratio), 2)
            w_pts = pts - h_pts
            # determine the points to sample along crop img width
            w_samples = np.linspace(0, w, w_pts+2).astype(int)[1:-1]
            # determine the points to sample along crop img height
            h_anchors = np.linspace(0, h, h_pts+2).astype(int)[1:-1]
            # repeate the given points for each
            h_anchors = np.tile(h_anchors, len(w_samples)//len(h_anchors))
            seed_pts = tuple(zip(w_samples, h_anchors))
        else:
            w_ratio = w / h_w_combined
            w_pts = max(int(pts * w_ratio), 2)
            h_pts = pts - w_pts
            # determine the points to sample along crop img width
            h_samples = np.linspace(0, w, h_pts+2).astype(int)[1:-1]
            # determine the points to sample along crop img height
            w_anchors = np.linspace(0, h, w_pts+2).astype(int)[1:-1]
            # repeate the given points for each
            w_anchors = np.tile(w_anchors, len(h_samples)//len(w_anchors))
            seed_pts = tuple(zip(h_samples, w_anchors))

        return seed_pts

    @staticmethod
    def det_isa_nano_seeds(h, w, pts=12):
        # establish a no-sample buffer zone
        h_buffer = h//10
        w_buffer = w//10
        #pts are the number of seed points to take per axis
        sample_hs = np.linspace(0, h, pts+2).astype(int)[1:-1]
        anchor_hs = np.tile([h_buffer, h - (h_buffer)], len(sample_hs)//2)
        sample_ws = np.linspace(0, w, pts+2).astype(int)[1:-1]
        anchor_ws = np.tile([w - (w_buffer), w_buffer], len(sample_ws)//2)
        h_seeds = tuple(zip(anchor_ws, sample_hs))
        w_seeds = tuple(zip(sample_ws, anchor_hs))
        # the result is a series of staggered points along border of the image.
        seed_pts = h_seeds + w_seeds
        
        return seed_pts


    @staticmethod
    def find_scale(im, patch_mm_area, seed_func, to_crop, retry=True):
        '''
        Finds the pixels to millimeter of image using the CRC patch size. Currently only works ISA ColorGauge Target
        CRCs (micro, nano, and pico versions)
        :param im: Image to be determined
        :type im: CV2.Image
        :param patch_mm_area: the expected patch w * h
        :param seed_func: the function to use for seed position determination.
        :type seed_func: a scaleRead seed method
        :param to_crop: Boolean condition if a high precision crop is called for.
        :type to_crop: bool
        :return: Returns the rounded pixels per millimeter and 95% CI.
        '''
        # Converting to HSV as it performs better during floodfill
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # if the color reference card benefits from a high precision crop
        # in the Q-13s, the partial crop detection is too aggressive after high
        # precision cropping.
        if to_crop:
            im = ScaleRead.high_precision_cc_crop(im)
        h, w = im.shape[0:2]
        # calculate a series of seed positions to start floodfilling
        seed_pts = seed_func(h, w)
        # determine reasonable area limits for squares
        area = h*w
        contour_area_floor = area // 50
        contour_area_ceiling = area // 10

        #determine reasonable lower, upper thresholds, 5% of lum range
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0])
        var_threshold = int((maxVal-minVal) * 0.05)
        # container to hold the results from each seed's floodfill
        pixels_per_mm = []
        orig_mask = np.zeros((h + 2, w + 2), np.uint8)
        for i, seed in enumerate(seed_pts):
            mask = orig_mask.copy()
            floodflags = 4
            floodflags |= (255 << 8)
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            # retval, image, mask, rect
            _, _, mask, rect = cv2.floodFill(im, mask, seed, (255,0,0),
                                             (var_threshold, )*3,
                                             (var_threshold, )*3, floodflags)

            x1, y1, patch_w, patch_h = rect
            patch_hw = [patch_w, patch_h]

            aspect_condition = (max(patch_hw) / min(patch_hw)) > 2
            area_condition = not (contour_area_floor < (patch_w * patch_h) < contour_area_ceiling)
            if aspect_condition or area_condition:
                continue
            # if any of the mask is along the img edge, assume it is incomplete
            # and omit the results of this seed from analysis.
            mask_edge_vectors = [
                    mask[2, :],   # first row adjusting for mask expansion
                    mask[-3, :],  # last row adjusting for mask expansion
                    mask[:, 2],  # first col adjusting for mask expansion
                    mask[:, -3]   # last col adjusting for mask expansion
                    ]
            if any([255 in x for x in mask_edge_vectors]):
                #print(f"Continuing, patch number {i}")
                continue

            squares = ScaleRead.find_squares(mask,
                                contour_area_floor=contour_area_floor,
                                contour_area_ceiling=contour_area_ceiling)

            if len(squares) > 0:
                biggest_square = max(squares, key=cv2.contourArea)
                square_area = cv2.contourArea(biggest_square)
                pixels_per_mm.append( np.sqrt(square_area / patch_mm_area))

            # Option to use the rectangles returned from floodfill operation
            #x,y,w,h = rect
            rect_area = np.sqrt(((patch_w+2) * (patch_h+2)) / patch_mm_area)
            pixels_per_mm.append(rect_area)

            # useful for debugging odd scal values generation
            #cv2.imwrite(f'{i}_mask.jpg', mask)

        # require a minimum qty of measurements before proceeding
        if len(pixels_per_mm) > 5:
            ppm_std = np.std(pixels_per_mm)
            pixel_mean = np.mean(pixels_per_mm)
            lower_bounds = pixel_mean - ppm_std # keep results within +/- 1 std
            upper_bounds = pixel_mean + ppm_std
            pixels_per_mm = [x for x in pixels_per_mm if lower_bounds< x < upper_bounds]
            # determine CI @ 95% : 1.96 SE = std / sqrt(len(array))
            ppm_uncertainty = round(1.96 * (np.std(pixels_per_mm)/ np.sqrt(len(pixels_per_mm))), 2)
            pixels_per_mm_avg = round(np.mean(pixels_per_mm), 2)
        elif retry:
            print('RETRYING')
            # if no scale det! try Lum expansion
            # based on: https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)  # convert from BGR to LAB color space
            l, a, b = cv2.split(lab)  # split on 3 different channels
            l2 = clahe.apply(l)  # apply CLAHE to the L-channel
            lab = cv2.merge((l2,a,b))  # merge channels
            im = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to BGR
            # retry param avoids infinate recursion
            # if not using ...large_crc_seeds, it is safe trying large method
            # yet vise versa is untrue, explicitly use det_large_crc_seeds
            return ScaleRead.find_scale(im, patch_mm_area, ScaleRead.det_large_crc_seeds, to_crop, retry=False)
        else:
            pixels_per_mm_avg = 0
            ppm_uncertainty = max(h,w)  # be really sure they get the point.

        return pixels_per_mm_avg, ppm_uncertainty
