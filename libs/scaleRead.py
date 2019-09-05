"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""

import cv2
import numpy as np
from itertools import cycle


class ScaleRead:

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
                    if len(cnt) == 4 and contour_area_floor < contourArea < contour_area_ceiling \
                            and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([ScaleRead.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                                          for i in range(4)])
                        if max_cos < 0.1:
                            squares.append(cnt)
        return squares

    @staticmethod
    def find_scale(im, patch_mm_size=3.17):
        '''
        Finds the pixels to millimeter of image using the CRC patch size. Currently only works ISA ColorGauge Target
        CRCs (micro, nano, and pico versions)
        :param im: Image to be determined
        :type im: CV2.Image
        :param patch_mm_size: the expected patchsize
        :return: Returns the pixels per millimeter *average* rounded to the nearest whole number
        '''
        # Converting to HSV as it performs better during floodfill
        #im = im.convert("HSV")
        #im = np.array(im)
        #cv2.imwrite('crc.jpg', im)
        #xcorrection, ycorrection = correction # correction=(x, y)

        # HSV, calibrated for saturated (already processed) images
        # desired_colors = [[20, 204, 247],
        #                   [62, 120, 135],
        #                   [135, 222, 196],
        #                   [2, 153, 204],
        #                   [226, 138, 219]]

        # HSV Colors, desaturated for RAW image processing
#        desired_colors = [[18, 135, 120],
#                          [77, 64, 92],
#                          [162, 153, 92],
#                          [2, 122, 85],
#                          [149, 125, 117]]
#
#        pixels_per_mm = []
        #for desired_color in desired_colors:
#            h, w, chn = im.shape
#
#            # Function is used in order to break out of the entire nested loop.
#            def find_seed():
#                for hi, height in enumerate(im):
#                    for wi, width in enumerate(height):
#                        if desired_color[0] - thresh < width[0] < desired_color[0] + thresh and \
#                                desired_color[1] - thresh * 2 < width[1] < desired_color[1] + thresh * 2 and \
#                                desired_color[2] - thresh * 2 < width[2] < desired_color[2] + thresh * 2:
#                            return wi, hi
#                else:
#                    return None
#
#            seed = find_seed()
        # convert the input to HSV
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        h, w = im.shape[0:2]
        # calculate a series of seed positions to start floodfilling
        sample_hs = np.linspace(0, h, 10).astype(int)[2:-2]
        anchor_hs = np.tile([h//10, h - (h//10)], 3)

        sample_ws = np.linspace(0, w, 10).astype(int)[2:-2]
        anchor_ws = np.tile([w - (w//10), w//10], 3)

        h_seeds = tuple(zip(anchor_ws, sample_hs))
        w_seeds = tuple(zip(sample_ws, anchor_hs))
        # the result is a series of staggered points along border of the image.
        seed_pts = h_seeds + w_seeds
        # determine reasonable area limits for squares
        area = h*w
        contour_area_floor = area // 50
        contour_area_ceiling = area // 4
        # determine reasonable lower, upper thresholds
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(cv2.cvtColor(im,cv2.COLOR_RGB2LAB)[..., 0])
        var_threshold = int((maxVal-minVal) * .05)
        # container to hold the results from each seed
        pixels_per_mm = []
        orig_mask = np.zeros((h + 2, w + 2), np.uint8)
        for i, seed in enumerate(seed_pts):

            mask = orig_mask.copy()
            floodflags = 4
            floodflags |= (255 << 8)
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            	#retval, image, mask, rect	
            _, _, mask, rect = cv2.floodFill(im, mask, seed, (255,0,0),
                                             (var_threshold,)*3,
                                             (var_threshold,)*3, floodflags)

            # useful for debugging odd scal values generation
            #cv2.imwrite(f'{i}_mask.jpg', mask)

            squares = ScaleRead.find_squares(mask,
                                contour_area_floor=contour_area_floor,
                                contour_area_ceiling=contour_area_ceiling)

            if len(squares) > 0:
                biggest_square = max(squares, key=cv2.contourArea)
                xs = biggest_square[..., 0]
                ys = biggest_square[..., 1]
                square_width = (max(xs) - min(xs)) #* ycorrection
                square_height = (max(ys) - min(ys))# * xcorrection
                pixels_per_mm.append((((square_width + square_height) / 2) / patch_mm_size))
            # It appears the bounding postion is inclusive to the object.
            # adding 1px to each side of h and w (+2) should correct this.
            #x,y,w,h = rect
            pixels_per_mm.append((((rect[2]+2 + rect[3]+2) / 2) / patch_mm_size))
        # require a minimum measurements before proceeding
        if len(pixels_per_mm) > 9:
            outlier_thresh = np.std(pixels_per_mm) * 1#.5
            pixel_mean = np.mean(pixels_per_mm)
            lower_bounds = pixel_mean - outlier_thresh
            upper_bounds = pixel_mean + outlier_thresh
            pixels_per_mm = [x for x in pixels_per_mm if lower_bounds< x < upper_bounds]

            pixels_per_mm_avg = round(np.mean(pixels_per_mm), 1)
            ppm_uncertainty = max( round(np.max(pixels_per_mm) - np.min(pixels_per_mm), 2), 1)
        else:
            pixels_per_mm_avg = 0
            ppm_uncertainty = max(h,w)  # be really sure they get the point.

        return pixels_per_mm_avg, ppm_uncertainty
