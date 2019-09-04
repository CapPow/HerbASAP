"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""

import cv2
import numpy as np


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
    def find_scale(im):
        '''
        Currently only works for nano CRCs
        :param im: Image to be determined
        :type im: PIL.Image
        :param scale: scale of the original image to the processing size (1250, 1875)
        :return: Returns the pixels per millimeter *average*
        '''
        im = im.convert("HSV")
        im = np.array(im)
        thresh = 5

        # RGB Colors (here for reference)
        # desired_colors = [[247, 142, 50],
        #                   [105, 135, 71],
        #                   [25, 165, 195],
        #                   [204, 86, 82],
        #                   [220, 102, 182]]

        # HSV
        # desired_colors = [[20, 204, 247],
        #                   [62, 120, 135],
        #                   [135, 222, 196],
        #                   [2, 153, 204],
        #                   [226, 138, 219]]

        # HSV Colors, desaturated for RAW image processing
        desired_colors = [[18, 135, 120],
                          [77, 64, 92],
                          [162, 153, 92],
                          [2, 122, 85],
                          [149, 125, 117]]

        pixels_per_mm = []
        for desired_color in desired_colors:
            seed = None
            h, w, chn = im.shape

            for hi, height in enumerate(im):
                for wi, width in enumerate(height):
                    if desired_color[0] - thresh < width[0] < desired_color[0] + thresh and \
                            desired_color[1] - thresh * 2 < width[1] < desired_color[1] + thresh * 2 and \
                            desired_color[2] - thresh * 2 < width[2] < desired_color[2] + thresh * 2:
                        seed = (wi, hi)
                        break

            mask = np.zeros((h + 2, w + 2), np.uint8)

            floodflags = 4
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            floodflags |= (255 << 8)

            _, _, _, _ = cv2.floodFill(im, mask, seed, (255, 0, 0), (15 - thresh, 15 - thresh * 2, 15 - thresh * 2),
                                                (15 + thresh, 15 + thresh * 2, 15 + thresh * 2), floodflags)
            # pixels_per_mm.append((((rect[2] + rect[3]) / 2) / 3.17))
            np_best_image = np.array(mask)

            cv_best_image = cv2.cvtColor(np_best_image, cv2.COLOR_GRAY2RGB)
            cv_best_image = cv2.cvtColor(cv_best_image, cv2.COLOR_RGB2HSV)
            squares = ScaleRead.find_squares(cv_best_image, contour_area_floor=100, contour_area_ceiling=10000)

            if len(squares) > 0:
                biggest_square = max(squares, key=cv2.contourArea)
                xs = biggest_square[..., 0]
                ys = biggest_square[..., 1]
                length = max(xs) - min(xs)
                length2 = max(ys) - min(ys)
                pixels_per_mm.append((((length + length2) / 2) / 3.17))
                cv2.drawContours(im, [biggest_square], -1, (0, 255, 0), 1)

        if len(pixels_per_mm) > 0:
            # print(pixels_per_mm)
            pixels_per_mm_avg = sum(pixels_per_mm) / len(pixels_per_mm)
        else:
            pixels_per_mm_avg = 0
        # print(f"Pixels per millimeter: {round(pixels_per_mm_avg)}")

        # Code for returning back the image, if wanted/needed.
        # im = Image.fromarray(im)
        # im = im.convert('RGB')
        # im.show()
        # im.save(f"scale_cc_test_images/results/{os.path.basename(fp)}-{pixels_per_mm_avg}.jpg")
        return pixels_per_mm_avg
