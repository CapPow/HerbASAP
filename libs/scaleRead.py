import cv2
import numpy as np
from PIL import Image
from scipy import stats
import glob


class ScaleRead:

    @staticmethod
    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    @staticmethod
    def find_squares(img):
        squares = []
        for gray in cv2.split(img):
            for thrs in range(0, 255, 6):
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 150, apertureSize=7)
                    bin = cv2.dilate(bin, None)
                else:
                    _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_len = cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                    contourArea = cv2.contourArea(cnt)
                    if len(cnt) == 4 and 5 < contourArea < 10000 \
                            and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                                          for i in range(4)])
                        if max_cos < 0.1:
                            squares.append(cnt)
        return squares

    @staticmethod
    def find_scale(image):
        img = cv2.imread(image)
        height, width, _ = img.shape
        squares = ScaleRead.find_squares(img)
        points = []

        for square in squares:
            M = cv2.moments(square)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if height > width:
                points.append(cY)
            elif width < height:
                points.append(cX)

        distances = []
        for point in points:
            lowest_distance = float('inf')
            for comparison_point in points:
                distance = abs(point - comparison_point)
                if 7 < distance < lowest_distance:
                    lowest_distance = distance
            distances.append(lowest_distance)

        cv2.drawContours(img, squares, -1, (0, 255, 0), 1)
        ch = cv2.waitKey()

        mode = stats.mode(distances)
        return mode.mode