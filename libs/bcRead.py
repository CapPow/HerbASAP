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
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""

import re
import numpy as np
import cv2

from math import cos as math_cos
from math import sin as math_sin
from math import radians
import site
from shutil import copyfile
from platform import system
from pylibdmtx.pylibdmtx import decode as libdmtx_decode
# import the pyzbar fork (local)
from .deps.pyzbar.pyzbar import decode as zbar_decode

###
# Developer note: the libraries: re, pyzbar, and pylibdmtx all have a "decode"
# method which are used in this class. This can cause difficult to debug issues
###
class bcRead():
    """A barcode reader class

    Args:

    patterns (str): A string of uncompiled, "|" concatenated regex patterns.

    backend (str): Either "zbar" or "libdmtx", to determine which libarary 
        should be used for decoding. Default is 'zbar.'

    rotation_list (iterable, optional): iterable containing a series of int
        representing image rotations (in degrees) to attempt if no barcode is 
        found. Default values are [9, 25, 18]. Rotation attempts stop after any
        results are found. The list's rotations are cumulative. Short or empty
        lists will decrease the time before giving up on finding a barcode.

    Attributes:
    rePattern (obj): A compiled regex pattern
    backend (str): a string to determine which decoder was imported.
    rotation_list (list): The saved rotation list
    """

    def __init__(self, patterns, backend='zbar',
                 rotation_list=[9, 25, 18], parent=None, *args):
        super(bcRead, self).__init__()
        self.parent = parent
        self.compileRegexPattern(patterns)
        # This might need promoted to a user preference in mainwindow
        self.rotation_list = rotation_list
        self.backend = backend

    def decode_zbar(self, im):
        return zbar_decode(im)

    def decode_libdmtx(self, im):
        return libdmtx_decode(im, timeout=1500)

    def set_backend(self, backend='zbar'):
        """
        Sets which libarary should be used for decoding. Default is 'zbar.'

        :param backend: string either 'zbar' or 'libdmtx' libdmtx is useful for
        datamatrix decoding.
        :type backend: str
        :return:
        """
        self.backend = backend
        if backend == 'zbar':
            self.decode = self.decode_zbar
        elif backend == 'libdmtx':
            self.decode = self.decode_libdmtx

    def compileRegexPattern(self, patterns):
        """ compiles a collection specific regex pattern """
        #  assume an empty pattern is a confused user, match everything.
        if patterns == '':
            patterns = '^(.*)'
        try:
            rePattern = re.compile(patterns)
            self.rePattern = rePattern
        except re.error:
            raise

    def decodeBC(self, img, verifyPattern=True, return_details=False):
        """ attempts to decode barcodes from an image array object.

        Given a np array image object (img), decodes BCs and returns those
        which match self.rePattern

        verifies if the bcData matches the compiled rePattern.

        Args:
        img (numpy.ndarray): a numpy image array object

        return_details (bool, optional): default = False. Whether or not to
            return the barcode(s) bounding box coordinate(s) and format(s) in
            addition to the barcode value(s). Return a list of dictionaries.

        Returns (list): a list of matched barcode value(s) found in the image.
            If return_details = True, then returns a list of dictionaries.
        """
        # the complete output from pyzbar which matches checkPattern
        backend = self.backend
        if backend == 'zbar':
            #code_reader = self.decode_zbar
            code_reader = self.extract_by_squares
        elif backend == 'libdmtx':
            code_reader = self.decode_libdmtx
        # decode each code found from bytes to utf-8
        bcRawData = [x.data.decode('utf-8') for x in code_reader(img)]
        if verifyPattern:  # limit the results to those matching rePattern
            bcRawData = [x for x in bcRawData if self.rePattern.match(x)]

        return bcRawData

    def rotateImg(self, img, angle, reversible=False):
        """ 
        given a np array image object (img), and an angle rotates the img
        without cropping the corners. If reversable == True, calculate the
        reversible matrix
        """
        # see: https://stackoverflow.com/questions/48479656/how-can-i-rotate-an-ndarray-image-properly
        # https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        (height, width) = img.shape[:2]
        (cent_x, cent_y) = (width // 2, height // 2)
        mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
        cos = np.abs(mat[0, 0])
        sin = np.abs(mat[0, 1])
        n_width = int((height * sin) + (width * cos))
        n_height = int((height * cos) + (width * sin))
        mat[0, 2] += (n_width / 2) - cent_x
        mat[1, 2] += (n_height / 2) - cent_y
        rotated_img = cv2.warpAffine(img, mat, (n_width, n_height))
        if reversible:  # now calculate the reverse matrix
            (r_height, r_width) = rotated_img.shape[:2]
            (cent_x, cent_y) = (r_width // 2, r_height // 2) 
            rev_mat = cv2.getRotationMatrix2D((cent_x, cent_y), angle, 1.0)
            rev_mat[0, 2] += (width / 2) - cent_x
            rev_mat[1, 2] += (height / 2) - cent_y
            return rotated_img, rev_mat
        else:  # return none so the results can be parsed similarly 
            return rotated_img, None
    
    def det_bc_center(self, rect, rev_mat):
        """
        Used to determine the center point of a rotation corrected bounding box
        
        :param rect: a pyzbar rectangle array structured as: 
            (left, top, width, height)
        :type rect: Rect, array
        :param angle: the angle of rotation applied to the initial image.
        :type angle: int
        :param rotated_shape: a tuple containing the rotated image's
            (height, width) .
        :type rotated_shape: tuple
        :return: Returns the center point of the barcode before rotation.
        :rtype: tuple, (x, y)
        
        """
        px = rect.left + (rect.width/2)
        py = rect.top + (rect.height/2)
        if not isinstance(rev_mat, np.ndarray):
            # no rotation, so current centerpoint is correct centerpoint
            return (int(px), int(py))
        # otherwise convert current centerpoint using reverse matrix
        nx, ny = rev_mat.dot(np.array((px, py) + (1,))).astype(int)
        return (nx, ny)

    def angle_cos(self, p0, p1, p2):
        """
        Utalized in find_squares, from opencv samples  
        """
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))
    
    def adjust_gamma(self, image, gamma=1.0):
        #from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def find_squares(self, img):
        """
        Heavily modified from opencv samples, attempts to identify squares
        in an img.
        """
        ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.GaussianBlur(img, (3, 3), 3)
        img = cv2.erode(img, None)
        img = cv2.dilate(img, None, iterations=2)

        squares = []
        for thrs in range(0, 255, 51):
            if thrs == 0:
                bin = cv2.Canny(img, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(img, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                contourArea = cv2.contourArea(cnt)
                if len(cnt) == 4 and contourArea > 25 and contourArea < 10000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1 :
                        squares.append(cnt)
        return squares

    def merge_proposals(self, images):
        """
        given a list of image slices, merges them vertically into one image.
        """
        min_pix_length = 10
        images = [x for x in images if x.shape[0] > min_pix_length]
        height = max(image.shape[0] for image in images) +1
        width = len(images) + 1
        output = np.zeros((height,width)).astype('uint8')
        y = 0
        for image in images:
            h = image.shape[0] - 1
            w = 1
            output[0:h+1, y+w] = image
            y += w 
        return output
    
    def det_midpoint(self, p1, p2):
        """
        called by det_centroid_intersection()
        """
        return int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
    
    def det_centroid_intersection(self, square):
        """
        given a square contour, returns 2 vectors intersecting the midpoint.
        """
        a, b, c, d = square
        ab_mid = self.det_midpoint(a, b)
        cd_mid = self.det_midpoint(c, d)
        da_mid = self.det_midpoint(d, a)
        bc_mid = self.det_midpoint(b, c)
        return ab_mid, cd_mid, da_mid, bc_mid
    
    def extend_vector(self, p1, p2, h, w, extend=500):
        """
        given 2 points of a vector, extends it an arbitrary amount not
        exceeding a given height or width and not less than 0.
        """
        theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
        endpt_x = max(0, min(p1[0] - extend*np.cos(theta), w))
        endpt_y = max(0, min(p1[1] - extend*np.sin(theta), h))
    
        theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        startpt_x = max(0, min(p2[0] - extend*np.cos(theta), w))
        startpt_y = max(0, min(p2[1] - extend*np.sin(theta), h))
        return startpt_x, startpt_y, endpt_x, endpt_y
    
    def extract_vector_coords(self, x1, y1, x2, y2, h, w):
        """
        given 2 points of a vector, returns coordinates for the nearest pixels
        traversed by that vector. 
         
        Modified from:
        https://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
        """
        length = int(np.hypot(x2-x1, y2-y1))
        x = np.linspace(x1, x2, length)
        x = np.rint(x).astype(int)
        y = np.linspace(y1, y2, length)
        y = np.rint(y).astype(int)
        pix_coords = y, x
        return pix_coords

    def extract_by_squares(self, gray, retry=True, extension=6):
        """
        given a numpy array image attempts to identify all barcodes using
        vector extraction.
        """
        # apparently this does not generalize well for very large resolutions
        h, w = gray.shape[0:2]
        if max(w,h) > 6800:
            new_size = (int(w*0.8), int(h*0.8))
            w, h = new_size
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_NEAREST)

        # ID squares
        squares = self.find_squares(gray)
        #print(f'found {len(squares)} squares.')
        if len(squares) < 1:
            z = zbar_decode(gray, y_density=3, x_density=3)
        else:
            # iterate over each and det their midpoint intersects

            h -= 1
            w -= 1
            line_data = []
            # extension happens in both directions, therefore effectively doubled.
            extend = min(h, w) // extension
            for square in squares:
                a, b, c, d = square
                ab_mid = self.det_midpoint(a, b)
                cd_mid = self.det_midpoint(c, d)
                x1, y1, x2, y2 = self.extend_vector(ab_mid, cd_mid, h, w, extend=extend)
                pix_coords = self.extract_vector_coords(x1, y1, x2, y2, h, w)
                zi = gray[pix_coords]
                line_data.append(zi)
        
                da_mid = self.det_midpoint(d, a)
                bc_mid = self.det_midpoint(b, c)
                x1, y1, x2, y2 = self.extend_vector(da_mid, bc_mid, h, w, extend=extend)
        
                pix_coords = self.extract_vector_coords(x1, y1, x2, y2, h, w)
                zi = gray[pix_coords]
                line_data.append(zi)
        
            merged_lines = self.merge_proposals(line_data)
            #print(f'merged_lines shape = {merged_lines_shape}')
            z = zbar_decode(merged_lines, y_density=0, x_density=1)
            if len(z) < 1:
                # first try darkening it
                merged_lines = self.adjust_gamma(merged_lines, 0.8)
                z = zbar_decode(merged_lines, y_density=0, x_density=1)
                if len(z) < 1:
                    very_gamma_lines = self.adjust_gamma(merged_lines, 0.4)
                    z = zbar_decode(very_gamma_lines, y_density=0, x_density=1)
                    if len(z) < 1:
                        # if that fails try sharpening it
                        blurred = cv2.GaussianBlur(merged_lines, (0, 0), 10)
                        merged_lines = cv2.addWeighted(merged_lines, 2, blurred, -1, 0)
                        z = zbar_decode(merged_lines, y_density=0, x_density=1)
                        #cv2.imwrite('sharp_merged_lines.jpg', merged_lines)
                        #if len(z) > 0:
                        #    print('sharpening worked')
                    #else:
                        #print('very gamma worked')
                    if len(z) < 1 & retry:
                        # if all that fails squares again but with a darker img
                        gray = self.adjust_gamma(gray, 0.4)
                        z = self.extract_by_squares(gray, retry=False)
                        if len(z) < 1 & retry:
                            # if that fails, try squares on shrunk img
                            o_h, o_w = gray.shape[0:2]
                            new_size = (int(o_h * 0.8), int(o_w * 0.8))
                            gray = cv2.resize(gray, new_size)
                            #print(f'retrying with size {new_size}')
                            z = self.extract_by_squares(gray, retry=False)
                        #else:
                        #    print('retrying with gamma worked')
                #else:
                #    print('gamma worked')
                    
                # if it failed, try once more after darkening (a lot)
                #gray = self.adjust_gamma(gray, 0.4)
                # then sharpening it
                #blurred = cv2.GaussianBlur(gray, (0, 0), 10)
                #gray = cv2.addWeighted(gray, 3, blurred, -1, 0)
                #return self.extract_by_squares(gray, retry=False)
                
        #cv2.imwrite('output.jpg', gray)
        #if len(z) > 0:
        #    if not retry:
        #        print('retrying worked')
        return z

    def testFeature(self, img):
        """Returns bool condition, if this module functions on a test input."""
        try:
            # set aside current pattern and check for ANYTHING
            decodedData = self.decodeBC(img, verifyPattern=True)
            # return current pattern
            if isinstance(decodedData, list):
                return True
            else:
                return False
        except Exception as e:
            print(e)
            # some unknown error, assume test failed
            return False

