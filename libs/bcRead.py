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
import re
import numpy as np
import cv2

from math import cos as math_cos
from math import sin as math_sin
from math import radians
import site
from shutil import copyfile
from platform import system

# Modified pyzbar check
site_packages_dir = site.getsitepackages()
pyzbar_file = None
pyzbar_loc = None
try:
    if system() == 'Windows':
        for potential_dirs in site_packages_dir:
            try:
                pyzbar_file = open(f"{potential_dirs}\\pyzbar\\pyzbar.py")
                pyzbar_loc = f"{potential_dirs}\\pyzbar\\pyzbar.py"
            except:
                continue
    else:
        for potential_dirs in site_packages_dir:
            try:
                pyzbar_file = open(f"{potential_dirs}/pyzbar/pyzbar.py")
                pyzbar_loc = f"{potential_dirs}/pyzbar/pyzbar.py"
            except:
                continue
except:
    print("Need to install the modified pyzbar.py manually as I don't know where your pyzbar is located at. Program may crash.")

if pyzbar_file is not None:
    pzfile_start = pyzbar_file.readlines()[0]
    if pzfile_start != "# autoPostProcessing":
        try:
            if system() == 'Windows':
                copyfile("libs/deps/pyzbar.py", f"{pyzbar_loc}")
            else:
                copyfile("libs/deps/pyzbar.py", f"{pyzbar_loc}")
        except:
            print("Need to install the modified pyzbar.py manually or run this program with sudo/admin! Program may crash.")

from pyzbar.pyzbar import decode as zbar_decode
from pylibdmtx.pylibdmtx import decode as libdmtx_decode

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
        #rot_deg = 0  # in case we don't rotate but want the 
        #rev_mat = None  # variable to hold the reverse matrix
        #len_bc = len(bcRawData)
        #if len_bc < 1:
        #    for deg in self.rotation_list:
        #        rot_deg += deg
        #        rotated_img, rev_mat = self.rotateImg(img, rot_deg, return_details)
        #        # decode each code found from bytes to utf-8
        #        bcRawData = [x.data.decode('utf-8') for x in code_reader(rotated_img)]
        #        if verifyPattern:  # limit the results to those matching rePattern
        #            bcRawData = [x for x in bcRawData if self.rePattern.match(x)]
        #        len_bc = len(bcRawData)
        #        if len_bc > 0:
        #            break
        #    else:  # if the rotation_list is exhausted, return None
        #        return None
        #if return_details:
        #    bcData = []
        #    for result in bcRawData:
        #        bcValue = result.data.decode("utf-8")
        #        bcBox = result.rect
        #        max_dim = max([bcBox.width, bcBox.height])
        #        center = self.det_bc_center(bcBox, rev_mat)
        #        bcType = result.type
        #        resultDict = {'value':bcValue,
        #                      'center':center,
        #                      'max_dim':max_dim,
        #                      'type':bcType}
        #        bcData.append(resultDict)
        #    return bcData
        #else:
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
    
    def find_squares(self, img):
        """
        Modified from: opencv samples, attempts to identify squares in img.
        """
        # Modified from: opencv samples    
        ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.erode(img, (5,5), iterations = 1)
        img = cv2.dilate(img, (5,5), iterations = 1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        squares = []
        for gray in cv2.split(img):  # this can probably be removed given it is always gray
            #for thrs in range(0, 255, 6):
            for thrs in range(0, 255, 18):
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                    bin = cv2.dilate(bin, None)
                else:
                    _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_len = cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                    contourArea = cv2.contourArea(cnt)
                    if len(cnt) == 4 and contourArea > 500 and contourArea < 100000 and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                        if max_cos < 0.2:
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
        length = int(np.hypot(x2-x1, y2-y1))-1
        x = np.linspace(x1, x2, length)
        x = np.rint(x).astype(int)
        y = np.linspace(y1, y2, length)
        y = np.rint(y).astype(int)
        pix_coords = y, x
        return pix_coords

    def extract_by_squares(self, gray):
        """
        given a numpy array image attempts to identify all barcodes using
        vector extraction.
        """
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ID squares
        squares = self.find_squares(gray)
        if len(squares) < 1:
            z = zbar_decode(gray, y_density=3, x_density=3)
        else:
            # iterate over each and det their midpoint intersects
            h, w = gray.shape[0:2]
            h -=1
            w -=1
            line_data = []
            # extension happens in both directions, therefore effectively doubled.
            extend = min(h,w) // 6
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
            z = zbar_decode(merged_lines, y_density=0, x_density=1)
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

