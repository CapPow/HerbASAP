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
                 rotation_list=[9,25,18], parent=None, *args):
        super(bcRead, self).__init__()
        self.parent = parent
        self.compileRegexPattern(patterns)
        # This might need promoted to a user preference in mainwindow
        self.rotation_list = rotation_list
        self.backend = backend

    def decode_zbar(self, im):
        print('dec zbar')
        return zbar_decode(im)

    def decode_libdmtx(self, im):
        print('dec libdmtx')
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
        print(backend)

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

    def checkPattern(self, bcData):
        """ verifies if the bcData matches the compiled rePattern.

            Args:
            bcData (str): a decoded barcode value string, which is checked
            against the collection pattern "self.rePattern"

            Returns (bool): success status of the match
        """
        bcData = bcData.data.decode("utf-8")
        if self.rePattern is None:
            return True
        elif self.rePattern.match(bcData):
            return True
        else:
            return False

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
            code_reader = self.decode_zbar
        elif backend == 'libdmtx':
            code_reader = self.decode_libdmtx

        if verifyPattern:
            bcRawData = [x for x in code_reader(img) if self.checkPattern(x)]
        else:
            # note the potential name collision between re.decode
            bcRawData = [x.data.decode('utf-8') for x in code_reader(img)]
        # if no results are found, start the using the rotation_list
        rot_deg = 0  # in case we don't rotate but want the 
        rev_mat = None  # variable to hold the reverse matrix
        len_bc = len(bcRawData)
        if len_bc < 1:
            for deg in self.rotation_list:
                rot_deg += deg
                rotated_img, rev_mat = self.rotateImg(img, rot_deg, return_details)
                if verifyPattern:
                    bcRawData = [x for x in code_reader(rotated_img) if self.checkPattern(x)]
                else:
                    bcRawData = [x.data.decode('utf-8') for x in code_reader(rotated_img)]
                len_bc = len(bcRawData)
                if len_bc > 0:
                    break
            else:  # if the rotation_list is exhausted, return None
                return None
        if return_details:
            bcData = []
            for result in bcRawData:
                bcValue = result.data.decode("utf-8")
                bcBox = result.rect
                max_dim = max([bcBox.width, bcBox.height])
                center = self.det_bc_center(bcBox, rev_mat)
                bcType = result.type
                resultDict = {'value':bcValue,
                              'center':center,
                              'max_dim':max_dim,
                              'type':bcType}
                bcData.append(resultDict)
            return bcData
        else:
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
        # now calculate the reverse matrix
        (r_height, r_width) = rotated_img.shape[:2]
        (cent_x, cent_y) = (r_width // 2, r_height // 2) 
        rev_mat = cv2.getRotationMatrix2D((cent_x, cent_y), angle, 1.0)
        rev_mat[0, 2] += (width / 2) - cent_x
        rev_mat[1, 2] += (height / 2) - cent_y

        return (rotated_img, rev_mat)
    
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

