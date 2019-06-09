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
from pyzbar.pyzbar import decode

import numpy as np
import cv2


class bcRead():
    """A barcode reader class

    Args:
    prefix (str, optional): a collection code prefix. If left empty,
        Matches any to no value(s)

    digits (int, optional): the quantity of digits expected in the barcode.
        If left empty, matches any to no digit(s)

    rotation_list (iterable, optional): iterable containing a series of int
        representing image rotations (in degrees) to attempt if no barcode is 
        found. Default values are [9, 15, 18]. Rotation attempts stop after any
        results are found. The list's rotations are cumulative. Short or empty
        lists will decrease the time before giving up on finding a barcode.
        
    Attributes:
    rePattern (obj): A compiled regex pattern
    rotation_list (list): The saved rotation list
    """

    def __init__(self, prefix=".*", digits="*", 
                 rotation_list=[9,15,18], parent=None, *args):
        super(bcRead, self).__init__()
        self.parent = parent
        self.compileRegexPattern(prefix, digits)
        # This might need promoted to a user preference in mainwindow
        self.rotation_list = rotation_list

    def compileRegexPattern(self, prefix, digits):
        """ compiles a collection specific regex pattern """
        #  if the digits are not explicit, match 0 to any quantity of digits.
        if digits == "*":
            collRegEx = rf'^({prefix}\d*)\D*'
        else:
            # triple { required for combining regex and f strings
            collRegEx = rf'^({prefix}\d{{{digits}}})\D*'
        rePattern = re.compile(collRegEx)
        # set the compiled regex as a class attribute.
        self.rePattern = rePattern

    def checkPattern(self, bcData):
        """ verifies if the bcData matches the compiled rePattern.

            Args:
            bcData (str): a decoded barcode value string, which is checked
            against the collection pattern "self.rePattern"

            Returns (bool): success status of the match
        """
        bcData = bcData.data.decode("utf-8")
        if self.rePattern.match(bcData):
            return True
        else:
            return False

    def decodeBC(self, img, return_details=False):
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
        bcRawData = [x for x in decode(img) if self.checkPattern(x)]
        # if no results are found, start the using the rotation_list
        if len(bcRawData) < 1:
            for deg in self.rotation_list:
                #print(deg) # useful to see how frequently this happens
                img = self.rotateImg(img, deg)
                bcRawData = [x for x in decode(img) if self.checkPattern(x)]
                if len(bcRawData) > 0:
                    break

        if return_details:
            bcData = []
            for result in bcRawData:
                bcValue = result.data.decode("utf-8")
                bcBox = result.rect
                bcType = result.type
                resultDict = {'value':bcValue,
                              'bbox':bcBox,
                              'type':bcType}
                bcData.append(resultDict)
        else:
            # filter out non-matching strings
            bcData = [x.data.decode("utf-8") for x in bcRawData]
            # a list of matched barcodes found in bcRawData

        return bcData

    def rotateImg(self, img, angle):
        """ given a np array image object (img), and an angle rotates the img
            without cropping the corners.
        """
        # see: https://stackoverflow.com/questions/48479656/how-can-i-rotate-an-ndarray-image-properly

        (height, width) = img.shape[:2]
        (cent_x, cent_y) = (width // 2, height // 2)
    
        mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
        cos = np.abs(mat[0, 0])
        sin = np.abs(mat[0, 1])
    
        n_width = int((height * sin) + (width * cos))
        n_height = int((height * cos) + (width * sin))
    
        mat[0, 2] += (n_width / 2) - cent_x
        mat[1, 2] += (n_height / 2) - cent_y
    
        return cv2.warpAffine(img, mat, (n_width, n_height))

    def testFeature(self, img):
        """Returns bool condition, if this module functions on a test input."""

        try:
            if isinstance(self.decodeBC(img), list):
                return True
            else:
                return False
        except:
            # some unknown error, assume test failed
            return False

