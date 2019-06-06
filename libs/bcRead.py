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
    def __init__(self, parent=None, *args):
        super(bcRead, self).__init__()
        self.parent = parent
        self.compileRegexPattern()
        # This might need promoted to a user preference in mainwindow
        self.rotationList = [12, -12,  21, -21]

    def compileRegexPattern(self):
        """ precompiles collection's regex pattern """
        prefix = self.parent.lineEdit_catalogNumberPrefix.text()
        digits = int(self.parent.spinBox_catalogDigits.value())
        # triple { required for combining regex and f strings
        collRegEx = fr'^({prefix}\d{{{digits}}})\D*'
        rePattern = re.compile(collRegEx)
        # set the compiled regex as a class attribute.
        self.rePattern = rePattern

    def checkPattern(self, bcData):
        """ verifies if the bcData matches the compiled rePattern.
        Returns bool """

        if self.rePattern.match(bcData):
            return True
        else:
            return False
    
    def decodeBC(self, img):
        """given a np array image object (img), decodes BCs and returns those
        which match self.rePattern"""

        bcRawData = decode(img)
        bcStrings = [x.data.decode("utf-8") for x in bcRawData]
        bcData = [x for x in bcStrings if self.checkPattern(x)]  # get the barcode data
        return bcData

    def retrieveBcMatches(self, img):
        """ given a np array image object (img), retrieves the BC matches """

        bcData = self.decodeBC(img)
        if len(bcData) < 1:
            # if no barcodes are detected
            for i in self.rotationList:
                # try 'wiggling' the image some
                img2 = rotateImg(img, i)
                bcData = self.decodeBC(img2)
                if len(bcData) > 0:
                    break
            #Give up on rotation, and handle null result
            return None

        elif len(bcData) == 1:
            # if only 1 barcode was detected:
            return bcData

        else:
            # if there are > 1 barcodes detected.
            # TODO if there are >1 bc, save the output under each valid bc number.
            # for now BREAK it
            breakPOINT


    
    def rotateImg(img, angle):
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
            if isinstance(self.retrieveBcMatches(img), list):
                return True
            else:
                return False
        except:
            # some unknown error, assume test failed
            return False

