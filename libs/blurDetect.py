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
import numpy as np
import cv2

class blurDetect():
    """A blur detection function wrapped up in a class.
    """

    def __init__(self, parent=None, *args):
        super(blurDetect, self).__init__()
        self.parent = parent

    def blur_check(self, img, threshold=0.008, return_details=False):
        """ Given a np array image object (img)attempts to determine if an
        image array object is blurry using a normalized variance of Laplacian.

        Args:
        img (numpy.ndarray): a numpy image array object

        threshold (float): default = 0.008. The threshold to determine if the
        image is blury.

        return_details (bool, optional): default = False. Whether or not to
            return the metrics used to make the blur determination. When True,
            returned value is a dictionary.

        Returns (bool): Boolean result of "is blurry" determination based on
            a normalized Laplacian variance.
            If return_details = True, then returns a dictionary.

        Modified from:
            https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

        Based on the work of: Pertuz, Domenec Puig, and Angel Garcia (2013)
        https://doi.org/10.1016/j.patcog.2012.11.011
        """

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # store the variance of openCV's Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        # calc the image's variance, to account for 'image busyness'
        imVar = np.var(gray)
        # normalize the laplacian variance by image variance
        lapNorm = laplacian / imVar
        # lower variance of laplacian means higher blur
        if lapNorm < threshold:
            isBlur = True
        else:
            isBlur = False
        # build the results based on return_details
        if return_details:
            result = {'isblurry': isBlur,
                      'laplacian': laplacian,
                      'imVar': imVar,
                      'lapNorm': lapNorm}
        else:
            result = isBlur

        return result

    def testFeature(self, img):
        """Returns bool condition, if this module functions on a test input."""

        try:
            if isinstance(self.blur_check(img), bool):
                return True
            else:
                return False
        except:
            # any error, assume test failed
            return False

