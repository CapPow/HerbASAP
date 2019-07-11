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

import piexif


class MetaRead:
    def __init__(self, parent=None, *args):
        super(MetaRead, self).__init__()
        self.parent = parent
        self.exif = None
        self.value = None
        # piexif's transplant function may be useful if the exif is dumped in
        # the saving process.
        # See https://piexif.readthedocs.io/en/latest/functions.html#transplant

    def set_exif(self, imgPath):
        """ given an image file path, attempts to determine
        the make/model of the camera body and lens. """
        # extract exif data as dict
        exifDict = piexif.load(imgPath)
        imgDict = {}
        for ifd in ("0th", "Exif", "GPS", "1st"):
            for tag in exifDict[ifd]:
                tagName = (piexif.TAGS[ifd][tag]["name"])  # , exif_dict[ifd][tag])
                # print(piexif.TAGS[ifd][tag]["name"], exifDict[ifd][tag])
                if tagName.lower() in (
                'make', 'model', 'lensmaker', 'lensmodel', 'focallength', 'fnumber', 'colorspace'):
                    imgDict[tagName.lower()] = exifDict[ifd][tag]
        self.exif = imgDict
        self.value = imgDict['colorspace']

    def get_exif(self):
        if self.exif is None:
            print("Exif not set!")
            return
        else:
            return self.exif

    def get_colorspace(self):
        if self.exif is None:
            print("Exif not set!")
            return
        else:
            return self.exif['colorspace']

    # def testFeature(self, imgPath):
    #     """Returns bool condition, if this module functions on a test input."""
    #
    #     try:
    #         result = self.detImagingEquipment(imgPath)
    #         if ((isinstance(result, dict)) &
    #                 ('cam' in result) &
    #                 ('lens' in result)):
    #             return True
    #         else:
    #             return False
    #     except Exception as e:
    #         print(e)
    #         # some unknown error, assume test failed
    #         return False
