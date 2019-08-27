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
import json
import copy


class MetaRead:
    def __init__(self, parent=None, *args):
        super(MetaRead, self).__init__()
        self.parent = parent
        self.exif = None
        self.value = None
        # piexif's transplant function may be useful if the exif is dumped in
        # the saving process.
        # See https://piexif.readthedocs.io/en/latest/functions.html#transplant
        # static_exif is a dict with the  UI harvested metadata details.
        self.static_exif = {}

    def update_static_exif(self, exif_dict):
        """
        Given an input dictionary, prepares it for storage as self.static_exif
        see: https://github.com/hMatoba/Piexif/blob/master/piexif/_exif.py
        """
        static_exif = {
                305:exif_dict.get('version', ''),
                315:exif_dict.get('collectionName', ''),
                33432: exif_dict.get('copywriteLicense', '')
                }
        # jsonize the entire passed in exif_dict
        
        #user_comments = json.dumps(exif_dict)
        
        # apply jsonized user_comments to static_exif[270], "ImageDescription"
        static_exif[270] = exif_dict
        # save the static_exif object as a class variable
        self.static_exif = static_exif
        
    def retrieve_src_exif(self, src, addtl_user_comments):
        """
        Given an input source (src) image file path, returns an dictionary
        containing the source exif data, updated with the static_exif data.
        """
        # prepare the user_comments
        static_exif = copy.deepcopy(self.static_exif)
        static_exif[270].update(addtl_user_comments)
        user_comments = json.dumps(static_exif[270])
        static_exif[270] = user_comments

        # extract exif data as dict
        exifDict = piexif.load(src)
        # update the input dict with the static_exif info
        exifDict['0th'].update(static_exif)
        # have to remove makerNotes or else it crashes with CR2s
        exifDict['Exif'].pop(37500, None)
        # dump exifDict to bytes
        exif_bytes = piexif.dump(exifDict)
        return exif_bytes

    def set_dst_exif(self, exif_dict, dst):
        """
        Given an exif object and a destination image, attempts to insert the
        exif metadata into the destination image.
        """
        piexif.insert(exif_dict, dst)
    
    def transplant_meta(src, dst):
        """
        transplants unmodified metadata from one file to another.
        """
        piexif.transplant(src, dst)

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
