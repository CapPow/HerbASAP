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
import lensfunpy
from lensfunpy import XMLFormatError
import piexif
import cv2
import glob
import numpy as np
import os

class eqRead():
    def __init__(self, parent=None, *args):
        super(eqRead, self).__init__()
        self.parent = parent
        try:  # prefer to use system db
            self.db = lensfunpy.Database()
        except XMLFormatError:
            # if there is a format error, attempt to use locally bundled xmls
            self.db = lensfunpy.Database(paths=glob.glob('libs/lensfunpy-db/*.xml')) # For future use            
        # piexif's transplant function may be useful if the exif is dumped in
        # the saving process.
        # See https://piexif.readthedocs.io/en/latest/functions.html#transplant

    def detImagingEquipment(self, imgPath):
        """ given an image file path, attempts to determine
        the make/model of the camera body and lens. """
        # extract exif data as dict
        exifDict = piexif.load(imgPath)
        imgDict = {}
        for ifd in ("0th", "Exif", "GPS", "1st"):
            for tag in exifDict[ifd]:
                tagName = (piexif.TAGS[ifd][tag]["name"])#, exif_dict[ifd][tag])
                #print(piexif.TAGS[ifd][tag]["name"], exifDict[ifd][tag])
                if tagName.lower() in ('make','model','lensmaker','lensmodel','focallength','fnumber','colorspace'):
                    imgDict[tagName.lower()] = exifDict[ifd][tag]

        for k,v in imgDict.items():
            if isinstance(v, bytes):
                imgDict[k] = v.decode("utf-8")
        #'Artist'
        #'Copyright'
        camMaker = imgDict.get('make','')
        camModel = imgDict.get('model','')
        lensMaker = imgDict.get('lensmaker','')
        lensModel = imgDict.get('lensmodel','')
        focalLength = imgDict.get('focallength','')[0]
        apertureValue = imgDict.get('fnumber','')[0]

        # load the equipment database
        db = self.db
        # lookup the camera details
        cam = db.find_cameras(camMaker, camModel, loose_search=False)[0]
        # lookup the lens details
        lens = db.find_lenses(cam, lensMaker, lensModel, loose_search=False)[0]
        # organize the result into a dict.
        result = {'cam':cam,
                  'lens':lens,
                  'focalLength':focalLength,
                  'apertureValue':apertureValue}

        # drop keys which are null
        for k,v in result.items():
            if v in ['',None,False]:
                del result[k]
        
        return result
    
    #def transPlantMetaData
    def lensCorrect(self, im, imgPath, focalDistance=0.255):
        """ Attempts to perform lens corrections using origional image metadata.
            im = an opened image object,
            imgPath = the origional file object (for metadata extraction)"""

        # extract the equipment details. Returned as dict (equip).
        equip = self.detImagingEquipment(imgPath)
        # setup equipment variables
        height, width, rgb = im.shape
        mod = lensfunpy.Modifier(equip['lens'],
                                 equip['cam'].crop_factor,
                                 width,
                                 height)
        # is lensfunpy.LensCalibTCA useful here?
        mod.initialize(equip['focalLength'],
                       equip['apertureValue'],
                       focalDistance,
                       flags=lensfunpy.ModifyFlags.ALL)
        # generate pixel map
        undist_coords = mod.apply_subpixel_geometry_distortion()        
        r = im[..., 0]
        # see for swapaxes: https://github.com/letmaik/lensfunpy/issues/17
        g = im[..., 1].swapaxes(0,1)
        b = im[..., 2]
        r_undistorted = cv2.remap(r, undist_coords[..., 0], None, cv2.INTER_LANCZOS4)
        g_undistorted = cv2.remap(g, undist_coords[..., 1], None, cv2.INTER_LANCZOS4)
        b_undistorted = cv2.remap(b, undist_coords[..., 2], None, cv2.INTER_LANCZOS4)
        # save the channel maps back to the image.
        im[..., 0] = r_undistorted
        im[..., 1] = g_undistorted
        im[..., 2] = b_undistorted
        return im

    def retrieveMetaDataBytes(self, srcPath, dstPath):
        """ given a source image, copys source meta data, and adds additional
            user data from group_metaDataApplication. Returns exif data as a
            bytes object."""

        exifDict = piexif.load(srcPath)
        
        collName = self.parent.plainTextEdit_collectionName.text()
        collURL = self.parent.plainTextEdit_collectionURL.text()
        collContactEmail = self.parent.plainTextEdit_contactEmail.text()
        collContactName = self.parent.plainTextEdit_contactName.text()
        collLicense = self.parent.plainTextEdit_copywriteLicense.text()

        #exif_dict["0th"][piexif.ImageIFD.DateTime]=datetime.strptime("2000/12/25 12:32","%Y/%m/%d %H:%M").strftime("%Y:%m:%d %H:%M:%S")
        exifBytes = piexif.dump(exif_dict)
        #im.save(dstPath, "jpeg", exif=exif_bytes, quality="keep", optimize=True)
        return exifBytes

    def testFeature(self, imgPath):
        """Returns bool condition, if this module functions on a test input."""

        try:
            result = self.detImagingEquipment(imgPath)
            if ((isinstance(result, dict)) &
                ('cam' in result) &
                ('lens' in result)):
                return True
            else:
                return False
        except Exception as e:
            print(e)
            # some unknown error, assume test failed
            return False
