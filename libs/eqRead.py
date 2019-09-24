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

import lensfunpy
from lensfunpy import XMLFormatError
import piexif
import cv2
import glob
import numpy as np


class eqRead():
    def __init__(self, parent=None, equipmentDict={}, *args):
        super(eqRead, self).__init__()
        self.parent = parent
        self.equipmentDict = equipmentDict
        try:  # prefer to use system db
            self.db = lensfunpy.Database()
        except XMLFormatError:
            # if there is a format error, attempt to use locally bundled xmls
            self.db = lensfunpy.Database(paths=glob.glob('libs/lensfunpy-db/*.xml'))

    def detImagingEquipment(self, imgPath):
        """ given an image file path, attempts to determine
        the make/model of the camera body and lens. """
        # extract exif data as dict
        exifDict = piexif.load(imgPath)
        imgDict = {}
        for ifd in ("0th", "Exif", "GPS", "1st"):
            for tag in exifDict[ifd]:
                tagName = (piexif.TAGS[ifd][tag]["name"])
                if tagName.lower() in ('make', 'model', 'lensmodel',
                                       'focallength', 'fnumber', 'colorspace'):
                    imgDict[tagName.lower()] = exifDict[ifd][tag]

        for k, v in imgDict.items():
            if isinstance(v, bytes):
                imgDict[k] = v.decode("utf-8")
        camMaker = imgDict.get('make', '')
        camModel = imgDict.get('model', '')
        #print(f'cam model = {camModel}')
        lensModel = imgDict.get('lensmodel', '')#[0]
        lensModel = lensModel.split(" ")[0]
        #print(f'lensmodel = {lensModel}')
        focalLength = imgDict.get('focallength', '')[0]
        #print(f'focallength = {focalLength}')
        apertureValue = imgDict.get('fnumber', '')[0]
        #print(f'apertureValue = {apertureValue}')
        # load the equipment database
        db = self.db
        # lookup the camera details
        cams = db.find_cameras(camMaker, camModel, loose_search=False)#[0]
        # lookup the lens details
        lenses = [x.model for x in db.find_lenses(cams[0], lens=lensModel, loose_search=False)]#[0]
        # organize the result into a dict.
        result = {#'cams':cams,
                  'lenses': lenses,
                  'focalLength': focalLength,
                  'apertureValue': apertureValue,
                  'camMaker': camMaker,
                  'camModel': camModel}

        # drop keys which are null
        #for k,v in result.items():
        #    if v in ['',None,False]:
        #        del result[k]
        
        return result

    def setMod(self, height=5760, width=3840):
        """
         Create the lensfunpy modifier
        """
        equip = self.equipmentDict
        try:
            cam = equip['cam']
            camMaker = equip.get('camMaker','')
            camModel = equip.get('camModel', '')
            cam = self.db.find_cameras(camMaker, camModel, loose_search=False)
            if isinstance(cam, list):
                cam = cam[0]
            lens = str(equip.get('lens',None))
            print(f'dict lens = {lens}')
            lens = self.db.find_lenses(cam, lens=lens, loose_search=True)
            if isinstance(lens, list):
                lens = lens[0]
            if lens is None:
                self.undist_coords = None
                return

            focalDistance = equip.get('focalDistance', 0.255)
            mod = lensfunpy.Modifier(lens,
                                     cam.crop_factor,
                                     width,
                                     height)
            # is lensfunpy.LensCalibTCA useful here?
            mod.initialize(equip['focalLength'],
                           equip['apertureValue'],
                           focalDistance,
                           flags=lensfunpy.ModifyFlags.ALL)
            self.undist_coords = mod.apply_subpixel_geometry_distortion()
        except KeyError as e:
            print(f'error {e}')
            self.undist_coords = None

    def rotate_undist_coords(self):
        """
        if an exception is thrown while attempting to apply lensCorrections,
        the self.undist_coords may need rotated. 
        """
        self.undist_coords = np.rot90(self.undist_coords, 1)

    #def transPlantMetaData
    def lensCorrect(self, im):
        """ Attempts to perform lens corrections using origional image metadata.
            im = an opened image object"""

        r = im[..., 0]
        # see for swapaxes: https://github.com/letmaik/lensfunpy/issues/17
        g = im[..., 1].swapaxes(0,1)
        b = im[..., 2]
        try:
            undist_coords = self.undist_coords  # check if undist coords exist yet
        except AttributeError:  # if not, generate them using input im shape
            print('generating undistcoords')
            h, w = im.shape[0:2]
            try:
                self.setMod(h, w)
                undist_coords = self.undist_coords
            except IndexError:
                pass

        try:
            # generate a corrected channel, then save it back to the image
            r_undistorted = cv2.remap(r, undist_coords[..., 0], None, cv2.INTER_LANCZOS4)
            im[..., 0] = r_undistorted
        except ValueError: # condition where resolution changed from first imputs.
            print('generating new undistcoords')
            h, w = im.shape[0:2]
            self.setMod(h, w)
            undist_coords = self.undist_coords
            r_undistorted = cv2.remap(r, undist_coords[..., 0], None, cv2.INTER_LANCZOS4)
            im[..., 0] = r_undistorted
        except UnboundLocalError:
            return im
        # by this point exceptions should be addressed
        g_undistorted = cv2.remap(g, undist_coords[..., 1], None, cv2.INTER_LANCZOS4)
        im[..., 1] = g_undistorted
        b_undistorted = cv2.remap(b, undist_coords[..., 2], None, cv2.INTER_LANCZOS4)
        im[..., 2] = b_undistorted

        return im

    def retrieveMetaDataBytes(self, srcPath, dstPath):
        """ given a source image, copys source meta data, and adds additional
            user data from group_metaDataApplication. Returns exif data as a
            bytes object."""

        parent = self.parent
        exifDict = piexif.load(srcPath)
        
        collName = parent.plainTextEdit_collectionName.text()
        collURL = parent.plainTextEdit_collectionURL.text()
        collContactEmail = parent.plainTextEdit_contactEmail.text()
        collContactName = parent.plainTextEdit_contactName.text()
        collLicense = parent.plainTextEdit_copywriteLicense.text()

        #exif_dict["0th"][piexif.ImageIFD.DateTime]=datetime.strptime("2000/12/25 12:32","%Y/%m/%d %H:%M").strftime("%Y:%m:%d %H:%M:%S")
        exifBytes = piexif.dump(exif_dict)
        #im.save(dstPath, "jpeg", exif=exif_bytes, quality="keep", optimize=True)
        return exifBytes

    def testFeature(self, img, imgPath, focalDistance):
        """Returns bool condition, if this module functions on a test input."""

        try:
            
            result = self.detImagingEquipment(imgPath)
            height, width = img.shape[0:2]
            self.setMod(result, height, width, focalDistance)
            
            if ((isinstance(result, dict)) &
                ('cam' in result) &
                ('lens' in result)):
                
                modShape = self.undist_coords.shape
                
                return (result, modShape)
            else:
                return False
        except Exception as e:
            print(e)
            # some unknown error, assume test failed
            return False
