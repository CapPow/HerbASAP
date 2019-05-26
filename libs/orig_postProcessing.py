#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:12:33 2019

@author: Caleb Powell

A starting point for a GUI automating image post processing in natural history
digitization efforts. The program is expected to be a user interface for 
setting preferences including: 

    - which post-processing functions to use 
        > i.e., should it rename files based on barcode values
    - how to use them
        > i.e., what is the barcode pattern to match on
    - where to look for unprocesesd images
        > i.e., where are the images going after being taken
    - what do do with processed images
        > i.e., where to save processed images, and in what format
        
In addition to setting preferences, predominately displayed on the GUI should
be a button to start monitoring a folder for post-processing. This should be
run while imaging is going on. The program will monitor a predetermined folder 
for new images, which need processed and process them as they're being created.

It may be ideal for it to always clear out the folder it is monitoring. This 
way backlogged postprocessing may also be cleared if the program was started 
later. This program will prevent collections managers from having to perform 
time intensive post processing.

Features in demand for such a program are:
    - rename file based on a barcode value present in image
        > functions exist, requires GUI elements, testing
    - correct lens aberration correction
        > functions exist, requires GUI elements, testing
    - load raw formats (.cr2), save into interoperable (.jpg, .tiff) formats.
        > functions exist, requires GUI elements, testing
    - apply exif data
        > no functions, capacity exists in piexif library.
    - image orientation (correct rotation)
        > assume color chip location is static, use GUI to select that location
        > no functions, requires color chip detection, GUI elements, testing
    - white balance
        > no functions, requires color chip detection, GUI elements, testing



"""

import lensfunpy
import piexif
import rawpy
from rawpy import LibRawNonFatalError
import PIL
from pyzbar.pyzbar import decode
import re
import cv2
import os

# this library can go away once we set up a GUI
# ideally, we'll monitor a folder while program is running 
# new additions to the folder will be processed according to GUI preferences
import glob

# below are early states of the various functions 

def detImagingEquipment(inputImage):
    """ given an image file object, attempts to determine
        the make/model of the camera body and lens. """
    # extract exif data as dict
    exifDict = piexif.load(inputImage)
    imgDict = {}
    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag in exifDict[ifd]:
            tagName = (piexif.TAGS[ifd][tag]["name"])#, exif_dict[ifd][tag])
            #print(piexif.TAGS[ifd][tag]["name"], exifDict[ifd][tag])
            if tagName.lower() in ('make','model','lensmaker','lensmodel','focallength', 'fnumber'):
                imgDict[tagName.lower()] = exifDict[ifd][tag]
    for k,v in imgDict.items():
        if isinstance(v, bytes):
            imgDict[k] = v.decode("utf-8")

    camMaker = imgDict.get('make','')
    camModel = imgDict.get('model','')
    lensMaker = imgDict.get('lensmaker','')
    lensModel = imgDict.get('lensmodel','')
    focalLength = imgDict.get('focallength','')[0]
    apertureValue = imgDict.get('fnumber','')[0]
    # load the equipment database
    db = lensfunpy.Database()
    # lookup the camera details
    cam = db.find_cameras(camMaker, camModel, loose_search=False)[0]
    # lookup the lens details
    lens = db.find_lenses(cam, lensMaker, lensModel, loose_search=False)[0]
    
    return cam, lens, focalLength, apertureValue

def lensCorrect(imCV, inputImg, distance = 0.3):
    """ Attempts to perform lens corrections using origional image metadata.
    
        imCV = an open cv formatted image object,
        inputImg = the origional file object (for metadata extraction)
        distance = the focal distance in meters to subject"""
        
    #TODO add GUI elements to determine focal distance for individual setup
    
    # extract the exif
    cam, lens, focalLength, apertureValue = detImagingEquipment(inputImg)
    # determine the image shape
    height, width = imCV.shape[0], imCV.shape[1]
    
    # use lensfunpy to calculate corrections
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focalLength, apertureValue, distance)
    undist_coords = mod.apply_geometry_distortion()

    # apply the corrections using openCV
    im_undistorted = cv2.remap(imCV, undist_coords, None, cv2.INTER_LANCZOS4)

    # save the results
    #pre, ext = os.path.splitext(outputImgName)
    #outputImgName = pre + outputFormat
    #cv2.imwrite(outputImgName, im_undistorted)
    return im_undistorted

def openRawAsOpenCV(inputImg):
    """ given an image file, attempts
        to return an openCV image object"""
    # use rawpy to convert raw to openCV
    try:
        with rawpy.imread(inputImg) as raw:
            rgb = raw.postprocess() # a numpy RGB array
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # the OpenCV image
    # if it is not a raw format, just try and open it.
    except LibRawNonFatalError:
        im = cv2.imread(inputImg)
    return im

def decodeBarcode(imCV):
    """ given an openCV image object, attempts to
        decode the barcodes, returning a list of
        matching results."""

    # option here to include code format restrictions
        # decode(Image.open('pyzbar/tests/qrcode.png'), symbols=[ZBarSymbol.QRCODE])
    # if successful, returns a list of barcodes similar to below:
    # [Decoded(data=b'ETSU006566', type='CODE39', rect=Rect(left=3278, top=219, width=523, height=100), 
    #   polygon=[Point(x=3278, y=219), Point(x=3278, y=319), Point(x=3801, y=318), Point(x=3801, y=222)])]

    bcData = decode(imCV)  # get the barcode data
    # if nothing is decoded, iterate through the rotation list and keep trying.
    if len(bcData) == 0:
        for i in rotationList:
            rotatedImg = rotateImage(imCV, i)
            bcData = decode(rotatedImg)
            if len(bcData) != 0:
                break
            # give up
            print('could not find a barcode value in image')
        # pull out all pattern matching bcValues
    bcValues = [x.data.decode("utf-8") for x in bcData if collectionPatterns.match(x.data.decode("utf-8"))]
    
    # if bcValues > 1:
    # add pyqt5 dialog box with list picker for bcValues

    if len(bcValues) == 1:
        bcValue = bcValues[0]
    else:
        bcValue = None

    return bcValue


def rotateImage(imCV, angle):
    """ given an opencv image object, and an angle 
        return the rotated image"""
    # see: https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/33564950#33564950

    print(f'trying {angle} deg rotation')

    height, width = imCV.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(imCV, rotation_mat, (bound_w, bound_h))
    return rotated_mat
    

def saveImg(imCV, saveName, outputFolder = '', outputFormat = '.jpg'):
    """ saves an openCV image object as an image."""

    #pre, ext = os.path.splitext(outputImgName)
    #outputImgName = pre + outputFormat
    savePath = outputFolder + saveName + outputFormat
    print('saving to: ', savePath)
    cv2.imwrite(savePath, imCV)
    
def processImage(inputImg):
    """ given a raw formatted image, attempts to
        perform the necessary postprocessing steps."""
    # open the image as an openCV object
    imCV = openRawAsOpenCV(inputImg)
    # attempt to make lens corrections
    imCV = lensCorrect(imCV, inputImg)
    # attempt to decode a barcode
    barCode = decodeBarcode(imCV)
    # if barcode decoded, use it to name the file
    if barCode != None:
        saveName = barCode
    else:
        saveName = os.path.basename(inputImg)

    saveImg(imCV, saveName, outputFolder='./correctedImages/')
    
    
    return imCV

# define some constants

# pre-compile the stored collection regex patterns.
collectionPatterns = [(r'^(UCHT\d{6})\D*'),
                      (r'^(TENN-V-\d{7})\D*'),
                      (r'^(APSC\d{7})\D*'),
                      (r'^(HTTU\d{6})\D*'),
                      (r'^(ETSU\d{6})\D*'),
                      (r'^(MTSU\d{6})\D*'),
                      (r'^(SWMT\d{5})\D*'),
                      (r'^(UTM\d{5})\D*'),
                      (r'^(UOS\d{5})\D*'),
                      (r'^(MEM\d{6})\D*'),
                      (r'^(GSMNP\d{5})\D*')]
# join the stored patterns into a single "OR" joined pattern
collectionPatterns = re.compile( '|'.join( collectionPatterns) )

# a list of rotations to attempt after failed barcode attempt decode
rotationList = [12, 24, 36, 48]

# run the script to demonstrate functions

for img in glob.glob('./testImages/*.*'):
    #outputImgName = f'./correctedImages/{os.path.basename(img)}'
    imCV = processImage(img)