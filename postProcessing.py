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
__author__ = "Caleb Powell, Joey Shaw"
__credits__ = ["Caleb Powell, Joey Shaw"]
__email__ = "calebadampowell@gmail.com"
__status__ = "Alpha"
__version__ = 'v0.0.1-alpha'

import os
import sys
# UI libs
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QSettings, Qt, QObject
# image libs
import lensfunpy
import piexif
import rawpy
from rawpy import LibRawNonFatalError
#from PIL import Image
import cv2
# internal libs
from ui.postProcessingUI import Ui_MainWindow
from libs.bcRead import bcRead
from libs.eqRead import eqRead
from libs.blurDetect import blurDetect

class appWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.mainWindow = Ui_MainWindow()
        self.mainWindow.setupUi(self)        
        # initiate the persistant settings
        # todo update this when name is decided on
        self.settings = QSettings('AYUP', 'AYUP')        
        self.settings.setFallbacksEnabled(False)    # File only, no fallback to registry.
        # populate the settings based on the previous preferences
        self.populateSettings()
        # fill in the previews
        self.updateCatalogNumberPreviews()
        
        prefix = self.mainWindow.lineEdit_catalogNumberPrefix.text()
        digits = int(self.mainWindow.spinBox_catalogDigits.value())
        self.bcRead = bcRead(prefix, digits, parent=self.mainWindow)

        self.blurDetect = blurDetect(parent=self.mainWindow)

        self.eqRead = eqRead(parent=self.mainWindow)
        # assign applicable user settings for eqRead. 
        # this function is here, for ease of slot assignment in pyqt designer
        self.updateEqSettings()

#        self.versionCheck()

#   when saving: quality="keep" the original quality is preserved 
#   The optimize=True "attempts to losslessly reduce image size


#    def versionCheck(self):
#        """ checks the github repo's latest release version number against
#        local and offers the user to visit the new release page if different"""
#        #  be sure to only do this once a day.
#        today = str(date.today())
#        lastChecked = self.settings.get('date_versionCheck', today)
#        self.w.date_versionCheck = today
#        if today != lastChecked:
#            import requests
#            import webbrowser
#            apiURL = 'https://api.github.com/repos/CapPow/collBook/releases/latest'
#            try:
#                apiCall = requests.get(apiURL)
#                status = str(apiCall)
#            except ConnectionError:
#                #  if no internet, don't bother the user.
#                pass
#            result = apiCall.json()
#            if '200' in status:  # if the return looks bad, don't bother user
#                url = result['html_url']
#                version = result['tag_name']
#                if version.lower() != self.w.version.lower():
#                    message = f'A new version ( {version} ) of collBook has been released. Would you like to visit the release page?'
#                    title = 'collBook Version'
#                    answer = self.userAsk(message, title, inclHalt = False)
#                    if answer:# == QMessageBox.Yes:
#                        link=url
#                        self.showMinimized() #  hide the app about to pop up.
#                        #  instead display a the new release
#                        webbrowser.open(link,autoraise=1)
#            self.settings.saveSettings()  # save the new version check date

#
#    def setMaxZoom(self):
#        try:
#            self.parent.updatePreview()
#        except AttributeError:
#            pass
#        screenSize = (self.parent.geometry())
#        screenX = screenSize.width()
#        xMax = screenX * 0.6
#        screenY = screenSize.height()
#        yMax = screenY * 0.75
#        #  resulting pdfs are 96dpi skip calling getPixmap twice, Assume 96
#        #  96dpi / 25.4mm per inch = 3.780 dots per mm
#        label_X = int(self.settingsWindow.value_X.value() * (3.780))
#        label_Y = int(self.settingsWindow.value_Y.value() * (3.780))
#        max_XZoom = xMax / label_X
#        max_YZoom = yMax / label_Y
#        max_Zoom = int(min(max_XZoom, max_YZoom) * 100)
#        self.parent.w.value_zoomLevel.setMaximum(max_Zoom)
#        currentZoom = int(self.parent.w.value_zoomLevel.value())
#        if currentZoom > max_Zoom:
#            valueToSet = (max_Zoom)
#        else:
#            valueToSet = currentZoom
#        self.parent.w.value_zoomLevel.setValue(valueToSet)
#        self.parent.w.label_zoomLevel.setText(f'{str(valueToSet).rjust(4," ")}%')  # update the label
#        self.settings.setValue('value_zoomLevel', valueToSet)  # update settings

    def testFunction(self):
        """ a development assistant function, connected to a GUI button
        used to test various functions before complete GUI integration."""

        imgPath, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Open Sample Image", QtCore.QDir.homePath())
        #  start a timer
        import time
        startTime = time.time()
        # open the file
        im = self.openImageFile(imgPath)
        cv2.imwrite('openedImg.jpg', im)
        # test for bluryness
        blurStatus = self.blurDetect.blur_check(im)
        print(f'blurStatus: {blurStatus}')
        # read the BC
        bc = self.bcRead.decodeBC(im)
        print(f'barcode(s) found: {bc}')
        # perform equipment corrections
        correctedImg = self.eqRead.lensCorrect(im, imgPath)
        # save output
        cv2.imwrite('alteredImg.jpg', correctedImg)
        # finish timer
        elapsedTime = int(time.time() - startTime)
        # test total elapsed time so far with rotated and straight images.
        print(f'opening raw, testing blur status, reading barcode, equipment corrections and saving outputs required {elapsedTime} seconds')

    def openImageFile(self, imgPath, demosaic = rawpy.DemosaicAlgorithm.AHD):
        """ given an image path, attempts to return a numpy array image object
        """

        try:  # use rawpy to convert raw to openCV
            with rawpy.imread(imgPath) as raw:
                bgr = raw.postprocess(chromatic_aberration=(1,1),
                                      demosaic_algorithm = demosaic) 
                im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # the OpenCV image
        # if it is not a raw format, just try and open it.
        except LibRawNonFatalError:
            im = cv2.imread(imgPath)
        return im

    def testFeatureCompatability(self):
        """ called by the "pushButton_selectTestImage."
            
            given image path input from the user, calls testFeature() for
            each process class. Enabling passing groups, while disabling 
            failing groups.
            
            Ideally, the user will select an example image from their 
            imaging setup and the available features will become available."""
        
        imgPath, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Open Sample Image", QtCore.QDir.homePath())

        try:
            im = self.openImageFile(imgPath)
            bcStatus = self.bcRead.testFeature(im)
            #ccStatus = self.colorChecker.testFeature(im)
            eqStatus = self.eqRead.testFeature(imgPath)
        except Exception as e:
            # TODO add user notify for this error.
            print(e)
            bcStatus = False
            ccStatus = False
            eqStatus = False 
        
        self.mainWindow.group_barcodeDetection.setEnabled(bcStatus)
        #self.mainWindow.group_colorCheckerDetection.setEnabled(ccStatus)
        self.mainWindow.group_equipmentDetection.setEnabled(eqStatus)

    def updateEqSettings(self):
        """ called when a change is made to any appropriate fields in 
        equipment detection group. Updates the eqRead class' properties.
        This avoids having to read from the UI each time a eqRead function is
        called."""

        cmDistance = self.mainWindow.doubleSpinBox_focalDistance.value()
        mDistance = cmDistance / 100
        # set focal distance for distortion correction
        self.eqRead.mDistance = mDistance

        # set preferences for equipment detection group processes
        self.eqRead.vignettingCorrection = self.mainWindow.checkBox_vignettingCorrection.isChecked()
        self.eqRead.distortionCorrection = self.mainWindow.checkBox_distortionCorrection.isChecked()
        self.eqRead.chromaticAberrationCorrection = self.mainWindow.checkBox_chromaticAberrationCorrection.isChecked()

    def updateCatalogNumberPreviews(self):
        """ called when a change is made to any of the appropriate fields in 
        barcode detection group. Updates the example preview labels"""

        # skip everything if the group is not enabled.
        if not self.mainWindow.group_renameByBarcode.isEnabled():
            return
        
        # recompile the regexPattern
        try:
            prefix = self.lineEdit_catalogNumberPrefix.text()
            digits = int(self.spinBox_catalogDigits.value())
            self.bcRead.bcReadcompileRegexPattern(prefix, digits)
        except AttributeError:
            # bcRead may not have been imported yet
            pass
        
        # update bc pattern preview
        prefix = self.mainWindow.lineEdit_catalogNumberPrefix.text()
        digits = int(self.mainWindow.spinBox_catalogDigits.value())
        startingNum = ''.join(str(x) for x in list(range(digits)))
        startingNum = startingNum.zfill(digits)  # fill in leading zeroes
        previewText = f'{prefix}{startingNum}'  # assemble the preview string.
        self.mainWindow.label_previewDisplay.setText(previewText) # set it
        # update dup naming preview
        dupNamingPolicy = self.mainWindow.comboBox_dupNamingPolicy.currentText()
        if dupNamingPolicy == 'append LOWER case letter':
            dupPreviewEnd = 'a'
        elif dupNamingPolicy == 'append UPPER case letter':
            dupPreviewEnd = 'A'
        elif dupNamingPolicy == 'append Number with underscore':
            dupPreviewEnd = '1'
        elif dupNamingPolicy == 'OVERWRITE original image with newest':
            dupPreviewEnd = ''
        else:
            return
        dupPreviewText = f'{previewText}{dupPreviewEnd}'
        self.mainWindow.label_dupPreviewDisplay.setText(dupPreviewText) # set it

    def setFolderPath(self):
        """ Called by all of the "Browse" buttons in Image Location Tab.
        Assigns selected folder name to the associated lineEdit.
        Uses hacky methods to lineEdit associated to button."""

        # this only works with strict object naming conventions.
        # get name of button pressed
        buttonPressed = self.sender().objectName().split('_')[-1]
        # use string methods to get the associated lineEdit name
        # use eval method to convert string to variable.
        targetField = eval(f'self.mainWindow.lineEdit_{buttonPressed}')
        targetDir = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Select a folder:', QtCore.QDir.homePath(),
                QtWidgets.QFileDialog.ShowDirsOnly)
        targetField.setText(targetDir)

    # Functions related to the saving and retrieval of preference settings

    def has(self, key):
        return self.settings.contains(key)

    def setValue(self, key, value):
        return self.settings.setValue(key, value)

    def get(self, key, altValue = None):
        result = self.settings.value(key, altValue)
        if result == 'true':
            result = True
        elif result == 'false':
            result = False
        return result
    
    def populateQComboBoxSettings(self, obj, value):
        """ sets a QComboBox based on a string value. Presumed to be a more
        durable method. obj is the qComboBox object, and value is a string
        to search for"""
        index = obj.findText(value)
        obj.setCurrentIndex(index)

    def convertCheckState(self, stringState):
        """ given a string either "true" or "false" returns the proper Qt.CheckState"""
        if str(stringState).lower() != 'true':
            return Qt.Unchecked
        else:
            return Qt.Checked
    
    def convertEnabledState(self, stringState):
        """ given a string either "true" or "false" returns the bool"""
        if str(stringState).lower() != 'true':
            return False
        else:
            return True

    def populateSettings(self):
        """ uses self.settings to populate the preferences widget's selections"""

        # Many fields are connected to save on 'changed' signals
        # block those emissions while populating values.
        children = self.mainWindow.tabWidget.findChildren(QObject)
        [x.blockSignals(True) for x in children]

        # QComboBox
        comboBox_imageOrientation = self.get('comboBox_imageOrientation', 'Portrait')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_imageOrientation, comboBox_imageOrientation)
        comboBox_colorCheckerPosition = self.get('comboBox_colorCheckerPosition', 'Upper right')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_colorCheckerPosition, comboBox_colorCheckerPosition)
        comboBox_dupNamingPolicy = self.get('comboBox_dupNamingPolicy', 'append LOWER case letter')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_dupNamingPolicy, comboBox_dupNamingPolicy)

        # QLineEdit
        lineEdit_catalogNumberPrefix = self.get('lineEdit_catalogNumberPrefix','')
        self.mainWindow.lineEdit_catalogNumberPrefix.setText(lineEdit_catalogNumberPrefix)
        lineEdit_inputPath = self.get('lineEdit_inputPath','')
        self.mainWindow.lineEdit_inputPath.setText(lineEdit_inputPath)
        lineEdit_pathUnalteredRaw = self.get('lineEdit_pathUnalteredRaw','')
        self.mainWindow.lineEdit_pathUnalteredRaw.setText(lineEdit_pathUnalteredRaw)
        lineEdit_pathProcessedJpg = self.get('lineEdit_pathProcessedJpg','')
        self.mainWindow.lineEdit_pathProcessedJpg.setText(lineEdit_pathProcessedJpg)
        lineEdit_pathProcessedTIFF = self.get('lineEdit_pathProcessedTIFF','')
        self.mainWindow.lineEdit_pathProcessedTIFF.setText(lineEdit_pathProcessedTIFF)
        lineEdit_pathProcessedPng = self.get('lineEdit_pathProcessedPng','')
        self.mainWindow.lineEdit_pathProcessedPng.setText(lineEdit_pathProcessedPng)

        # QPlainTextEdit
        plainTextEdit_collectionName = self.get('plainTextEdit_collectionName','')
        self.mainWindow.plainTextEdit_collectionName.setPlainText(plainTextEdit_collectionName)
        plainTextEdit_collectionURL = self.get('plainTextEdit_collectionURL','')
        self.mainWindow.plainTextEdit_collectionURL.setPlainText(plainTextEdit_collectionURL)
        plainTextEdit_contactEmail = self.get('plainTextEdit_contactEmail','')
        self.mainWindow.plainTextEdit_contactEmail.setPlainText(plainTextEdit_contactEmail)
        plainTextEdit_contactName = self.get('plainTextEdit_contactName','')
        self.mainWindow.plainTextEdit_contactName.setPlainText(plainTextEdit_contactName)
        plainTextEdit_copywriteLicense = self.get('plainTextEdit_copywriteLicense','')
        self.mainWindow.plainTextEdit_copywriteLicense.setPlainText(plainTextEdit_copywriteLicense)

        # QCheckBox
        # note: the fallback value of '' will trigger an unchecked condition in self.convertCheckState()
        checkBox_performWhiteBalance = self.convertCheckState(self.get('checkBox_performWhiteBalance',''))
        self.mainWindow.checkBox_performWhiteBalance.setCheckState(checkBox_performWhiteBalance)
        checkBox_vignettingCorrection = self.convertCheckState(self.get('checkBox_vignettingCorrection',''))
        self.mainWindow.checkBox_vignettingCorrection.setCheckState(checkBox_vignettingCorrection)
        checkBox_distortionCorrection = self.convertCheckState(self.get('checkBox_distortionCorrection',''))
        self.mainWindow.checkBox_distortionCorrection.setCheckState(checkBox_distortionCorrection)
        checkBox_chromaticAberrationCorrection = self.convertCheckState(self.get('checkBox_chromaticAberrationCorrection',''))
        self.mainWindow.checkBox_chromaticAberrationCorrection.setCheckState(checkBox_chromaticAberrationCorrection)
        checkBox_metaDataApplication = self.convertCheckState(self.get('checkBox_metaDataApplication',''))
        self.mainWindow.checkBox_metaDataApplication.setCheckState(checkBox_metaDataApplication)

        # QGroupbox (checkstate)
        #group_renameByBarcode = self.get('group_renameByBarcode','')
        group_renameByBarcode = self.get('group_renameByBarcode',False)
        self.mainWindow.group_renameByBarcode.setChecked(group_renameByBarcode)
        group_keepUnalteredRaw = self.get('group_keepUnalteredRaw',False)
        self.mainWindow.group_keepUnalteredRaw.setChecked(group_keepUnalteredRaw)
        group_saveProcessedJpg = self.get('group_saveProcessedJpg',False)
        self.mainWindow.group_saveProcessedJpg.setChecked(group_saveProcessedJpg)
        group_saveProcessedTIFF = self.get('group_saveProcessedTIFF',False)
        self.mainWindow.group_saveProcessedTIFF.setChecked(group_saveProcessedTIFF)
        group_saveProcessedPng = self.get('group_saveProcessedPng',False)
        self.mainWindow.group_saveProcessedPng.setChecked(group_saveProcessedPng)
        
        # QGroupbox (enablestate)
        group_barcodeDetection = self.convertEnabledState(self.get('group_barcodeDetection',''))
        self.mainWindow.group_barcodeDetection.setEnabled(group_barcodeDetection)
        group_colorCheckerDetection = self.convertEnabledState(self.get('group_colorCheckerDetection',''))
        self.mainWindow.group_colorCheckerDetection.setEnabled(group_colorCheckerDetection)
        group_verifyRotation = self.convertEnabledState(self.get('group_verifyRotation',''))
        self.mainWindow.group_verifyRotation.setEnabled(group_verifyRotation)
        group_equipmentDetection = self.convertEnabledState(self.get('group_equipmentDetection',''))
        self.mainWindow.group_equipmentDetection.setEnabled(group_equipmentDetection)
        # metaDataApplication should always be an option
        #group_metaDataApplication = self.convertEnabledState(self.get('group_metaDataApplication','true'))
        #self.mainWindow.group_metaDataApplication.setEnabled(group_metaDataApplication)

        # QSpinBox
        spinBox_catalogDigits = int(self.get('spinBox_catalogDigits', 6))
        self.mainWindow.spinBox_catalogDigits.setValue(spinBox_catalogDigits)
        
        # QDoubleSpinBox
        doubleSpinBox_focalDistance = float(self.get('doubleSpinBox_focalDistance', 25.5))
        self.mainWindow.doubleSpinBox_focalDistance.setValue(doubleSpinBox_focalDistance)

        # slider
        #value_LogoScaling = int(self.get('value_LogoScaling', 100))
        #self.settings.value_LogoScaling.setValue(value_LogoScaling)
        #self.scalingChanged(value_LogoScaling)

        # radiobutton
        #value_DarkTheme = self.get('value_DarkTheme', False)
        #self.settings.value_DarkTheme.setChecked(value_DarkTheme)
        #value_LightTheme = self.get('value_LightTheme', True)
        #self.settings.value_LightTheme.setChecked(value_LightTheme)

        # clean up
        # allow signals again
        [x.blockSignals(False) for x in children]

    def saveSettings(self):
        """ stores the preferences widget's selections to self.settings"""

        # save the version number
#        version = self.version
#        self.setValue('version', version)
#        # save the laste date we checked the version number
#        try:
#            date_versionCheck = self.parent.w.date_versionCheck
#        except AttributeError:  # first run may not yet have this saved.
#            date_versionCheck = ""
#        self.setValue('date_versionCheck', date_versionCheck)

        # QComboBox
        comboBox_imageOrientation = self.mainWindow.comboBox_imageOrientation.currentText()
        self.settings.setValue('comboBox_imageOrientation', comboBox_imageOrientation)
        comboBox_colorCheckerPosition = self.mainWindow.comboBox_colorCheckerPosition.currentText()
        self.settings.setValue('comboBox_colorCheckerPosition', comboBox_colorCheckerPosition)
        
        # QLineEdit
        lineEdit_catalogNumberPrefix = self.mainWindow.lineEdit_catalogNumberPrefix.text()
        self.settings.setValue('lineEdit_catalogNumberPrefix', lineEdit_catalogNumberPrefix)
        lineEdit_inputPath = self.mainWindow.lineEdit_inputPath.text()
        self.settings.setValue('lineEdit_inputPath', lineEdit_inputPath)
        lineEdit_pathUnalteredRaw = self.mainWindow.lineEdit_pathUnalteredRaw.text()
        self.settings.setValue('lineEdit_pathUnalteredRaw', lineEdit_pathUnalteredRaw)
        lineEdit_pathProcessedJpg = self.mainWindow.lineEdit_pathProcessedJpg.text()
        self.settings.setValue('lineEdit_pathProcessedJpg', lineEdit_pathProcessedJpg)
        lineEdit_pathProcessedTIFF = self.mainWindow.lineEdit_pathProcessedTIFF.text()
        self.settings.setValue('lineEdit_pathProcessedTIFF', lineEdit_pathProcessedTIFF)
        lineEdit_pathProcessedPng = self.mainWindow.lineEdit_pathProcessedPng.text()
        self.settings.setValue('lineEdit_pathProcessedPng', lineEdit_pathProcessedPng)

        # QPlainTextEdit        
        plainTextEdit_collectionName = self.mainWindow.plainTextEdit_collectionName.toPlainText()
        self.settings.setValue('plainTextEdit_collectionName', plainTextEdit_collectionName)
        plainTextEdit_collectionURL = self.mainWindow.plainTextEdit_collectionURL.toPlainText()
        self.settings.setValue('plainTextEdit_collectionURL', plainTextEdit_collectionURL)
        plainTextEdit_contactEmail = self.mainWindow.plainTextEdit_contactEmail.toPlainText()
        self.settings.setValue('plainTextEdit_contactEmail', plainTextEdit_contactEmail)
        plainTextEdit_contactName = self.mainWindow.plainTextEdit_contactName.toPlainText()
        self.settings.setValue('plainTextEdit_contactName', plainTextEdit_contactName)
        plainTextEdit_copywriteLicense = self.mainWindow.plainTextEdit_copywriteLicense.toPlainText()
        self.settings.setValue('plainTextEdit_copywriteLicense', plainTextEdit_copywriteLicense)

        # QCheckBox
        checkBox_performWhiteBalance = self.mainWindow.checkBox_performWhiteBalance.isChecked()
        self.settings.setValue('checkBox_performWhiteBalance', checkBox_performWhiteBalance)
        checkBox_vignettingCorrection = self.mainWindow.checkBox_vignettingCorrection.isChecked()
        self.settings.setValue('checkBox_vignettingCorrection', checkBox_vignettingCorrection)
        checkBox_distortionCorrection = self.mainWindow.checkBox_distortionCorrection.isChecked()
        self.settings.setValue('checkBox_distortionCorrection', checkBox_distortionCorrection)
        checkBox_chromaticAberrationCorrection = self.mainWindow.checkBox_chromaticAberrationCorrection.isChecked()
        self.settings.setValue('checkBox_chromaticAberrationCorrection', checkBox_chromaticAberrationCorrection)
        checkBox_metaDataApplication = self.mainWindow.checkBox_metaDataApplication.isChecked()
        self.settings.setValue('checkBox_metaDataApplication', checkBox_metaDataApplication)

        # QGroupbox (checkstate)
        group_renameByBarcode = self.mainWindow.group_renameByBarcode.isChecked()
        self.settings.setValue('group_renameByBarcode',group_renameByBarcode)
        group_keepUnalteredRaw = self.mainWindow.group_keepUnalteredRaw.isChecked()
        self.settings.setValue('group_keepUnalteredRaw',group_keepUnalteredRaw)
        group_saveProcessedTIFF = self.mainWindow.group_saveProcessedTIFF.isChecked()
        self.settings.setValue('group_saveProcessedTIFF',group_saveProcessedTIFF)
        group_saveProcessedJpg = self.mainWindow.group_saveProcessedJpg.isChecked()
        self.settings.setValue('group_saveProcessedJpg',group_saveProcessedJpg)
        group_saveProcessedPng = self.mainWindow.group_saveProcessedPng.isChecked()
        self.settings.setValue('group_saveProcessedPng',group_saveProcessedPng)

        # QGroupbox (enablestate)
        group_barcodeDetection = self.mainWindow.group_barcodeDetection.isEnabled()
        self.settings.setValue('group_barcodeDetection', group_barcodeDetection)
        group_colorCheckerDetection = self.mainWindow.group_colorCheckerDetection.isEnabled()
        self.settings.setValue('group_colorCheckerDetection',group_colorCheckerDetection)
        group_verifyRotation = self.mainWindow.group_verifyRotation.isEnabled()
        self.settings.setValue('group_verifyRotation',group_verifyRotation)
        group_equipmentDetection = self.mainWindow.group_equipmentDetection.isEnabled()
        self.settings.setValue('group_equipmentDetection',group_equipmentDetection)
        # metaDataApplication should always be an option
        #group_metaDataApplication = self.mainWindow.group_metaDataApplication.isEnabled()
        #self.settings.setValue('group_metaDataApplication',group_metaDataApplication)

        # QSpinBox
        spinBox_catalogDigits = self.mainWindow.spinBox_catalogDigits.value()
        self.settings.setValue('spinBox_catalogDigits', spinBox_catalogDigits)
        
        # QDoubleSpinBox
        doubleSpinBox_focalDistance = self.mainWindow.doubleSpinBox_focalDistance.value()
        self.settings.setValue('doubleSpinBox_focalDistance', doubleSpinBox_focalDistance)
        
        # slider
        #value_LogoScaling = self.mainWindow.value_LogoScaling.value()
        #self.settings.setValue('value_LogoScaling', value_LogoScaling)

        # radiobutton
        #value_DarkTheme = self.mainWindow.value_DarkTheme.isChecked()
        #self.settings.setValue('value_DarkTheme', value_DarkTheme)
        #value_LightTheme = self.mainWindow.value_LightTheme.isChecked()
        #self.settings.setValue('value_LightTheme', value_LightTheme)

app = QtWidgets.QApplication(sys.argv)
w = appWindow()
# check if there are theme settings
#if w.settings.get('value_DarkTheme', False):
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
w.show()

sys.exit(app.exec_())
