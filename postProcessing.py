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
import string
# UI libs
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog
from PyQt5.QtCore import (QSettings, Qt, QObject, QThreadPool, QEventLoop)
# image libs
import rawpy
from rawpy import LibRawNonFatalError, LibRawFatalError
import cv2
import numpy as np
# internal libs
from ui.postProcessingUI import Ui_MainWindow
from ui.imageDialog import Ui_Dialog
from libs.bcRead import bcRead
from libs.eqRead import eqRead
from libs.blurDetect import blurDetect
from libs.ccRead import ColorchipRead
from libs.folderMonitor import Folder_Watcher
from libs.folderMonitor import Save_Output_Handler
from libs.folderMonitor import New_Image_Emitter

from libs.boss_worker import (Boss, BCWorkerData, BlurWorkerData, EQWorkerData,
                         Job, BossSignalData, WorkerSignalData, WorkerErrorData)


class ImageDialog(QDialog):
    def __init__(self, img_array_object):
        super().__init__()
        self.init_ui(img_array_object)

    def init_ui(self, img_array_object):
        mb = Ui_Dialog()
        mb.setupUi(self)
        width, height = img_array_object.size
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(np.array(img_array_object), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        mb.label_Image.setPixmap(pixmap_image)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

class Image_Complete_Emitter(QtCore.QObject):
    """
    used to alert when image processing is complete
    """
    completed = QtCore.pyqtSignal()


class appWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.mainWindow = Ui_MainWindow()
        self.mainWindow.setupUi(self)

        # set up a threadpool
        self.threadPool = QThreadPool().globalInstance()
        # determine & assign a safe quantity of threads 75% of resources
        maxThreads = max([1, int(self.threadPool.maxThreadCount() * .75)])
        self.threadPool.setMaxThreadCount(maxThreads)
        print(f"Multithreading with maximum {self.threadPool.maxThreadCount()} threads")
        # initiate the persistant settings
        # todo update this when name is decided on
        self.settings = QSettings('AYUP', 'AYUP')
        self.settings.setFallbacksEnabled(False)    # File only, no fallback to registry.
        # populate the settings based on the previous preferences
        self.populateSettings()

        self.New_Image_Emitter = New_Image_Emitter()
        # initalize the folder_watcher using current user inputs
        self.setup_Folder_Watcher()
        # initalize the Save_Output_Handler using current user inputs
        self.setup_Output_Handler()
        self.image_queue = []
        self.Image_Complete_Emitter = Image_Complete_Emitter()
        self.Image_Complete_Emitter.completed.connect(self.process_from_queue)
        # fill in the previews
        self.updateCatalogNumberPreviews()
        prefix = self.mainWindow.lineEdit_catalogNumberPrefix.text()
        digits = int(self.mainWindow.spinBox_catalogDigits.value())
        self.bcRead = bcRead(prefix, digits, parent=self.mainWindow)
        self.blurDetect = blurDetect(parent=self.mainWindow)
        self.colorchipDetect = ColorchipRead(parent=self.mainWindow)
        self.eqRead = eqRead(parent=self.mainWindow)
        # self.ccRead = ccRead(parent=self.mainWindow)

        # assign applicable user settings for eqRead.
        # this function is here, for ease of slot assignment in pyqt designer
        self.reset_working_variables()
        # self.updateEqSettings()

        # create boss thread, it starts itself and is running the __boss_function
        self.boss_thread = Boss(self.threadPool)
        # setup Boss's signals
        self.boss_thread.signals.boss_started.connect(self.handle_boss_started)
        self.boss_thread.signals.boss_closed.connect(self.handle_boss_finished)
        self.boss_thread.signals.job_started.connect(self.handle_job_started)
        self.boss_thread.signals.job_finished.connect(self.handle_job_finished)
        self.boss_thread.signals.job_result.connect(self.handle_job_result)
        self.boss_thread.signals.job_error.connect(self.handle_job_error)
        # start the boss thread's "event loop"
        # https://doc.qt.io/qt-5/qthread.html#start
        self.boss_thread.start()
        print('boss thread started')

#       self.versionCheck()

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

    def setup_Folder_Watcher(self, raw_image_patterns=None):
        """
        initiates self.foldeR_watcher with user inputs.
        """
        if not raw_image_patterns:
            raw_image_patterns = ['*.tmp', '*.TMP',
                                  '*.cr2', '*.CR2',
                                  '*.tiff', '*.TIFF',
                                  '*.nef', '*.NEF',
                                  '*.orf', '*.ORF']

        lineEdit_inputPath = self.mainWindow.lineEdit_inputPath.text()
        self.folder_watcher = Folder_Watcher(lineEdit_inputPath, raw_image_patterns)
        self.folder_watcher.emitter.new_image_signal.connect(self.queue_image)
        self.mainWindow.pushButton_beginMonitoring.clicked.connect(self.folder_watcher.run)
        
    def setup_Output_Handler(self):
        """
        initiates self.save_output_handler with user inputs.
        """

        group_keepUnalteredRaw = self.mainWindow.group_keepUnalteredRaw.isChecked()
        lineEdit_pathUnalteredRaw = self.mainWindow.lineEdit_pathUnalteredRaw.text()

        group_saveProcessedJpg = self.mainWindow.group_saveProcessedJpg.isChecked()
        lineEdit_pathProcessedJpg = self.mainWindow.lineEdit_pathProcessedJpg.text()

        group_saveProcessedTIFF = self.mainWindow.group_saveProcessedTIFF.isChecked()
        lineEdit_pathProcessedTIFF = self.mainWindow.lineEdit_pathProcessedTIFF.text()

        group_saveProcessedPng = self.mainWindow.group_saveProcessedPng.isChecked()
        lineEdit_pathProcessedPng = self.mainWindow.lineEdit_pathProcessedPng.text()

        output_map = [(group_keepUnalteredRaw, lineEdit_pathUnalteredRaw, None),
                      (group_saveProcessedJpg, lineEdit_pathProcessedJpg, '.jpg'),
                      (group_saveProcessedTIFF, lineEdit_pathProcessedTIFF, '.tiff'),
                      (group_saveProcessedPng, lineEdit_pathProcessedPng, '.png')]
        dupNamingPolicy = self.mainWindow.comboBox_dupNamingPolicy.currentText()

        self.save_output_handler = Save_Output_Handler(output_map, dupNamingPolicy)

    def handle_blur_result(self, result):
        self.is_blurry = result

    def alert_blur_finished(self):
        """ called when the results are in from blur detection. """
        if self.is_blurry:
            notice_title = 'Blurry Image Warning'
            if self.bc_code:
                notice_text = f'Warning, {self.bc_code} is blurry.'
            else:
                notice_text = f'Warning, {self.base_file_name} is blurry.'
            detail_text = f'Blurry Image Path: {self.img_path}'
            self.userNotice(notice_text, notice_title, detail_text)

    def handle_bc_result(self, result):
        self.bc_code = result

    def alert_bc_finished(self):
        """ called when the results are in from bcRead."""
        print('bc detection finished')

    def handle_eq_result(self, result):
        # this is the corrected image array
        self.im = result
        # should probably use this to store the updated image

    def alert_eq_finished(self):
        """ called when the results are in from eqRead."""
        print('eq corrections finished')

    # boss signal handlers
    def handle_boss_started(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            if boss_signal_data.signal_data is str:
                print(boss_signal_data.signal_data)

    def handle_boss_finished(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            if boss_signal_data.signal_data is str:
                print(boss_signal_data.signal_data)

    def handle_job_started(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            if isinstance(boss_signal_data.signal_data, WorkerSignalData):
                worker_signal_data = boss_signal_data.signal_data
                # print(worker_signal_data.worker_name)
                print(worker_signal_data.signal_data)

    def handle_job_finished(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            if isinstance(boss_signal_data.signal_data, WorkerSignalData):
                worker_signal_data = boss_signal_data.signal_data
                # print(worker_signal_data.worker_name)
                print(worker_signal_data.signal_data)

    def handle_job_result(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            if isinstance(boss_signal_data.signal_data, WorkerSignalData):
                worker_signal_data = boss_signal_data.signal_data
                if worker_signal_data.worker_name == 'bc_worker':
                    self.handle_bc_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'blur_worker':
                    self.handle_blur_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'eq_worker':
                    self.handle_eq_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'save_worker':
                    pass  # ???

    def handle_job_error(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            if isinstance(boss_signal_data.signal_data, WorkerSignalData):
                worker_signal_data = boss_signal_data.signal_data
                print(f'error in worker: {worker_signal_data.worker_name}')
                print(f'caught exception: {worker_signal_data.signal_data.exception}')
                print(f'error type: {worker_signal_data.signal_data.exctype}')
                if worker_signal_data.signal_data is WorkerErrorData:
                    worker_error_data = worker_signal_data.signal_data
                    print('error data:')
                    print(f'exception type: {str(worker_error_data.exctype)}')
                    print(f'value: {str(worker_error_data.value)}')
                    print(f'formatted exception: {str(worker_error_data.format_exc)}')

    def white_balance_image(self, im, whiteR, whiteG, whiteB):
        """
        Given an image array, and RGB values for the lightest portion of a
        color standard, returns the white balanced image array.
        
        :param im: An image array
        :type im: ndarray
        :param whiteR: the red pixel value for the lightest portion of the
        color standard
        :type whiteR: int
        :param whiteG: the green pixel value for the lightest portion of the
        color standard
        :type whiteG: int
        :param whiteB: the blue pixel value for the lightest portion of the
        color standard
        :type whiteB: int
        
        see: https://stackoverflow.com/questions/54470148/white-balance-a-photo-from-a-known-point
        """
        
        #lum = (whiteR + whiteG + whiteB)/3
        lum = (whiteR *0.2126 + whiteG *0.7152 + whiteB *0.0722)
        imgR = im[..., 0].copy()
        imgG = im[..., 1].copy()
        imgB = im[..., 2].copy()
        imgR = imgR * lum / whiteR
        imgG = imgG * lum / whiteG
        imgB = imgB * lum / whiteB
        # scale each channel
        im[..., 0] = imgR
        im[..., 1] = imgG
        im[..., 2] = imgB

        return im

    def scale_images_with_info(self, im, largest_dim=1875):
        """
        Function that scales images proportionally, and returns both the original image size as a tuple of image
        dimensions, and the scaled down image.
        :param im: Image to be scaled down.
        :type im: cv2 Image
        :param largest_dim: The largest dimension to be scaled down to.
        :type largest_dim: int
        :return: Returns both the original image dimensions as a tuple and the scaled image.
        :rtype: tuple, cv2 Image
        """
        image_height, image_width = im.shape[0:2]

        if image_width > image_height:
            reduced_im = cv2.resize(im, (largest_dim, round((largest_dim / image_width) * image_height)),
                                    interpolation=cv2.INTER_AREA)
        else:
            reduced_im = cv2.resize(im, (round((largest_dim / image_height) * image_width), largest_dim),
                                    interpolation=cv2.INTER_AREA)
        return (image_width, image_height), reduced_im

    def queue_image(self, imgPath):
        """
        adds an image path to self.image_queue. Called when a new image is
        queued for processing. After adding the item it also attempts to
        process_from_queue. If self.processing_image == False it starts the
        process.
        """
        self.image_queue.append(imgPath)
        self.process_from_queue()

    def process_from_queue(self):
        """
        attempts to process an image in self.image_queue
        """
        if not self.processing_image:
            # if processing is not ongoing reset_working_variables
            self.reset_working_variables()
            if len(self.image_queue) > 0:
                to_process = self.image_queue.pop(0)
                self.processImage(to_process)

    def reset_working_variables(self):
        """
        sets all class variables relevant to the current working image to None.
        """
        self.base_file_name = None
        self.bc_code = None
        self.im = None
        self.img_path = None
        self.is_blurry = None  # could be helpful for 'line item warnings'
        self.cc_quadrant = None
        self.cc_avg_white = None
        self.processing_image = False
        print('working variables reset')

    def processImage(self, img_path):
        """
        given a path to an unprocessed image, performs the appropriate
        processing steps.
        """
        try:
            im = self.openImageFile(img_path)
        except LibRawFatalError:
            # empty path passed
            if img_path == '':
                return
            text = 'Corrupted or incompatible image file.'
            title = 'Error opening file'
            detail_text = f'LibRawFatalError opening: {img_path}\nUsually this indicates a corrupted input image file.'
            self.userNotice(text, title, detail_text)
            return
        
        self.processing_image = True
        print(f'processing: {img_path}')
        # debugging, save 'raw-ish' version of jpg before processing
        for_cv2_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('input.jpg', for_cv2_im)
        self.img_path = img_path
        file_name, file_ext = os.path.splitext(img_path)
        self.base_file_name = os.path.basename(file_name)

        # converting to greyscale
        original_size, reduced_img = self.scale_images_with_info(im)
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if self.mainWindow.group_renameByBarcode:
            # retrieve the barcode values from image
            bc_worker_data = BCWorkerData(grey)
            bc_job = Job('bc_worker', bc_worker_data, self.bcRead.decodeBC)
            self.boss_thread.request_job(bc_job)

        if self.mainWindow.checkBox_blurDetection:
            # test for bluryness
            blur_threshold = self.mainWindow.doubleSpinBox_blurThreshold.value()
            blur_worker_data = BlurWorkerData(grey, blur_threshold)
            blur_job = Job('blur_worker', blur_worker_data, self.blurDetect.blur_check)
            self.boss_thread.request_job(blur_job)

        if self.mainWindow.checkBox_lensCorrection:
            # equipment corrections
            cm_distance = self.mainWindow.doubleSpinBox_focalDistance.value()
            m_distance = round(cm_distance / 100, 5)
            eq_worker_data = EQWorkerData(im, img_path, m_distance)
            eq_job = Job('eq_worker', eq_worker_data, self.eqRead.lensCorrect)
            self.boss_thread.request_job(eq_job)

        if self.mainWindow.group_colorCheckerDetection:
            # colorchecker functions
#            cc_size = self.colorchipDetect.predict_colorchip_size(reduced_img)
#            if cc_size == 'big':
#                cc_position, cropped_cc = self.colorchipDetect.process_colorchip_big(im)
#            else:
#                cc_position, cropped_cc = self.colorchipDetect.process_colorchip_small(reduced_img, original_size)
            if self.mainWindow.radioButton_colorCheckerSmall.isChecked():
                cc_position, cropped_cc = self.colorchipDetect.process_colorchip_small(reduced_img, original_size)
            else:
                cc_position, cropped_cc = self.colorchipDetect.process_colorchip_big(im)
            #for_cv2_im_cc = cv2.cvtColor(np.array(cropped_cc), cv2.COLOR_RGB2BGR)
            #cv2.imwrite('cc.jpg', for_cv2_im_cc)
            self.cc_quadrant = self.colorchipDetect.predict_color_chip_quadrant(original_size, cc_position)
            self.cc_avg_white = self.colorchipDetect.predict_color_chip_whitevals(cropped_cc)
            print(f"CC | Position: {cc_position}, Quadrant: {self.cc_quadrant} | AVG White: {self.cc_avg_white}")
        """
        waiting on bcWorker happens in Boss thread
        """
        save_job = Job('save_worker', None, self.save_when_finished)
        self.boss_thread.request_job(save_job)

    def save_finished(self):
        print(f'saving {self.img_path} has finished.')

    def save_when_finished(self):
        """
        combines async results and saves final output.
        """
        im = self.im
        print(self.cc_avg_white)
        im = self.white_balance_image(im, *self.cc_avg_white)
        # reminder to address the quadrant checker here
        if self.mainWindow.group_verifyRotation.isChecked():
            user_def_loc = self.mainWindow.comboBox_colorCheckerPosition.currentText()
            quad_map = ['Upper right',
                        'Lower right',
                        'Lower left',
                        'Upper left']
            user_def_quad = quad_map.index(user_def_loc) + 1
            # cc_quadrant starts at first,
            im = self.orient_image(im, self.cc_quadrant, user_def_quad)

        self.save_output_handler.save_output_images(im, self.img_path,
                                                    self.bc_code)
        # inform the app when image processing is complete
        self.processing_image = False
        self.Image_Complete_Emitter.completed.emit()

    def orient_image(self, im, picker_quadrant, desired_quadrant):
        '''
        corrects image rotation using the position of the color picker.
        picker_quadrant = the known quadrant of a color picker location,
        desired_quadrant = the position the color picker should be in.
        '''
        try:
            rotation_qty = (picker_quadrant - desired_quadrant)
            im = np.rot90(im, rotation_qty)
        except TypeError:
            # alert the user of an issue
            msg_text = 'Could not infer color checker location'
            title_text = 'Error finding color checker'
            detail_text = f'TypeError retrieving quadrant from {self.img_path}'
            self.userNotice(msg_text, title_text, detail_text)
        return im

    def testFunction(self):
        """ a development assistant function, connected to a GUI button
        used to test various functions before complete GUI integration."""

        img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "Open Sample Image")

        import time
        #  start a timer
        startTime = time.time()
        self.queue_image(img_path)
        #self.processImage(img_path)
        # finish timer

    def openImageFile(self, imgPath,
                      demosaic=rawpy.DemosaicAlgorithm.AHD):
        """ given an image path, attempts to return a numpy array image object
        """
        usr_gamma = self.mainWindow.doubleSpinBox_gammaValue.value()
        gamma_value = (usr_gamma, usr_gamma)
        try:  # use rawpy to convert raw to openCV
            with rawpy.imread(imgPath) as raw:
                im = raw.postprocess(chromatic_aberration=(1, 1),
                                      demosaic_algorithm=demosaic,
                                      gamma=gamma_value,
                                      output_color=rawpy.ColorSpace.raw)

        # if it is not a raw format, just try and open it.
        except LibRawNonFatalError:
            bgr = cv2.imread(imgPath)
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except LibRawFatalError:
            raise
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
        if imgPath == '':
            return
        im = self.openImageFile(imgPath)
        original_size, reduced_img = self.scale_images_with_info(im)
        grey = cv2.cvtColor(reduced_img, cv2.COLOR_BGR2GRAY)
        try:
            bcStatus = self.bcRead.testFeature(grey)
        except Exception as e:
            bcStatus = False
            print('bcRead returned exception:')
            print(e)
        try:
            blurStatus = self.blurDetect.testFeature(grey)
        except Exception as e:
            blurStatus = False
            print('blurStatus returned exception:')
            print(e)
        try:
            # NOTE This currently checks radio button for large / small
            # If the size determiner becomes 
            if self.mainWindow.radioButton_colorCheckerSmall.isChecked():
                cc_size = 'small'
            else:
                cc_size = 'big'
            ccStatus, cropped_img = self.colorchipDetect.test_feature(reduced_img, original_size, cc_size)
            # if the status was true, use the image dialog box to ask the
            # user if it properly detected the color chip.
            if ccStatus:
                mb = ImageDialog(cropped_img)
                mb.exec()
                if mb:
                    ccStatus = True
                else:
                    ccStatus = False
        except Exception as e:
            ccStatus = False
            print('ccStatus returned exception:')
            print(e)
        try:                
            eqStatus = self.eqRead.testFeature(imgPath)
        except Exception as e:
            eqStatus = False
            print('eqStatus returned exception:')
            print(e)

        self.mainWindow.group_barcodeDetection.setEnabled(bcStatus)
        self.mainWindow.group_blurDetection.setEnabled(blurStatus)
        self.mainWindow.group_colorCheckerDetection.setEnabled(ccStatus)
        self.mainWindow.group_verifyRotation.setEnabled(ccStatus)
        self.mainWindow.checkBox_performWhiteBalance.setEnabled(ccStatus)
        self.mainWindow.groupBox_colorCheckerSize.setEnabled(ccStatus)
        self.mainWindow.group_equipmentDetection.setEnabled(eqStatus)

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
            self.suffix_lookup = {n+1: ch for n, ch in enumerate(string.ascii_lowercase)}
        elif dupNamingPolicy == 'append UPPER case letter':
            dupPreviewEnd = 'A'
            self.suffix_lookup = {n+1: ch for n, ch in enumerate(string.ascii_uppercase)}
        elif dupNamingPolicy == 'append Number with underscore':
            dupPreviewEnd = '1'
            self.suffix_lookup = False
        elif dupNamingPolicy == 'OVERWRITE original image with newest':
            dupPreviewEnd = ''
            self.suffix_lookup = False
        else:
            self.suffix_lookup = False
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

    def userAsk(self, text, title='', detailText=None):
        """ a general user dialog with yes / cancel options"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText(text)
        msg.setWindowTitle(title)
        if detailText != None:
            msg.setDetailedText(detailText)

        msg.setDefaultButton(QMessageBox.No)
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            return True
        elif reply == QMessageBox.No:
            return False
        elif reply == QMessageBox.Cancel:
            return False
        elif reply == QMessageBox.Retry:
            return True
        else:
            return "cancel"

    def userNotice(self, text, title='', detailText = None):
        """ a general user notice """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(text)
        if detailText != None:
            msg.setDetailedText(detailText)
        #msg.setInformativeText("This is additional information")
        msg.setWindowTitle(title)
        reply = msg.exec_()
        return reply

    # Functions related to the saving and retrieval of preference settings

    def has(self, key):
        return self.settings.contains(key)

    def setValue(self, key, value):
        return self.settings.setValue(key, value)

    def get(self, key, altValue=None):
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
        #checkBox_vignettingCorrection = self.convertCheckState(self.get('checkBox_vignettingCorrection',''))
        #self.mainWindow.checkBox_vignettingCorrection.setCheckState(checkBox_vignettingCorrection)
        #checkBox_distortionCorrection = self.convertCheckState(self.get('checkBox_distortionCorrection',''))
        #self.mainWindow.checkBox_distortionCorrection.setCheckState(checkBox_distortionCorrection)
        #checkBox_chromaticAberrationCorrection = self.convertCheckState(self.get('checkBox_chromaticAberrationCorrection',''))
        #self.mainWindow.checkBox_chromaticAberrationCorrection.setCheckState(checkBox_chromaticAberrationCorrection)
        checkBox_lensCorrection = self.convertCheckState(self.get('checkBox_lensCorrection',''))
        self.mainWindow.checkBox_lensCorrection.setCheckState(checkBox_lensCorrection)
        checkBox_metaDataApplication = self.convertCheckState(self.get('checkBox_metaDataApplication',''))
        self.mainWindow.checkBox_metaDataApplication.setCheckState(checkBox_metaDataApplication)
        checkBox_blurDetection = self.convertCheckState(self.get('checkBox_blurDetection',''))
        self.mainWindow.checkBox_blurDetection.setCheckState(checkBox_blurDetection)

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
        group_verifyRotation_checkstate = self.get('group_verifyRotation_checkstate', False)
        self.mainWindow.group_verifyRotation .setChecked(group_verifyRotation_checkstate)

        # QGroupbox (enablestate)
        group_barcodeDetection = self.convertEnabledState(self.get('group_barcodeDetection',False))
        self.mainWindow.group_barcodeDetection.setEnabled(group_barcodeDetection)
        group_colorCheckerDetection = self.convertEnabledState(self.get('group_colorCheckerDetection',False))
        self.mainWindow.group_colorCheckerDetection.setEnabled(group_colorCheckerDetection)
        group_verifyRotation = self.convertEnabledState(self.get('group_verifyRotation',False))
        self.mainWindow.group_verifyRotation.setEnabled(group_verifyRotation)
        group_equipmentDetection = self.convertEnabledState(self.get('group_equipmentDetection',False))
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
        doubleSpinBox_blurThreshold = float(self.get('doubleSpinBox_blurThreshold', 0.08))
        self.mainWindow.doubleSpinBox_blurThreshold.setValue(doubleSpinBox_blurThreshold)
        doubleSpinBox_gammaValue = float(self.get('doubleSpinBox_gammaValue', 2.20))
        self.mainWindow.doubleSpinBox_gammaValue.setValue(doubleSpinBox_gammaValue)

        # slider
        #value_LogoScaling = int(self.get('value_LogoScaling', 100))
        #self.settings.value_LogoScaling.setValue(value_LogoScaling)
        #self.scalingChanged(value_LogoScaling)

        # radiobutton
        #value_DarkTheme = self.get('value_DarkTheme', False)
        #self.settings.value_DarkTheme.setChecked(value_DarkTheme)
        #value_LightTheme = self.get('value_LightTheme', True)
        #self.settings.value_LightTheme.setChecked(value_LightTheme)
        radioButton_colorCheckerSmall = self.get('radioButton_colorCheckerSmall', True)
        self.mainWindow.radioButton_colorCheckerSmall.setChecked(radioButton_colorCheckerSmall)
        radioButton_colorCheckerLarge = self.get('radioButton_colorCheckerLarge', False)
        self.mainWindow.radioButton_colorCheckerLarge.setChecked(radioButton_colorCheckerLarge)


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
        #checkBox_vignettingCorrection = self.mainWindow.checkBox_vignettingCorrection.isChecked()
        #self.settings.setValue('checkBox_vignettingCorrection', checkBox_vignettingCorrection)
        #checkBox_distortionCorrection = self.mainWindow.checkBox_distortionCorrection.isChecked()
        #self.settings.setValue('checkBox_distortionCorrection', checkBox_distortionCorrection)
        #checkBox_chromaticAberrationCorrection = self.mainWindow.checkBox_chromaticAberrationCorrection.isChecked()
        #self.settings.setValue('checkBox_chromaticAberrationCorrection', checkBox_chromaticAberrationCorrection)
        checkBox_lensCorrection = self.mainWindow.checkBox_lensCorrection.isChecked()
        self.settings.setValue('checkBox_lensCorrection', checkBox_lensCorrection)
        checkBox_metaDataApplication = self.mainWindow.checkBox_metaDataApplication.isChecked()
        self.settings.setValue('checkBox_metaDataApplication', checkBox_metaDataApplication)
        checkBox_blurDetection = self.mainWindow.checkBox_blurDetection.isChecked()
        self.settings.setValue('checkBox_blurDetection', checkBox_blurDetection)

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
        group_verifyRotation_checkstate = self.mainWindow.group_verifyRotation.isChecked()
        self.settings.setValue('group_verifyRotation_checkstate', group_verifyRotation_checkstate)

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
        doubleSpinBox_blurThreshold = self.mainWindow.doubleSpinBox_blurThreshold.value()
        self.settings.setValue('doubleSpinBox_blurThreshold', doubleSpinBox_blurThreshold)
        doubleSpinBox_gammaValue = self.mainWindow.doubleSpinBox_gammaValue.value()
        self.settings.setValue('doubleSpinBox_gammaValue', doubleSpinBox_gammaValue)

        # slider
        #value_LogoScaling = self.mainWindow.value_LogoScaling.value()
        #self.settings.setValue('value_LogoScaling', value_LogoScaling)

        # radiobutton
        #value_DarkTheme = self.mainWindow.value_DarkTheme.isChecked()
        #self.settings.setValue('value_DarkTheme', value_DarkTheme)
        #value_LightTheme = self.mainWindow.value_LightTheme.isChecked()
        #self.settings.setValue('value_LightTheme', value_LightTheme)
        radioButton_colorCheckerLarge = self.mainWindow.radioButton_colorCheckerLarge.isChecked()
        self.settings.setValue('radioButton_colorCheckerLarge', radioButton_colorCheckerLarge)
        radioButton_colorCheckerSmall = self.mainWindow.radioButton_colorCheckerSmall.isChecked()
        self.settings.setValue('radioButton_colorCheckerSmall', radioButton_colorCheckerSmall)
        # cleanup, after changes are made, should re-initalize some classes
        self.setup_Folder_Watcher()
        self.setup_Output_Handler()

app = QtWidgets.QApplication(sys.argv)
w = appWindow()
# check if there are theme settings
#if w.settings.get('value_DarkTheme', False):
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
w.show()

sys.exit(app.exec_())
