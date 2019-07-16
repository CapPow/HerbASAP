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

import time
import os
import sys
import string
import glob
import re
from shutil import move as shutil_move
# UI libs
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog
from PyQt5.QtCore import (QSettings, Qt, QObject, QThreadPool, QEventLoop,
                          QMimeData)
# image libs
import rawpy
from rawpy import LibRawNonFatalError, LibRawFatalError
import cv2
import numpy as np
# internal libs
from ui.postProcessingUI import Ui_MainWindow
from ui.imageDialogUI import Ui_Dialog_image
from ui.noBcDialogUI import Ui_Dialog_noBc
from libs.bcRead import bcRead
from libs.eqRead import eqRead
from libs.blurDetect import blurDetect
from libs.ccRead import ColorchipRead
from libs.folderMonitor import Folder_Watcher
from libs.folderMonitor import New_Image_Emitter
from libs.boss_worker import (Boss, BCWorkerData, BlurWorkerData, EQWorkerData,
                              Job, BossSignalData, WorkerSignalData,
                              WorkerErrorData, SaveWorkerData)
from libs.metaRead import MetaRead

class ImageDialog(QDialog):
    def __init__(self, img_array_object):
        super().__init__()
        self.init_ui(img_array_object)

    def init_ui(self, img_array_object):
        mb = Ui_Dialog_image()
        mb.setupUi(self)
        width, height = img_array_object.size
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(np.array(img_array_object), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        mb.label_Image.setPixmap(pixmap_image)

#    def retranslateUi(self, Dialog):
#        _translate = QtCore.QCoreApplication.translate
#        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))


class BcDialog(QDialog):
    """
    a simple user dialog, for asking what the user to enter a barcode value.
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.mb = Ui_Dialog_noBc()
        self.mb.setupUi(self)
        self.mb.lineEdit_userBC.setFocus(True)

    def ask_for_bc(self):
        dialog = self.exec()
        if dialog:
            result = self.mb.lineEdit_userBC.text()
            return result
        else:
            return None


class Image_Complete_Emitter(QtCore.QObject):
    """
    used to alert when image processing is complete
    """
    completed = QtCore.pyqtSignal()


class Timer_Emitter(QtCore.QObject):
    """
    used to alert start and stop for entire processing time
    """
    timerStart = QtCore.pyqtSignal()
    timerStop = QtCore.pyqtSignal()


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
        ###
        # initalize the folder_watcher using current user inputs
        ###
        self.setup_Folder_Watcher()
        self.setup_Output_Handler()
        ###
        # signaling for the to process image queue
        ###
        self.image_queue = []
        self.New_Image_Emitter = New_Image_Emitter()
        self.Image_Complete_Emitter = Image_Complete_Emitter()
        self.Image_Complete_Emitter.completed.connect(self.process_from_queue)
        # using signals to start/stop timers
        self.Timer_Emitter = Timer_Emitter()
        self.Timer_Emitter.timerStart.connect(self.start_timer)
        self.Timer_Emitter.timerStop.connect(self.stop_timer)
        self.timer_start = time.time()
        ####
        #set up bcRead & associated ui
        ###
        self.updateCatalogNumberPreviews()
        patterns = self.retrieve_bc_patterns()
        self.fill_patterns(patterns)
        backend = self.mainWindow.comboBox_bcBackend.currentText()
        self.bcRead = bcRead(patterns, backend, parent=self.mainWindow)
        ###
        #
        ###
        self.blurDetect = blurDetect(parent=self.mainWindow)
        self.colorchipDetect = ColorchipRead(parent=self.mainWindow)
        self.eqRead = eqRead(parent=self.mainWindow)
        self.metaRead = MetaRead(parent=self.mainWindow)

        self.reset_working_variables()
        ###
        # job manaager "boss_thread", it starts itself and is running the __boss_function
        ###
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
        
        # setup static UI buttons
        self.mainWindow.toolButton_removePattern.pressed.connect(self.remove_pattern)
        self.mainWindow.toolButton_addPattern.pressed.connect(self.add_pattern)

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

    def start_timer(self):
        self.timer_start = time.time()

    def stop_timer(self):
        end_time = time.time()
        run_time = round(end_time - self.timer_start, 3)
        self.mainWindow.label_processtime.setText(str(run_time))
        print(f'Elapsed runtime: {run_time}')

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
        self.mainWindow.pushButton_toggleMonitoring.clicked.connect(self.toggle_folder_monitoring)
        
    def toggle_folder_monitoring(self):
        pushButton = self.mainWindow.pushButton_toggleMonitoring
        if self.folder_watcher.is_monitoring:
            pushButton.setText('Begin folder monitoring')
            self.folder_watcher.stop()
        else:  # if no input path, ask the user for it.
            if self.mainWindow.lineEdit_inputPath.text() == '':
                self.mainWindow.pushButton_inputPath.click()
                # if they insist on an empty path give up on folder monitoring.
                if self.mainWindow.lineEdit_inputPath.text() == '':
                    return
                self.setup_Folder_Watcher()
            pushButton.setText(' Stop folder monitoring')
            self.folder_watcher.run()


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
        # each value is a tuple containing (bool if checked, path, extension)
        self.output_map = {
                '.tif': (group_saveProcessedTIFF, lineEdit_pathProcessedTIFF),
                '.png': (group_saveProcessedPng, lineEdit_pathProcessedPng),
                '.jpg': (group_saveProcessedJpg, lineEdit_pathProcessedJpg),
                '.raw': (group_keepUnalteredRaw, lineEdit_pathUnalteredRaw)}
        dupNamingPolicy = self.mainWindow.comboBox_dupNamingPolicy.currentText()

        # establish self.suffix_lookup according to dupNamingPolicy
        # given an int (count of how many files have exact matching names,
        # returns an appropriate file name suffix)
        if dupNamingPolicy == 'LOWER case letter':
            self.suffix_lookup = lambda x: {n+1: ch for n, ch in enumerate(string.ascii_lowercase)}.get(x)
        elif dupNamingPolicy == 'UPPER case letter':
            self.suffix_lookup = lambda x: {n+1: ch for n, ch in enumerate(string.ascii_uppercase)}.get(x)
        elif dupNamingPolicy == 'Number with underscore':
            self.suffix_lookup = lambda x: f'_{x}'
        elif dupNamingPolicy == 'OVERWRITE original':
            self.suffix_lookup = lambda x: ''
        else:
            self.suffix_lookup = False

    def save_output_images(self, im, im_base_names, orig_img_path, orig_im_ext,
                           meta_data=None):
        """
        Function that saves processed images to the appropriate format and
        locations.
        :param im: Processed Image array to be saved.
        :type im: cv2 Array
        :param im_base_names: the destination file(s) base names. Usually a
        catalog number. Passed in as a list of strings.
        :type im_base_names: list
        :param orig_img_path: Path to the original image
        :type orig_img_path: str, path object
        :param ext: Extension of the original image with the "."
        :type ext: str
        :param meta_data: Optional, metadata dictionary organized with
        keys as destination metadata tag names, values as key value.
        :type meta_data: dict
        """
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        output_map = self.output_map
        # todo pass these off to boss_worker
        # for each bc_value or file name passed in
        for bc in im_base_names:
            # breakout output_map into a list of tuples containing (ext, path)
            # '.raw': (group_keepUnalteredRaw, lineEdit_pathUnalteredRaw)
            to_save = [(x,y[1]) for x, y in output_map.items() if y[0]]
            # store how many save_jobs are expected to global variable
            self.working_save_jobs = self.working_save_jobs
            if self.working_save_jobs < 1:  # condition when no outputs saved
                self.handle_save_result('')
            #  '.jpg':(group_saveProcessedJpg, lineEdit_pathProcessedJpg),
            for ext, path in to_save:
                # not sure of slick way to avoid checking ext twice
                if ext == '.raw':
                    ext = orig_im_ext
                fileQty = len(glob.glob(f'{path}//{bc}*{ext}'))
                if fileQty > 0:
                    new_file_suffix = self.suffix_lookup(fileQty)
                    new_file_base_name = f'{bc}{new_file_suffix}'
                    new_file_name = f'{path}//{new_file_base_name}{ext}'
                else:
                    new_file_name = f'{path}//{bc}{ext}'
                if ext == orig_im_ext:
                    # if it is a raw move just perform it
                    try:
                        shutil_move(orig_img_path, new_file_name)
                    except FileNotFoundError:
                        # condition when multiple bcs try to move > once.
                        pass
                    #  treat this process as if it was passed to boss_worker
                    self.handle_save_result('')
                else:
                    # if it a cv2 save function pass it to boss_worker
                    save_worker_data = SaveWorkerData(new_file_name, im)
                    save_job = Job('save_worker', save_worker_data, cv2.imwrite)
                    self.boss_thread.request_job(save_job)
                    #cv2.imwrite(new_file_name,im)

    def handle_blur_result(self, result):
        is_blurry = result['isblurry']
        if is_blurry:
            notice_title = 'Blurry Image Warning'
            if self.bc_code:
                notice_text = f'Warning, {self.bc_code} is blurry.'
            else:
                notice_text = f'Warning, {self.img_path} is blurry.'
            detail_text = f'laplacian={result["laplacian"]}\nnormalized laplacian={result["lapNorm"]}\nimg variance ={result["imVar"]}\n {self.base_file_name}'
            self.userNotice(notice_text, notice_title, detail_text)
        self.is_blurry = is_blurry
        self.mainWindow.label_isBlurry.setText(str(is_blurry))
        self.mainWindow.label_laplacian.setText(str(round(result['laplacian'], 3)))
        self.mainWindow.label_imvar.setText(str(round(result['imVar'], 3)))
        self.mainWindow.label_lapnorm.setText(str(round(result['lapNorm'], 3)))

    def handle_pp_result(self, result):
        # if no bc_code was found or checking for it is turned off
        if self.bc_code:
            names = self.bc_code
        else: # name based on base_file_name
            names = [self.base_file_name]
        self.save_output_images(self.im, names, self.img_path, self.ext)

    def handle_save_result(self, result):
        """ called when the the save_worker finishes up """
        # tick off one working_save_job 
        self.working_save_jobs -= 1
        # if we're down to 0 then wrap up
        if self.working_save_jobs < 1:
            # inform the app when image processing is complete
            self.processing_image = False
            # these are happening too soon
            self.Image_Complete_Emitter.completed.emit()
            self.Timer_Emitter.timerStop.emit()

    def alert_blur_finished(self):
        """ called when the results are in from blur detection. """
 
    def handle_bc_result(self, result):
        if not result:
            userDialog = BcDialog()
            result = [userDialog.ask_for_bc()]
#            self.bc_code = None
#            self.bc_code = result
#            notice_title = 'No Barcode Warning'
#            notice_text = f'Warning, No Barcode found!'
#            detail_text = f'No barcode found in image named: {self.img_path}'
#            self.userNotice(notice_text, notice_title, detail_text)
            # todo, have user entry option here return and store as result
        self.bc_code = result
        self.mainWindow.label_barcodes.setText(', '.join(result))

    def alert_bc_finished(self):
        """ called when the results are in from bcRead."""

    def handle_eq_result(self, result):
        # this is the corrected image array
        self.im = result
        # should probably use this to store the updated image

    def alert_eq_finished(self):
        """ called when the results are in from eqRead."""

    # boss signal handlers
    def handle_boss_started(self, boss_signal_data):
        pass
        #if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            #if boss_signal_data.signal_data is str:


    def handle_boss_finished(self, boss_signal_data):
        pass
        #if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            #if boss_signal_data.signal_data is str:

    def handle_job_started(self, boss_signal_data):
        pass
        #if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
            #if isinstance(boss_signal_data.signal_data, WorkerSignalData):
                #worker_signal_data = boss_signal_data.signal_data
                # print(worker_signal_data.worker_name)
                #print(worker_signal_data.signal_data)

    def handle_job_finished(self, boss_signal_data):
        pass
#            if boss_signal_data is not None and isinstance(boss_signal_data, BossSignalData):
#            if isinstance(boss_signal_data.signal_data, WorkerSignalData):
#                worker_signal_data = boss_signal_data.signal_data
#                print(worker_signal_data.worker_name)
#                print(worker_signal_data.signal_data)

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
                elif worker_signal_data.worker_name == 'pp_worker':
                    self.handle_pp_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'save_worker':
                    self.handle_save_result(worker_signal_data.signal_data)

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

    def white_balance_image(self, im, color_chip_im, style="clip"):
        """

        :param im:
        :param color_chip_im:
        :param style:
        :return:
        """
        # lum = (whiteR + whiteG + whiteB) / 3

        np_im = np.array(im)
        np_color_chip_im = np.array(color_chip_im, np.uint8)

        if style == "clip":
            try:
                whiteR, whiteG, whiteB = self.colorchipDetect.predict_color_chip_whitevals(np_color_chip_im)
                lum = (whiteR * 0.2126 + whiteG * 0.7152 + whiteB * 0.0722)
                imgR = im[..., 0].copy()
                imgG = im[..., 1].copy()
                imgB = im[..., 2].copy()

                imgR = imgR * lum / whiteR
                imgR = np.where(imgR > 255, 255, imgR)
                imgR = np.where(imgR < 0, 0, imgR)
                imgG = imgG * lum / whiteG
                imgG = np.where(imgG > 255, 255, imgG)
                imgG = np.where(imgG < 0, 0, imgG)
                imgB = imgB * lum / whiteB
                imgB = np.where(imgB > 255, 255, imgB)
                imgB = np.where(imgB < 0, 0, imgB)

                im[..., 0] = imgR
                im[..., 1] = imgG
                im[..., 2] = imgB
            except Exception as e:
                print(f"[ERROR] Error in {style} style white balancing: {e}")
        elif style == "retinex":
            # Code modified from a literal Japanese god here: https://gist.github.com/shunsukeaihara/4603234
            nimg = np_color_chip_im.transpose(2, 0, 1).astype(np.uint32)
            orig_image = np_im.transpose(2, 0, 1).astype(np.uint32)

            sum_r = np.sum(nimg[0])
            sum_r2 = np.sum(nimg[0] ** 2)
            max_r = nimg[0].max()
            max_r2 = max_r ** 2
            sum_g = np.sum(nimg[1])
            max_g = nimg[1].max()
            coefficient = np.linalg.solve(np.array([[sum_r2, sum_r], [max_r2, max_r]]),
                                          np.array([sum_g, max_g]))
            orig_image[0] = np.minimum((orig_image[0] ** 2) * coefficient[0] + orig_image[0] * coefficient[1], 255)
            sum_b = np.sum(nimg[1])
            sum_b2 = np.sum(nimg[1] ** 2)
            max_b = nimg[1].max()
            max_b2 = max_r ** 2
            coefficient = np.linalg.solve(np.array([[sum_b2, sum_b], [max_b2, max_b]]), np.array([sum_g, max_g]))
            orig_image[1] = np.minimum((orig_image[1] ** 2) * coefficient[0] + orig_image[1] * coefficient[1], 255)
            return orig_image.transpose(1, 2, 0).astype(np.uint8)
        
        elif style == 'max_white':
            color_chip_im = np_color_chip_im.transpose(2, 0, 1)
            color_chip_im = color_chip_im.astype(np.int32)
            im = np_im.transpose(2, 0, 1)
            im = im.astype(np.int32)

            im[0] = np.minimum(im[0] * (255 / float(color_chip_im[0].max())), 255)
            im[1] = np.minimum(im[1] * (255 / float(color_chip_im[1].max())), 255)
            im[2] = np.minimum(im[2] * (255 / float(color_chip_im[2].max())), 255)
            return im.transpose(1, 2, 0).astype(np.uint8)
        elif style == 'avg_white':
            #avg_white = self.colorchipDetect.predict_color_chip_whitevals(np_color_chip_im)
            avg_white = self.cc_avg_white
            brightest = np.array(avg_white).max()

            color_chip_im = np_color_chip_im.transpose(2, 0, 1)
            color_chip_im = color_chip_im.astype(np.int32)
            im = np_im.transpose(2, 0, 1)
            im = im.astype(np.int32)

            im[0] = np.minimum(im[0] * (brightest / float(color_chip_im[0].max())), 255)
            im[1] = np.minimum(im[1] * (brightest / float(color_chip_im[1].max())), 255)
            im[2] = np.minimum(im[2] * (brightest / float(color_chip_im[2].max())), 255)
            return im.transpose(1, 2, 0).astype(np.uint8)
        else:
            raise NotImplementedError("This white balancing style does not exist")

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
        self.img_path = None
        # self.metaRead = None
        self.base_file_name = None
        self.ext = None
        self.im = None
        self.bc_code = None
        self.is_blurry = None  # could be helpful for 'line item warnings'
        self.cc_quadrant = None
        self.cropped_cc = None
        self.working_save_jobs = 0
        self.processing_image = False

    def processImage(self, img_path):
        """
        given a path to an unprocessed image, performs the appropriate
        processing steps.
        """
        self.processing_image = True
        self.Timer_Emitter.timerStart.emit()
        try:
            im = self.openImageFile(img_path)
        except LibRawFatalError:
            # empty path passed
            self.processing_image = True
            return

        print(f'processing: {img_path}')
        self.img_path = img_path
        file_name, self.ext = os.path.splitext(img_path)
        self.base_file_name = os.path.basename(file_name)

        # converting to greyscale
        original_size, reduced_img = self.scale_images_with_info(im)
        self.reduced_img = reduced_img  # storing to use as preview later
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if self.mainWindow.group_renameByBarcode.isChecked():
            # retrieve the barcode values from image
            bc_worker_data = BCWorkerData(grey)
            bc_job = Job('bc_worker', bc_worker_data, self.bcRead.decodeBC)
            self.boss_thread.request_job(bc_job)

        if self.mainWindow.checkBox_blurDetection.isChecked():
            # test for bluryness
            blur_threshold = self.mainWindow.doubleSpinBox_blurThreshold.value()
            blur_worker_data = BlurWorkerData(grey, blur_threshold, True)
            blur_job = Job('blur_worker', blur_worker_data, self.blurDetect.blur_check)
            self.boss_thread.request_job(blur_job)

        if self.mainWindow.checkBox_lensCorrection.isChecked():
            # equipment corrections
            cm_distance = self.mainWindow.doubleSpinBox_focalDistance.value()
            m_distance = round(cm_distance / 100, 5)
            eq_worker_data = EQWorkerData(im, img_path, m_distance)
            eq_job = Job('eq_worker', eq_worker_data, self.eqRead.lensCorrect)
            self.boss_thread.request_job(eq_job)
            # equipment corrections should set self.im
        else:
            self.im = im
        if self.mainWindow.group_colorCheckerDetection:
            # colorchecker functions
            if self.mainWindow.radioButton_colorCheckerSmall.isChecked():
                cc_position, cropped_cc, cc_crop_time = self.colorchipDetect.process_colorchip_small(reduced_img, original_size, stride_style='ultraquick', high_precision=False, buffer_size=10)
            else:
                cc_position, cropped_cc, cc_crop_time = self.colorchipDetect.process_colorchip_big(im)
            self.cc_quadrant = self.colorchipDetect.predict_color_chip_quadrant(original_size, cc_position)
            cropped_cc = np.array(cropped_cc, np.uint8)
            self.cropped_cc = cropped_cc
            cc_avg_white = self.colorchipDetect.predict_color_chip_whitevals(cropped_cc)
            self.cc_avg_white = cc_avg_white
            print(f"CC | Position: {cc_position}, Quadrant: {self.cc_quadrant}")
            self.update_cc_info(self.cc_quadrant, cropped_cc, cc_crop_time, cc_avg_white)

        # waiting on all workers before saveing happens in Boss thread
        pp_job = Job('pp_worker', None, self.post_processing)
        self.boss_thread.request_job(pp_job)
        # process is now handed off too self.post_processing()

    def update_cc_info(self, cc_position, cropped_cc, cc_crop_time, cc_avg_white):
        """
        updates cc related diagnostic details.
        """
        height, width = cropped_cc.shape[0:2]
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(cropped_cc, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)#.rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap)
        cc_view_label = self.mainWindow.label_cc_image
        cc_view_label.setPixmap(pixmap_image)
        self.mainWindow.label_runtime.setText(str(cc_crop_time))
        # there has got to be a better way to convert this list of np.floats
        self.mainWindow.label_whitergb.setText(', '.join([str(int(x)) for x in cc_avg_white]))
        self.mainWindow.label_quad.setText(str(cc_position))

    def update_preview_img(self, im):
        # trying smaller preview
        h,w = im.shape[0:2]
        width = 300
        hpercent = (width/float(w))
        height = int((float(h)*float(hpercent)))
        size = (width, height)
        im = cv2.resize(im, size,interpolation=cv2.INTER_LINEAR)
        #height, width= im.shape[0:2]
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(im, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)#.rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap)
        preview_label = self.mainWindow.label_imPreview
        preview_label.setPixmap(pixmap_image)

    def save_finished(self):
        print(f'saving {self.img_path} has finished.')

    def post_processing(self):
        """
        combines async results and performs the post processing steps
        white balance, rotation verification then calls the saving functions
        on the final output.
        """
        im = self.im
        cropped_cc = self.cropped_cc
        # white balance params or styles will probably be a UI element.
        if self.mainWindow.checkBox_performWhiteBalance.isChecked():
            im = self.white_balance_image(im, cropped_cc, style='avg_white')

        if self.mainWindow.group_verifyRotation.isChecked():
            user_def_loc = self.mainWindow.comboBox_colorCheckerPosition.currentText()
            quad_map = ['Upper right',
                        'Lower right',
                        'Lower left',
                        'Upper left']
            user_def_quad = quad_map.index(user_def_loc) + 1
            # cc_quadrant starts at first,
            im = self.orient_image(im, self.cc_quadrant, user_def_quad)
        # pass off whatever was done to the image
        self.update_preview_img(im)
        self.im = im
        # process is now handed off too self.handle_pp_result()

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

        img_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Sample Image")
        self.queue_image(img_path)

    def openImageFile(self, imgPath,
                      demosaic=rawpy.DemosaicAlgorithm.AHD):
        """ given an image path, attempts to return a numpy array image object
        """
        #image_meta = self.metaRead.set_exif(imgPath)
        usr_gamma = self.mainWindow.doubleSpinBox_gammaValue.value()
        gamma_value = (usr_gamma, usr_gamma)
        try:  # use rawpy to convert raw to openCV
            with rawpy.imread(imgPath) as raw:
                wb = raw.camera_whitebalance
                im = raw.postprocess(chromatic_aberration=(1, 1),
                                     user_wb=wb,
                                     demosaic_algorithm=demosaic,
                                     gamma=gamma_value)

        # if it is not a raw format, just try and open it.
        except LibRawNonFatalError:
            bgr = cv2.imread(imgPath)
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except LibRawFatalError:
            if imgPath == '':
                raise
            text = 'Corrupted or incompatible image file.'
            title = 'Error opening file'
            detail_text = f'LibRawFatalError opening: {imgPath}\nUsually this indicates a corrupted input image file.'
            self.userNotice(text, title, detail_text)
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
        try:
            im = self.openImageFile(imgPath)
        except LibRawFatalError:
            return
        original_size, reduced_img = self.scale_images_with_info(im)
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        bcStatus = self.bcRead.testFeature(grey)
        if not bcStatus:
            # if first try fails, swap backend and retry
            backend = self.mainWindow.comboBox_bcBackend.currentText()
            if backend == 'zbar':
                backend = 'libdmtx'
                self.bcRead.backend = backend
            elif backend == 'libdmtx':
                backend = 'zbar'
                self.bcRead.backend = backend
                # try with alternative backend
            print(f'trying with {backend}')
            bcStatus = self.bcRead.testFeature(grey)
            if bcStatus:  # if it worked, change the comboBox
                self.populateQComboBoxSettings(self.mainWindow.comboBox_bcBackend, backend)
        try:
            blurStatus = self.blurDetect.testFeature(grey)
        except Exception as e:
            blurStatus = False
            print('blurStatus returned exception:')
            print(e)
        try:
            # If the size determiner becomes more useful we can use it to make this choice
            # for now iterate over both big and small options.
            for cc_size, ui_radio_element in {'small':self.mainWindow.radioButton_colorCheckerSmall,
                                              'big':self.mainWindow.radioButton_colorCheckerLarge}.items():
                
                ccStatus, cropped_img = self.colorchipDetect.test_feature(reduced_img, original_size, cc_size)
                # if result seems appropriate, use the image dialog box to ask
                # the user if it properly detected the color chip.
                if ccStatus:
                    mb = ImageDialog(cropped_img)
                    if mb.exec():
                        ccStatus = True
                        ui_radio_element.setChecked(True)
                        break
                    else:
                        ui_radio_element.setChecked(False)
            else:  # if the user agrees to neither, then deactivate cc functions.
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
        # each relevant feature, and the associated status
        features = {self.mainWindow.group_barcodeDetection: bcStatus,
                    self.mainWindow.group_renameByBarcode: bcStatus,
                    self.mainWindow.group_blurDetection: blurStatus,
                    self.mainWindow.group_colorCheckerDetection: ccStatus,
                    self.mainWindow.group_verifyRotation: ccStatus,
                    self.mainWindow.checkBox_performWhiteBalance: ccStatus,
                    self.mainWindow.groupBox_colorCheckerSize: ccStatus,
                    self.mainWindow.group_equipmentDetection: eqStatus,
                    self.mainWindow.checkBox_lensCorrection: eqStatus}

        for feature, status in features.items():
            feature.setEnabled(status)
            #if not status:
            feature.setChecked(status)

        # store the discovered settings
        self.saveSettings()

    def fill_patterns(self, joinedPattern):
        """
        Populates the listWidget_patterns with saved patterns.
        """
        patterns = self.retrieve_bc_patterns()
        patterns = joinedPattern.split('|')
        listWidget_patterns = self.mainWindow.listWidget_patterns
        listWidget_patterns.clear()
        for i, pattern in enumerate(patterns):
            # save the item
            listWidget_patterns.addItem(pattern)
            # now that it exists set the flag as editable
            item = listWidget_patterns.item(i)
            item.setFlags(item.flags() | Qt.ItemIsEditable)

    def add_pattern(self):
        """
        called when toolButton_addPattern is pressed
        """
        prefix = self.mainWindow.lineEdit_catalogNumberPrefix.text()
        digits = int(self.mainWindow.spinBox_catalogDigits.value())
        if digits == 0:
            collRegEx = rf'^({prefix}.*)\D*'
        else:
            collRegEx = rf'^({prefix}\d{{{digits}}})\D*'
        try:  # test compile the new pattern
            re.compile(collRegEx)
        except re.error:
            notice_title = 'Regex Pattern Error'
            notice_text = f'Warning, improper regex pattern.'
            detail_text = f'Regex patterns failed to compile.\n{collRegEx}'
            self.userNotice(notice_text, notice_title, detail_text)
            return
        listWidget_patterns = self.mainWindow.listWidget_patterns
        listWidget_patterns.addItem(collRegEx)
        item_pos = listWidget_patterns.count() - 1
        item = listWidget_patterns.item(item_pos)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.set_bc_pattern()
        

    def remove_pattern(self):
        """
        called when toolButton_removePattern is pressed
        """
        listWidget_patterns = self.mainWindow.listWidget_patterns
        selection = listWidget_patterns.currentRow()
        item = listWidget_patterns.takeItem(selection)
        item = None
        self.set_bc_pattern()
    
    def retrieve_bc_patterns(self):
        """
        harvests all pattern strings in the listWidget_patterns and returns
        them as a unique "|" joined set
        """
        listWidget_patterns = self.mainWindow.listWidget_patterns
        # is there really no way to get everything from a listWidget?
        patterns = listWidget_patterns.findItems('', Qt.MatchContains)
        patterns = "|".join(set(x.text() for x in patterns))
        return patterns

    def set_bc_pattern(self):
        """ harvests all pattern strings in the listWidget_patterns, joins
        them and sends them to self.bcRead.compileRegexPattern which in turn
        sets the bcRead.rePattern attribute."""
        patterns = self.retrieve_bc_patterns()
        try:
            self.bcRead.compileRegexPattern(patterns)
        except re.error:
            notice_title = 'Regex Pattern Error'
            notice_text = f'Warning, improper regex pattern.'
            detail_text = f'Regex patterns failed to compile.\n{patterns}'
            self.userNotice(notice_text, notice_title, detail_text)

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
        except AttributeError:
            # bcRead may not have been imported yet
            pass

        # update bc pattern preview
        prefix = self.mainWindow.lineEdit_catalogNumberPrefix.text()
        digits = int(self.mainWindow.spinBox_catalogDigits.value())
        if digits == 0:
            startingNum = '(anything)'    
        else:
            startingNum = ''.join(str(x) for x in list(range(digits)))
            startingNum = startingNum.zfill(digits)  # fill in leading zeroes
        previewText = f'{prefix}{startingNum}'  # assemble the preview string.
        #self.mainWindow.label_previewDisplay.setText(previewText) # set it
        # update dup naming preview
        self.setup_Output_Handler()
        dupPreviewEnd = self.suffix_lookup(1)
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
        msg.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
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
            return False

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

        # populate listWidget_patterns        
        self.fill_patterns(self.get('joinedPattern', ''))
       
        # QComboBox
        comboBox_colorCheckerPosition = self.get('comboBox_colorCheckerPosition', 'Upper right')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_colorCheckerPosition, comboBox_colorCheckerPosition)
        comboBox_dupNamingPolicy = self.get('comboBox_dupNamingPolicy', 'LOWER case letter')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_dupNamingPolicy, comboBox_dupNamingPolicy)
        comboBox_bcBackend = self.get('comboBox_bcBackend', 'zbar')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_bcBackend, comboBox_bcBackend)

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
        checkBox_performWhiteBalance = self.convertCheckState(self.get('checkBox_performWhiteBalance','true'))
        self.mainWindow.checkBox_performWhiteBalance.setCheckState(checkBox_performWhiteBalance)
        checkBox_lensCorrection = self.convertCheckState(self.get('checkBox_lensCorrection','true'))
        self.mainWindow.checkBox_lensCorrection.setCheckState(checkBox_lensCorrection)
        checkBox_metaDataApplication = self.convertCheckState(self.get('checkBox_metaDataApplication','true'))
        self.mainWindow.checkBox_metaDataApplication.setCheckState(checkBox_metaDataApplication)
        checkBox_blurDetection = self.convertCheckState(self.get('checkBox_blurDetection','true'))
        self.mainWindow.checkBox_blurDetection.setCheckState(checkBox_blurDetection)
        
        # QCheckBox (enable state)
        checkBox_performWhiteBalance_enabled = self.convertEnabledState(self.get('checkBox_performWhiteBalance_enabled', 'true'))
        self.mainWindow.checkBox_performWhiteBalance.setEnabled(checkBox_performWhiteBalance_enabled)
        checkBox_lensCorrection_enabled = self.convertEnabledState(self.get('checkBox_lensCorrection_enabled', 'true'))
        self.mainWindow.checkBox_lensCorrection.setEnabled(checkBox_lensCorrection_enabled)

        # QGroupbox (checkstate)
        #group_renameByBarcode = self.get('group_renameByBarcode','')
        group_renameByBarcode = self.convertCheckState(self.get('group_renameByBarcode','true'))
        self.mainWindow.group_renameByBarcode.setChecked(group_renameByBarcode)
        group_keepUnalteredRaw = self.convertCheckState(self.get('group_keepUnalteredRaw','false'))
        self.mainWindow.group_keepUnalteredRaw.setChecked(group_keepUnalteredRaw)
        group_saveProcessedJpg = self.convertCheckState(self.get('group_saveProcessedJpg','false'))
        self.mainWindow.group_saveProcessedJpg.setChecked(group_saveProcessedJpg)
        group_saveProcessedTIFF = self.convertCheckState(self.get('group_saveProcessedTIFF','false'))
        self.mainWindow.group_saveProcessedTIFF.setChecked(group_saveProcessedTIFF)
        group_saveProcessedPng = self.convertCheckState(self.get('group_saveProcessedPng','false'))
        self.mainWindow.group_saveProcessedPng.setChecked(group_saveProcessedPng)
        group_verifyRotation_checkstate = self.convertCheckState(self.get('group_verifyRotation_checkstate', 'false'))
        self.mainWindow.group_verifyRotation .setChecked(group_verifyRotation_checkstate)
        

        # QGroupbox (enablestate)
        group_renameByBarcode_enabled = self.convertEnabledState(self.get('group_renameByBarcode_enabled','true'))
        self.mainWindow.group_renameByBarcode.setEnabled(group_renameByBarcode_enabled)
        group_barcodeDetection_enabled = self.convertEnabledState(self.get('group_barcodeDetection_enabled','true'))
        self.mainWindow.group_barcodeDetection.setEnabled(group_barcodeDetection_enabled)
        group_colorCheckerDetection = self.convertEnabledState(self.get('group_colorCheckerDetection','true'))
        self.mainWindow.group_colorCheckerDetection.setEnabled(group_colorCheckerDetection)
        groupBox_colorCheckerSize = self.convertEnabledState(self.get('groupBox_colorCheckerSize','true'))
        self.mainWindow.groupBox_colorCheckerSize.setEnabled(groupBox_colorCheckerSize)
        group_verifyRotation = self.convertEnabledState(self.get('group_verifyRotation','true'))
        self.mainWindow.group_verifyRotation.setEnabled(group_verifyRotation)
        group_equipmentDetection = self.convertEnabledState(self.get('group_equipmentDetection','true'))
        self.mainWindow.group_equipmentDetection.setEnabled(group_equipmentDetection)
        # metaDataApplication should always be an option
        #group_metaDataApplication = self.convertEnabledState(self.get('group_metaDataApplication','true'))
        #self.mainWindow.group_metaDataApplication.setEnabled(group_metaDataApplication)

        # QSpinBox
        spinBox_catalogDigits = int(self.get('spinBox_catalogDigits', 0))
        self.mainWindow.spinBox_catalogDigits.setValue(spinBox_catalogDigits)

        # QDoubleSpinBox
        doubleSpinBox_focalDistance = float(self.get('doubleSpinBox_focalDistance', 25.5))
        self.mainWindow.doubleSpinBox_focalDistance.setValue(doubleSpinBox_focalDistance)
        doubleSpinBox_blurThreshold = float(self.get('doubleSpinBox_blurThreshold', 0.008))
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
        radioButton_colorCheckerSmall =  self.convertCheckState(self.get('radioButton_colorCheckerSmall', 'true'))
        self.mainWindow.radioButton_colorCheckerSmall.setChecked(radioButton_colorCheckerSmall)
        radioButton_colorCheckerLarge =  self.convertCheckState(self.get('radioButton_colorCheckerLarge', 'false'))
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

        # save the stored barcode patterns
        self.settings.setValue('joinedPattern', self.retrieve_bc_patterns())
        # be sure the bcRead backend is updated
        self.bcRead.backend = self.mainWindow.comboBox_bcBackend.currentText()

        # QComboBox
        comboBox_colorCheckerPosition = self.mainWindow.comboBox_colorCheckerPosition.currentText()
        self.settings.setValue('comboBox_colorCheckerPosition', comboBox_colorCheckerPosition)
        comboBox_dupNamingPolicy = self.mainWindow.comboBox_dupNamingPolicy.currentText()
        self.settings.setValue('comboBox_dupNamingPolicy', comboBox_dupNamingPolicy)
        comboBox_bcBackend = self.mainWindow.comboBox_bcBackend.currentText()
        self.settings.setValue('comboBox_bcBackend', comboBox_bcBackend)

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
        checkBox_performWhiteBalance = self.mainWindow.checkBox_performWhiteBalance.isChecked()
        self.settings.setValue('checkBox_performWhiteBalance', checkBox_performWhiteBalance)
        checkBox_lensCorrection = self.mainWindow.checkBox_lensCorrection.isChecked()
        self.settings.setValue('checkBox_lensCorrection', checkBox_lensCorrection)
        checkBox_metaDataApplication = self.mainWindow.checkBox_metaDataApplication.isChecked()
        self.settings.setValue('checkBox_metaDataApplication', checkBox_metaDataApplication)
        checkBox_blurDetection = self.mainWindow.checkBox_blurDetection.isChecked()
        self.settings.setValue('checkBox_blurDetection', checkBox_blurDetection)

        # QCheckBox (enable state)
        checkBox_performWhiteBalance_enabled = self.mainWindow.checkBox_performWhiteBalance.isEnabled()
        self.settings.setValue('checkBox_performWhiteBalance_enabled', checkBox_performWhiteBalance_enabled)
        checkBox_lensCorrection_enabled = self.mainWindow.checkBox_lensCorrection.isEnabled()
        self.settings.setValue('checkBox_lensCorrection_enabled', checkBox_lensCorrection_enabled)

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
        group_renameByBarcode_enabled = self.mainWindow.group_renameByBarcode.isEnabled()
        self.settings.setValue('group_renameByBarcode_enabled', group_renameByBarcode_enabled)
        group_barcodeDetection_enabled = self.mainWindow.group_barcodeDetection.isEnabled()
        self.settings.setValue('group_barcodeDetection_enabled', group_barcodeDetection_enabled)
        group_colorCheckerDetection = self.mainWindow.group_colorCheckerDetection.isEnabled()
        self.settings.setValue('group_colorCheckerDetection',group_colorCheckerDetection)
        groupBox_colorCheckerSize = self.mainWindow.groupBox_colorCheckerSize.isEnabled()
        self.settings.setValue('groupBox_colorCheckerSize',groupBox_colorCheckerSize)
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

    def reset_all_settings(self):
        """
        called by the pushButton_resetSettings, removes all settings and loads
        default settings.
        """
        text = "Reset ALL settings to default?"
        title = "Are you sure?"
        detailText = "This will revert all settings, and Image locations to default."
        user_agree = self.userAsk(text, title, detailText)
        if user_agree:
            self.settings.remove('')
            self.populateSettings()
            self.saveSettings()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = appWindow()
    # check if there are theme settings
    #if w.settings.get('value_DarkTheme', False):
    #    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    w.show()

    sys.exit(app.exec_())
