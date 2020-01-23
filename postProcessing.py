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

__author__ = "Caleb Powell, Dakila Ledesma, Jacob Motley, Jason Best"
__credits__ = ["Caleb Powell", "Dakila Ledesma", "Jacob Motley", "Jason Best",
               "Hong Qin", "Joey Shaw"]
__email__ = "calebadampowell@gmail.com"
__status__ = "Alpha"
__version__ = 'v0.0.1-alpha'

import time
from datetime import date
import os
import platform
import sys
import string
import glob
import re
from shutil import move as shutil_move
# UI libs
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog, QSizePolicy
from PyQt5.QtCore import (QSettings, Qt, QObject, QThreadPool, QEventLoop)
# image libs
import rawpy
from rawpy import LibRawNonFatalError, LibRawFatalError
import cv2
import numpy as np
# internal libs
from ui.styles import darkorange
from ui.postProcessingUI import Ui_MainWindow
from ui.noBcDialogUI import Ui_Dialog_noBc
from ui.technicianNameDialogUI import Ui_technicianNameDialog
from ui.imageDialogUI import Ui_Dialog_image
from libs.bcRead import bcRead
from libs.eqRead import eqRead
from libs.blurDetect import blurDetect
from libs.ccRead import ColorchipRead, ColorChipError, SquareFindingFailed
from libs.folderMonitor import Folder_Watcher
from libs.folderMonitor import New_Image_Emitter
from libs.boss_worker import (Boss, BCWorkerData, BlurWorkerData, EQWorkerData,
                              Job, BossSignalData, WorkerSignalData,
                              WorkerErrorData, SaveWorkerData)
from libs.metaRead import MetaRead
from libs.scaleRead import ScaleRead
from libs.settingsWizard import SettingsWizard


if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class Canvas(QtWidgets.QLabel):
    def __init__(self, im, parent=None):
        super().__init__()
        # pixmap = QtGui.QPixmap(600, 300)
        # self.setPixmap(pixmap)
        self.parent = parent
        self.setObjectName("canvas")
        self.backDrop, self.correction = self.genPixBackDrop(im)
        self.setPixmap(self.backDrop)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.end = QtCore.QPoint()
        # the box_size, is the point of the entire canvas.
        # The scale corrected larger dim of the annotated rect

    def genPixBackDrop(self, im):
        # if it is oriented in landscape, rotate it
        h, w = im.shape[0:2]
        rotated = False
        if w < h:
            im = np.rot90(im, 3)  # 3 or 1 would be equally appropriate
            h, w = w, h  # swap the variables after rotating
            rotated = True
        bytesPerLine = 3 * w
        # odd bug here, must use .copy() to avoid a mem error.
        # see: https://stackoverflow.com/questions/48639185/pyqt5-qimage-from-numpy-array
        qImg = QtGui.QImage(im.copy(), w, h, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        width = 120
        height = 120
        pixmap = pixmap.scaled(width, height,
                               QtCore.Qt.KeepAspectRatio,
                               Qt.FastTransformation)
        # corrections are doubled due to display image bieng opened at half res
        h_correction = h / height
        w_correction = w / width
        if rotated:
            correction = (w_correction, h_correction)
        else:
            correction = (h_correction, w_correction)
        return pixmap, correction

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)

        qp.drawEllipse(40, 40, 400, 400)
        qp.drawPixmap(self.rect(), self.backDrop)
        qp.setPen(QtGui.QPen(Qt.green, 1, Qt.SolidLine))
        qp.drawEllipse(self.end.x() - 3, self.end.y() - 3, 6, 6)

    def mousePressEvent(self, event):
        # dialog = self.exec()
        self.end = event.pos()

        width = self.width()
        height = self.height()
        end_point = event.pos()
        # ensure the annotations are "inside the lines"
        e_x = end_point.x()
        if e_x < 0:
            end_point.setX(0)
        elif e_x > width:
            end_point.setX(width)
        e_y = end_point.y()
        if e_y < 0:
            end_point.setY(0)
        elif e_y > height:
            end_point.setY(height)
        self.end = end_point
        self.update()
        self.updateWP()

    def updateWP(self):
        # qpoint is formatted as (xpos, ypos)
        # x col, y row
        # save the scale corrected start / end points
        x_corr, y_corr = self.correction
        e_x = self.end.x()
        e_y = self.end.y()
        # determine the end (e) points, adjusted for scale
        se_x, se_y= int(e_x * x_corr), int(e_y * y_corr)
        # update the seed_point attribute in the parent dialog
        #self.parent.seed_point = (se_y, se_x)
        self.parent.seed_point = (se_x, se_y)
        
class ImageDialog(QDialog):
    def __init__(self, img_array_object):
        super().__init__()
        self.init_ui(img_array_object)
        # set an initial seed_point which fails an if check
        self.seed_point = None

    def init_ui(self, img_array_object):
        mb = Ui_Dialog_image()
        mb.setupUi(self)
        _translate = QtCore.QCoreApplication.translate
        mb.label_dialog.setText(_translate("White Point", "Failed to determine the white point from the CRC. Please CLICK THE WHITE POINT."))
        canv = Canvas(im=img_array_object, parent=self)
        mb.gridLayout.addWidget(canv)

    def ask_user_for_seed(self):
        dialog = self.exec()
        if dialog:
            result = self.seed_point
            print(result)
            return result
        else:
            return None


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


class TechnicianNameDialog(QDialog):
    """
    a UI element to display & and edit the list of stored technician names.
    """
    def __init__(self, nameList=[]):
        super().__init__()
        self.init_ui(nameList)

    def init_ui(self, nameList):
        self.dialog = Ui_technicianNameDialog()
        self.dialog.setupUi(self)
        self.dialog.lineEdit_newTechnician.setFocus(True)
        # populate the listwidget with nameList passed on creation
        listWidget_technicianNames = self.dialog.listWidget_technicianNames
        for i, name in enumerate(nameList):
            # save the item
            listWidget_technicianNames.addItem(name)
            # now that it exists set the flag as editable
            item = listWidget_technicianNames.item(i)
            item.setFlags(item.flags() | Qt.ItemIsEditable)

    def edit_technician_list(self):
        dialog = self.exec()
        if dialog:
            result = self.retrieve_technician_names()
        else:
            result = None
        return result

    def retrieve_technician_names(self):
        """
        harvests all unique technician names
        """
        listWidget_technicianNames = self.dialog.listWidget_technicianNames
        # Is there no better way to get everything from a listWidget?
        names = listWidget_technicianNames.findItems('', Qt.MatchContains)
        # alphabetize unique values and return a list
        names = list(set(x.text() for x in names))
        if '' not in names:
            names = [''] + names
        return names

    def add_item(self):
        """ connected to pushButton_add """
        listWidget_technicianNames = self.dialog.listWidget_technicianNames
        lineEdit_newTechnician = self.dialog.lineEdit_newTechnician
        newName = lineEdit_newTechnician.text()
        if len(newName) > 0:  # if something is written add it to list
            listWidget_technicianNames.addItem(newName)
            lineEdit_newTechnician.clear()
        else:  # otherwise, drop a hint: focus on lineEdit_newTechnician
            self.dialog.lineEdit_newTechnician.setFocus(True)

    def remove_item(self):
        """ connected to pushButton_remove """
        listWidget_technicianNames = self.dialog.listWidget_technicianNames
        selection = listWidget_technicianNames.currentRow()
        listWidget_technicianNames.takeItem(selection)
        #item = None


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
        print(f"Multithreading with maximum "
              f"{self.threadPool.maxThreadCount()} threads")
        # initiate the persistant settings
        # todo update this when name is decided on
        self.settings = QSettings('HerbASAP', 'HerbASAP')
        self.setWindowIcon(QtGui.QIcon('docs/icon_a.png'))
        self.settings.setFallbacksEnabled(False)  # File only, no fallback to registry.
        # populate the settings based on the previous preferences
        self.populateSettings() # this also loads in the settings profile
        # restore window geometry & state.
        saved_win_geom = self.get("geometry", "")
        if saved_win_geom != '':
            self.restoreGeometry(saved_win_geom)
        saved_win_state = self.get("windowState", "")
        if saved_win_state != '':
            self.restoreGeometry(saved_win_state)
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
       # set up classes which do not need re init upon profile loading
        self.blurDetect = blurDetect(parent=self.mainWindow)
        self.colorchipDetect = ColorchipRead(parent=self.mainWindow)
        # Establish base working condition
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
        self.boss_thread.start(priority=QtCore.QThread.TimeCriticalPriority)

        # setup static UI buttons
        self.mainWindow.toolButton_delPreviousImage.pressed.connect(self.delete_previous_image)
        self.mainWindow.toolButton_editTechnicians.pressed.connect(self.edit_technician_list)
        self.versionCheck()

    def versionCheck(self):
        """ checks the github repo's latest release version number against
        local and offers the user to visit the new release page if different"""
        #  be sure to only do this once a day.

        today = str(date.today())
        lastChecked = self.get('date_versionCheck', today)
        # save the new version check date
        self.settings.setValue("date_versionCheck", today)
        self.saveSettings()

        if today != lastChecked:
            import requests
            from requests.exceptions import ConnectionError
            import webbrowser
            apiURL = 'https://api.github.com/repos/CapPow/HerbASAP/releases/latest'
            try:
                apiCall = requests.get(apiURL)
                status = str(apiCall)
            except ConnectionError:
                #  if no internet, don't bother the user.
                return
            result = apiCall.json()
            if '200' in status:  # if the return looks bad, don't bother user
                url = result['html_url']
                version = result['tag_name']
                if version.lower() != __version__.lower():
                    message = f'A new version ( {version} ) of HerbASAP has been released. Would you like to visit the release page?'
                    title = 'HerbASAP Version'
                    answer = self.userAsk(message, title, inclHalt = False)
                    if answer:# == QMessageBox.Yes:
                        link=url
                        self.showMinimized() #  hide the app about to pop up.
                        #  instead display a the new release
                        webbrowser.open(link,autoraise=1)

    def closeEvent(self, event):
        """
        Reimplimentation of closeEvent handling. This will be run when the user
        closes the program.
        """
        # check if any critical processes are running
        len_queue = len(self.image_queue)
        qty_save_jobs = self.working_save_jobs

        criticalConditions = [len_queue  > 0,
                              qty_save_jobs > 0,
                              self.processing_image]
        if any(criticalConditions):
            # if anything important is running
            if len_queue < 2:
                # if the queue is short just wait for it to finish
                while len_queue > 0:
                    # could add a status bar update here if we use a status bar
                    len_queue = len(self.image_queue)  # the update variable 
                    time.sleep(1)  # a long wait time avoids slowing processing
            else:
                    # if the queue is longer then ask about force quitting.
                text = 'Currently Processing Images! Interrupt the running tasks?'
                title = "Halt Processing?"
                detailText = (
            'Closing will interrupt image processing for all queued tasks!\n'
            f'Items queued for processing: {len_queue}\n'
            f'Qty save jobs running: {qty_save_jobs}\n'
            )
                user_agree = self.userAsk(text, title, detailText)
                if not user_agree:
                    # user does not want to exit.
                    return
        # if above conditions did not prematurely return then start shutdown.
        if self.folder_watcher.is_monitoring:
            # if the folder watcher is running, go ahead and stop it.
            self.toggle_folder_monitoring()
        # save the current window state & location        
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        # save the current settings
        self.saveSettings()
        QMainWindow.closeEvent(self, event)

    def start_timer(self):
        self.timer_start = time.time()

    def stop_timer(self):
        end_time = time.time()
        run_time = round(end_time - self.timer_start, 3)
        self.mainWindow.label_processtime.setText(str(run_time))
        print(f'Elapsed runtime: {run_time}')

        # give app a moment to update
        app.processEvents()

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

        lineEdit_inputPath = self.profile.get('inputPath', '')
        self.folder_watcher = Folder_Watcher(lineEdit_inputPath,
                                             raw_image_patterns)

        self.folder_watcher.emitter.new_image_signal.connect(self.queue_image)

        self.mainWindow.pushButton_toggleMonitoring.clicked.connect(
                self.toggle_folder_monitoring)

    def toggle_folder_monitoring(self):
        pushButton = self.mainWindow.pushButton_toggleMonitoring
        if self.folder_watcher.is_monitoring:
            pushButton.setText('Begin folder monitoring')
            self.folder_watcher.stop()
        else:  # if somehow no input path, end early.
            if self.profile.get('inputPath', '') == '':
                return
            pushButton.setText(' Stop folder monitoring')
            self.folder_watcher.run()
            self.update_session_stats()

    def setup_Output_Handler(self):
        """
        initiates self.save_output_handler with user inputs.
        """
        # each key is a file extension
        # each value is a tuple containing (bool if checked, dst path)
        self.output_map = {
                '.jpg': (self.profile.get('saveProcessedJpg', False), self.profile.get('pathProcessedJpg', '')),
                '.raw': (self.profile.get('keepUnalteredRaw', False), self.profile.get('pathUnalteredRaw', ''))}
        dupNamingPolicy = self.profile.get('dupNamingPolicy')

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


    def update_metadata(self):
        
        exif_dict = {
                # the program's name and verison number
                'software': f'HerbASAP, {__version__} ({platform.system()})',
                'settingProfile': self.mainWindow.comboBox_profiles.currentText(),
                # settings metadata
                'collectionName': self.profile.get('collectionName', ''),
                'collectionURL': self.profile.get('collectionURL', ''),
                'contactEmail': self.profile.get('contactEmail', ''),
                'contactName': self.profile.get('contactName', ''),
                'copywriteLicense': self.profile.get('copywriteLicense', ''),
                # session metadata
                'scientificName': self.mainWindow.lineEdit_taxonName.text(),
                'recordedBy': self.mainWindow.lineEdit_collectorName.text(),
                'imagedBy': self.mainWindow.comboBox_technician.currentText()
                }

        self.metaRead.update_static_exif(exif_dict)

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
        # disable the toolButton_delPreviousImage button until this is done
        self.mainWindow.toolButton_delPreviousImage.setEnabled(False)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # reset recently_produced_images
        self.recently_produced_images = [orig_img_path]
        # see setup_Output_Handler() for details concerning self.output_map
        output_map = self.output_map
        # breakout output_map into a list of tuples containing (ext, path)
        to_save = [(x, y[1]) for x, y in output_map.items() if y[0]]
        # store how many save_jobs are expected to global variable
        self.working_save_jobs = len(im_base_names) * len(to_save)

        # retrieve the source image exif data
        # add Additional user comment details for the metadata
        addtl_user_comments = {
                'avgWhiteRGB': str(self.cc_avg_white),
                'barcodeValues': self.bc_code,
                'isBlurry': str(self.is_blurry),
                'ccQuadrant': str(self.cc_quadrant),
                'ccLocation': str(self.cc_location),
                'pixelsPerMM': str(self.ppmm),
                'pixelsPerMMConfidence': str(self.ppmm_uncertainty)
                }

        self.meta_data = self.metaRead.retrieve_src_exif(orig_img_path,
                                                         addtl_user_comments)
        if self.working_save_jobs < 1:  # condition when no outputs saved
            #  treat this process as if it was passed to boss_worker
            self.handle_save_result(orig_img_path)
        else:
            # for each bc_value or file name passed in...
            for bc in im_base_names:
                for ext, path in to_save:
                    # not sure of slick way to avoid checking ext twice
                    if ext == '.raw':
                        ext = orig_im_ext
                    file_list = glob.glob(f'{path}//{bc}*{ext}')

                    file_quantity = len(file_list)
                    file_suffixes = set()
                    if file_quantity > 0:
                        for filename in file_list:
                            suffix = os.path.basename(filename).replace(bc, '').replace(ext, '')
                            file_suffixes.add(suffix)

                    duplicate_naming_policy = self.profile.get('dupNamingPolicy')
                    new_file_suffix = ''
                    if duplicate_naming_policy == 'LOWER case letter':
                        letters = list(string.ascii_lowercase)
                        while new_file_suffix in file_suffixes:
                            new_file_suffix += 'a'
                            for letter in letters:
                                new_file_suffix = new_file_suffix[:-1]
                                new_file_suffix += letter
                                if new_file_suffix not in file_suffixes:
                                    break
                    elif duplicate_naming_policy == 'UPPER case letter':
                        letters = list(string.ascii_uppercase)
                        while new_file_suffix in file_suffixes:
                            new_file_suffix += 'A'
                            for letter in letters:
                                new_file_suffix = new_file_suffix[:-1]
                                new_file_suffix += letter
                                if new_file_suffix not in file_suffixes:
                                    break
                    elif duplicate_naming_policy == 'Number with underscore':
                        num = 0
                        while new_file_suffix in file_suffixes:
                            if new_file_suffix not in file_suffixes:
                                break
                            new_file_suffix = f"_{num}"
                            num += 1

                    # print(f"File suffixes: {file_suffixes}")
                    # print(f"New file suffix: {new_file_suffix}")

                    new_file_base_name = f'{bc}{new_file_suffix}'
                    new_file_name = f'{path}//{new_file_base_name}{ext}'
                    if ext == orig_im_ext:
                        # if it is a raw move just perform it
                        try:
                            shutil_move(orig_img_path, new_file_name)
                            #  treat this process as if it was passed to boss_worker
                            self.handle_save_result(new_file_name)
                        except FileNotFoundError:
                            # condition when multiple bcs try to move > once.
                            #  treat this process as if it was passed to boss_worker
                            self.handle_save_result(False)
                    else:
                        # if it a cv2 save function pass it to boss_worker
                        save_worker_data = SaveWorkerData(new_file_name, im)
                        save_job = Job('save_worker',
                                       save_worker_data, self._cv2_save)
                        self.boss_thread.request_job(save_job)
        # re-enable the toolButton_delPreviousImage
        self.mainWindow.toolButton_delPreviousImage.setEnabled(True)

    def _cv2_save(self, new_file_name, im):
        """
        cv2.imwrite helper, returns the path if it succeeds, otherwise false.
        Exists so that self.handle_save_result recieves filepath if save_job in
        save_output_images is properly executed. This is necessary to build the
        "self.recently_produced_images" list used in self.delete_previous_image
        This could be cleaner, but it'll take a lot of reworking boss_worker.
        """
        if cv2.imwrite(new_file_name, im):
            return new_file_name
        else:
            return False

    def delete_previous_image(self):
        """
        called by pushButton_delPreviousImage, will remove the raw input image
        and ALL derived images.
        """
        text = ("PERMANENTLY DELETE the most recently captured image AND any "
                "images derived from it?")
        title = "DELETE Last Image?"
        # generate a textual, newline seperated list of items to be removed.
        fileList = '\n' + '\n'.join(self.recently_produced_images)
        detailText = f'This will permanently the following files:{fileList}'
        user_agree = self.userAsk(text, title, detailText)
        if user_agree:
            self.reset_preview_details()
            for imgPath in self.recently_produced_images:
                if os.path.isfile(imgPath):  # does it exist?
                    os.remove(imgPath)  #  if so, remove the file
            # disable toolButton_delPreviousImage
            self.mainWindow.toolButton_delPreviousImage.setEnabled(False)
            self.recently_produced_images = []
            if self.folder_watcher.is_monitoring:
                # if the folder_watcher is on, subtract removed image from the count
                self.folder_watcher.img_count -= 1
                self.update_session_stats()

    def reset_diagnostic_details(self):
        """
        resets the diagnostic texts.
        """
        self.mainWindow.label_processtime.setText('')
        self.mainWindow.label_barcodes.setText('')
        self.mainWindow.label_runtime.setText('')
        self.mainWindow.label_whitergb.setText('')
        self.mainWindow.label_isBlurry.setText('')
        self.mainWindow.label_lapnorm.setText('')

    def reset_preview_details(self):
        """
        resets the image specific GUI elements.
        """
        self.mainWindow.label_imPreview.clear()
        self.mainWindow.label_cc_image.clear()
        self.reset_diagnostic_details()

    def handle_blur_result(self, result):
        is_blurry = result['isblurry']
        if is_blurry:
            # update preview for the coming notice.
            if self.im is None:
                self.update_preview_img(self.reduced_img)
            else:
                self.update_preview_img(self.im)
            notice_title = 'Blurry Image Warning'
            if self.bc_code:
                notice_text = f'Warning, {self.bc_code} is blurry.'
            else:
                notice_text = f'Warning, {self.img_path} is blurry.'
            notice_text = notice_text + ('\n\nConsider deleting the image '
                                         '(using the trash can icon), '
                                         'adjusting focus, and retaking the '
                                         'image.')
            #  lapNorm = laplacian / imVar
            detail_text = (
                    "Normalized laplacian is the variance of openCV's "
                    "Laplacian operator divided by the variance of the "
                    "image. Higher values are less blurry. A threshold can be "
                    "set in the settings tab to determine when this dialog "
                    "should appear."
                    f"\n\nlaplacian={result['laplacian']}"
                    f"\nImg variance ={result['imVar']}\nnormalized "
                    f"laplacian={result['lapNorm']}")
            self.userNotice(notice_text, notice_title, detail_text)
        self.is_blurry = is_blurry
        self.mainWindow.label_isBlurry.setText(str(is_blurry))
        self.mainWindow.label_lapnorm.setText(str(round(result['lapNorm'], 3)))

    def handle_save_result(self, result):
        """ called when the the save_worker finishes up """
        # if result is a path to a new file...
        if result:
            # for now can only save meta data on jpgs
            if result.lower().endswith('.jpg'):
                # Add that path to the class variable storing the most recent products.
                self.recently_produced_images.append(result)
                # Apply the metadata to that saved object            
                self.metaRead.set_dst_exif(self.meta_data,
                                           result)
        # tick off one working_save_job
        self.working_save_jobs -= 1
        # if we're down to 0 then wrap up
        if self.working_save_jobs < 1:
            # inform the app when image processing is complete
            self.processing_image = False
            # reset the stored meta_data 
            self.meta_data = None
            # these are happening too soon
            self.Image_Complete_Emitter.completed.emit()
            self.Timer_Emitter.timerStop.emit()

    def alert_blur_finished(self):
        """ called when the results are in from blur detection. """
 
    def handle_bc_result(self, result):
        if not result:
            # update preview for the coming dialog.
            if self.im is None:
                self.update_preview_img(self.reduced_img)
            else:
                self.update_preview_img(self.im)
            userDialog = BcDialog()
            result = [userDialog.ask_for_bc()]
        if result in [[None], ['']]:
            result = [self.base_file_name]
        self.bc_code = result
        self.mainWindow.label_barcodes.setText(', '.join(result))

    def alert_bc_finished(self):
        """ called when the results are in from bcRead."""

    def handle_eq_result(self, result):
        # this is returning the corrected image array
        self.im = result

    def alert_eq_finished(self):
        """ called when the results are in from eqRead."""

    # boss signal handlers
    def handle_boss_started(self, boss_signal_data):
        pass


    def handle_boss_finished(self, boss_signal_data):
        pass

    def handle_job_started(self, boss_signal_data):
        pass

    def handle_job_finished(self, boss_signal_data):
        pass


    def handle_job_result(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data,
                                                       BossSignalData):

            if isinstance(boss_signal_data.signal_data, WorkerSignalData):
                worker_signal_data = boss_signal_data.signal_data
                if worker_signal_data.worker_name == 'bc_worker':
                    self.handle_bc_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'blur_worker':
                    self.handle_blur_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'eq_worker':
                    self.handle_eq_result(worker_signal_data.signal_data)
                elif worker_signal_data.worker_name == 'save_worker':
                    self.handle_save_result(worker_signal_data.signal_data)

    def handle_job_error(self, boss_signal_data):
        if boss_signal_data is not None and isinstance(boss_signal_data,
                                                       BossSignalData):

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
            reduced_im = cv2.resize(im,
                                    (largest_dim,
                                     round((largest_dim / image_width) * image_height)),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            reduced_im = cv2.resize(im,
                                    (round((largest_dim / image_height) * image_width),
                                     largest_dim),
                                    interpolation=cv2.INTER_NEAREST)
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
        self.base_file_name = None
        self.flip_value = 0
        self.ext = None
        self.im = None
        self.cc_avg_white = None
        self.bc_code = []
        self.is_blurry = None  # could be helpful for 'line item warnings'
        self.cc_quadrant = None
        self.cc_location = None
        self.cropped_cc = None
        self.working_save_jobs = 0
        self.processing_image = False
        self.ppmm = 'N/A'
        self.ppmm_uncertainty = 'N/A'

        try:
            if self.raw_base:  # try extra hard to free up these resources.
                print('manually forcing closure')
                self.raw_base.close()
                # reset the image preview text
                self.mainWindow.label_imPreview.setText('')
                # if some failure occurred attempt to restart the queue.
        except AttributeError:
            pass  # occasion where no raw images have been unpacked yet
        self.raw_base = None

    def processImage(self, img_path):
        """
        given a path to an unprocessed image, performs the appropriate
        processing steps.
        """
        self.processing_image = True
        # reset the diagnostic details
        self.reset_preview_details()
        # retrieve the profile as a local variable
        profile = self.profile
        
        #self.reset_diagnostic_details()
        self.mainWindow.label_imPreview.setText('...working')
        # give app a moment to update
        app.processEvents()

        self.Timer_Emitter.timerStart.emit()

        self.img_path = img_path
        file_name, self.ext = os.path.splitext(img_path)
        self.base_file_name = os.path.basename(file_name)        
        try:
            im = self.openImageFile(img_path)
        except (LibRawFatalError, LibRawNonFatalError) as e:
            # prepare to wipe the slate clean and exit (this function)
            self.reset_working_variables()
            # attempt to recover from the error
            self.process_from_queue()
            # if the folder_watcher is operating, update session stats
            if self.folder_watcher.is_monitoring:
                self.folder_watcher.img_count += 1
                self.update_session_stats()
            return
        print(f'processing: {img_path}')
        # converting to greyscale
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if profile.get('renameByBarcode', True):
            # retrieve the barcode values from image
            bc_worker_data = BCWorkerData(grey)
            bc_job = Job('bc_worker', bc_worker_data, self.bcRead.decodeBC)
            self.boss_thread.request_job(bc_job)
        if profile.get('blurDetection', True):
            # test for bluryness
            blur_threshold = profile.get('blurThreshold', 0.045)
            blur_worker_data = BlurWorkerData(grey, blur_threshold, True)
            blur_job = Job('blur_worker',
                           blur_worker_data, self.blurDetect.blur_check)
            self.boss_thread.request_job(blur_job)
        # reduce the image for the cnn, store it incase of problem dialogs
        original_size, reduced_img = self.scale_images_with_info(im)
        self.reduced_img = reduced_img
        # currently this is always returning True as it does not exist in self.profile
        
        scaleDetermination = profile.get('scaleDetermination', False)
        verifyRotation = profile.get('verifyRotation', False)
        performWhiteBalance = profile.get('performWhiteBalance', False)
        
        if any([scaleDetermination, verifyRotation, performWhiteBalance]):
            # colorchecker functions

            try:
                crc_type = profile.get('crcType', "ISA ColorGauge Nano")
                if crc_type == "ISA ColorGauge Nano":  # aka small crc
                    partition_size = profile.get('partition_size', 125)
                    cc_location, cropped_cc, cc_crop_time = \
                        self.colorchipDetect.process_colorchip_small(reduced_img,
                                                                     original_size,
                                                                     high_precision=True,
                                                                     partition_size=partition_size)
                elif crc_type == 'Tiffen / Kodak Q-13  (8")':
                    cc_location, cropped_cc, cc_crop_time = self.colorchipDetect.process_colorchip_big(im, pp_fix=1)
                else:
                    cc_location, cropped_cc, cc_crop_time = self.colorchipDetect.process_colorchip_big(im)
                x1, y1, x2, y2 = cc_location
                if scaleDetermination:
                    # scale determination code
                    full_res_cc = im[y1:y2, x1:x2]
                    # lookup the patch area and seed function
                    patch_mm_area, seed_func, to_crop = self.scaleRead.scale_params.get(crc_type)
                    self.ppmm, self.ppmm_uncertainty = self.scaleRead.find_scale(full_res_cc,
                                                                                 patch_mm_area,
                                                                                 seed_func,
                                                                                 to_crop)
                    print(f"Pixels per mm for {os.path.basename(img_path)}: {self.ppmm}, +/- {self.ppmm_uncertainty}")

                self.cc_quadrant = self.colorchipDetect.predict_color_chip_quadrant(original_size, cc_location)
                self.cropped_cc = cropped_cc
                if performWhiteBalance:
                    try:
                        self.cc_avg_white, self.cc_blk_point = self.colorchipDetect.predict_color_chip_whitevals(cropped_cc, crc_type)
                    # if colorchipDetect fails it will raise a SquareFindingFailed error
                    except SquareFindingFailed:
                        # catch it and call the imageDialog
                        seedDialog = ImageDialog(self.cropped_cc)
                        seed_pt = seedDialog.ask_user_for_seed()
                        if seed_pt:
                            self.cc_avg_white, self.cc_blk_point = self.colorchipDetect.predict_color_chip_whitevals(cropped_cc, crc_type, seed_pt=seed_pt)
                        else:
                            raise ColorChipError
                            
                if verifyRotation:
                    user_def_loc = profile.get('colorCheckerPosition', 'Upper right')
                    quad_map = ['Upper right',
                                'Upper left',
                                'Lower left',
                                'Lower right']
                    user_def_quad = quad_map.index(user_def_loc) + 1
                    # cc_quadrant starts at first,
                    # determine the proper rawpy flip value necessary
                    rotation_qty = (self.cc_quadrant - user_def_quad)
                    # rawpy: [0-7] Flip image (0=none, 3=180, 5=90CCW, 6=90CW)
                    # create a list to travel based on difference
                    rotations = [6, 3, 5, 0, 6, 3, 5]
                    startPos = 3  # starting position in the list
                    endPos = rotation_qty + startPos  # ending index in the list
                    self.flip_value = rotations[endPos]  # value at that position

                self.apply_corrections()

                width, height = original_size
                if self.flip_value == 3:
                    x1, x2, y1, y2 = width - x2, width - x1, height - y2, height - y1

                elif self.flip_value == 5:
                    x1, y1, x2, y2 = y1, width - x2, y2, width - x1

                elif self.flip_value == 6:
                    x1, x2, y1, y2 = height - y2, height - y1, x1, x2

                cc_location = x1, y1, x2, y2
                self.cc_location = cc_location

                # print(f"CC Position after calc.: {cc_location}")

                self.update_cc_info(self.cc_quadrant, cropped_cc,
                                    cc_crop_time, self.cc_avg_white)
            # apply corrections based on what is learned from the colorchipDetect
            except ColorChipError as e: 
                notice_title = 'Error Determining Color Chip Location'
                notice_text = 'Critical Error: Image was NOT processed!'
                detail_text = ('While attempting to determine the color chip '
                               'location the following exception was rasied:'
                               f'\n{e}')
                self.userNotice(notice_text, notice_title, detail_text)
                # prepare to wipe the slate clean and exit
                self.reset_working_variables()
                # attempt to recover from the error
                self.process_from_queue()
                return

        # optional manually implimented white balance
        #self.high_precision_wb(cc_location)

        # pass off what was learned and properly open image.
        # add the (mostly) corrected image to the preview
        # equipment corrections remain. let user look at this while that runs.

        self.update_preview_img(self.im)
        if profile.get('lensCorrection', True):
            # equipment corrections
            eq_worker_data = EQWorkerData(self.im)
            eq_job = Job('eq_worker', eq_worker_data, self.eqRead.lensCorrect)
            self.boss_thread.request_job(eq_job)
            # equipment corrections should set self.im
        wait_event = QEventLoop()
        # if there are any worker threads working then wait on them to finish
        if any([

                self.boss_thread._Boss__is_bc_worker_running,
                self.boss_thread._Boss__is_blur_worker_running,
                self.boss_thread._Boss__is_eq_worker_running]):

            self.boss_thread.signals.clear_to_save.connect(wait_event.quit)
            wait_event.exec()
        # now that it appears all workers are wrapped up, bundle the save ops.
        if len(self.bc_code) > 0:
            names = self.bc_code
        else:  # name based on base_file_name
            names = [self.base_file_name]
        self.save_output_images(self.im, names, self.img_path, self.ext)
        # if the folder_watcher is operating, update session stats
        if self.folder_watcher.is_monitoring:
            self.folder_watcher.img_count += 1
            self.update_session_stats()

    def update_cc_info(self, cc_location, cropped_cc, cc_crop_time,
                       cc_avg_white):
        """
        updates cc related diagnostic details.
        """
        cc_view_label = self.mainWindow.label_cc_image
        # if the crop is vertically oriented... rotate it
        h, w = cropped_cc.shape[0:2]
        if h > w:
            cropped_cc = np.rot90(cropped_cc, 1)
            h, w = w, h  # swamp the variables after rotating
        bytesPerLine = 3 * w
        # odd bug here, must use .copy() to avoid a mem error.
        # see: https://stackoverflow.com/questions/48639185/pyqt5-qimage-from-numpy-array
        qImg = QtGui.QImage(cropped_cc.copy(), w, h, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        width = cc_view_label.width()
        height = cc_view_label.height()
        pixmap = pixmap.scaled(width, height,
                               QtCore.Qt.KeepAspectRatio,
                               Qt.FastTransformation)
        cc_view_label.setPixmap(pixmap)

        self.mainWindow.label_runtime.setText(str(cc_crop_time))
        # there has got to be a better way to convert this list of np.floats
        self.mainWindow.label_whitergb.setText(
                ', '.join([str(int(x)) for x in cc_avg_white]))
        # give app a moment to update
        app.processEvents()

    def update_preview_img(self, im):
        h, w = im.shape[0:2]
        preview_label = self.mainWindow.label_imPreview
        bytesPerLine = 3 * w
        qImg = QtGui.QImage(im, w, h, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        width = preview_label.width()
        height = preview_label.height()
        pixmap = pixmap.scaled(width, height,
                               QtCore.Qt.KeepAspectRatio,
                               Qt.FastTransformation)
        preview_label.setPixmap(pixmap)
        # give app a moment to update
        app.processEvents()

    def edit_technician_list(self):
        """
        called by toolButton_editTechnicians asks the user to edit/save a list
        of technician names. Then saves the result to qSetting 'technicianList'
        """
        # be sure an empty string always preceeds the list.
        nameList = self.get('technicianList', [''])
        if '' not in nameList:
            nameList = [''] + nameList
        technicianDialog = TechnicianNameDialog(nameList)
        result = technicianDialog.edit_technician_list()
        if result:
            self.settings.setValue('technicianList', list(set(result)))
            # update the comboBox_technician options
            self.populate_technician_list()

    def populate_technician_list(self):
        # identify the target comboBox
        tech_comboBox = self.mainWindow.comboBox_technician
        # clear current items in the list
        tech_comboBox.clear() 
        # populate the list
        nameList = self.get('technicianList', [])
        tech_comboBox.addItems(nameList)
        # reset setting to ''
        tech_comboBox.setCurrentText('')

    def update_session_stats(self):
        """
        updates the folder monitor session stats in gridGroupBox_sessionRates
        """
        run_time = self.folder_watcher.get_runtime()
        self.mainWindow.label_runTime.setText(str(run_time))
        img_count = self.folder_watcher.img_count
        if img_count == 0:
            self.mainWindow.label_speedAnimal.clear()
            self.mainWindow.label_imgCount.setText('0')
            self.mainWindow.label_rate.setText('0.00')
        else:
            self.mainWindow.label_imgCount.setText(str(img_count))
            rate = round(img_count / run_time, 3)
            self.mainWindow.label_rate.setText(str(rate))
            if img_count % 12 == 0:  # every n images, refresh this
                animalScore = min(4, int(rate))
                animalList = [":/icon/turtle.png",
                              ":/icon/human.png",
                              ":/icon/rabbit.png",
                              ":/icon/rabbit2.png",
                              ":/icon/rabbit3.png"]
                animal = QtGui.QPixmap(animalList[animalScore])
                an_label = self.mainWindow.label_speedAnimal
                an_label.setPixmap(animal.scaled(an_label.size(),
                                                 Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation))

        # give app a moment to update
        app.processEvents()

    def save_finished(self):
        # I don't believe this ever triggers
        print(f'saving {self.img_path} has finished.')

    def process_selected_items(self, selection, selection_type):
        """
        Process selected files. Called by either process_image_directory or
        process_image_selection.
        """
        if selection_type == 'directory':
            raw_image_patterns = ['*.cr2', '*.CR2',
                                  '*.tiff', '*.TIFF',
                                  '*.nef', '*.NEF',
                                  '*.orf', '*.ORF']
            to_process = []
            for extension in raw_image_patterns:
                files_to_add = glob.glob(f'{selection}//*{extension}')
                # clear out empies
                files_to_add = [x for x in files_to_add if x != []]
                to_process.extend(files_to_add)
        else:  # otherwise it must be a list of objects in which, attempt all
            to_process = selection
        # queue up each item
        qty_to_process = len(to_process)
        if qty_to_process > 5:
            text = f'Process ALL {qty_to_process} images?'
            title = f'Process {qty_to_process} Images?'
            detail_text = 'Items to be processed:' + '\n\n'.join(to_process)
            user_agree = self.userAsk(text, title, detail_text)
            if not user_agree:  # if user 'nopes out' exit prematurely
                return
        # if above does not exit prematurely, process each item
        for img_path in to_process:
            self.queue_image(img_path)

    def process_image_directory(self):
        """
        Called by pushButton_processMulti to process a single folder.
        """
        # If folder monitoring is on turn it off
        if self.folder_watcher.is_monitoring:
            self.toggle_folder_monitoring()
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Directory")
        # pass off the results to be processed
        self.process_selected_items(dir_path, 'directory')

    def process_image_selection(self):
        """
        Called by pushButton_processSingle to process a selection of
        images.
        """
        # If folder monitoring is on turn it off
        if self.folder_watcher.is_monitoring:
            self.toggle_folder_monitoring()
        img_path, _ = QtWidgets.QFileDialog.getOpenFileNames(self,
                                                    "Open Unprocessed Image")
        # pass off the results to be processed
        self.process_selected_items(img_path, 'selection')

    def high_precision_wb(self, cc_location):
        x1, y1, x2, y2 = cc_location
        cc_im = self.im[y1:y2, x1:x2]

        grayImg = cv2.cvtColor(cc_im, cv2.COLOR_RGB2GRAY)  # convert to gray
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
        # determine an allowable range for the floodfill
        var_threshold = int((maxVal - minVal) * .1)
        h, w, chn = cc_im.shape
        seed = maxLoc
        mask = np.zeros((h + 2, w + 2), np.uint8)
        floodflags = 8
        floodflags |= cv2.FLOODFILL_FIXED_RANGE
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        floodflags |= (int(maxVal) << 8)
        num, cropped_cc, mask, rect = cv2.floodFill(cc_im, mask, seed,
                                                    0,
                                                    (var_threshold,) * 3,
                                                    (var_threshold,) * 3,
                                                    floodflags)
        # correct for the mask expansion
        mask = mask[1:-1, 1:-1, ...]
        squares = ColorchipRead.find_squares(mask, 100, 10000)
        try:
            biggest_square = max(squares, key=cv2.contourArea)
            x_arr = biggest_square[..., 0]
            y_arr = biggest_square[..., 1]
            x1, y1, x2, y2 = np.min(x_arr), np.min(y_arr), np.max(x_arr), np.max(y_arr)
            # biggest_square = (x1, y1, x2, y2)
            cc_im = cc_im[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
            #cc_im1 = cv2.cvtColor(cc_im, cv2.COLOR_RGB2BGR)
        except:
            return

        color_chip_im = cc_im.transpose(2, 0, 1)
        color_chip_im = color_chip_im.astype(np.int32)
        im = self.im.transpose(2, 0, 1)
        im = im.astype(np.int32)

        im[0] = np.minimum(im[0] * (255 / float(color_chip_im[0].max()) - 0.2), 255)
        im[1] = np.minimum(im[1] * (255 / float(color_chip_im[1].max()) - 0.2), 255)
        im[2] = np.minimum(im[2] * (255 / float(color_chip_im[2].max()) - 0.2), 255)
        self.im = im.transpose(1, 2, 0).astype(np.uint8)

    def openImageFile(self, imgPath,
                      demosaic=rawpy.DemosaicAlgorithm.AHD):
        """
        given an image path, attempts to return a numpy array image object
        """
        # first open an unadulterated reference version of the image
        ext_wb = [1, 0.5, 1, 0.5]
        try:  # use rawpy to convert raw to openCV
            raw_base = rawpy.imread(imgPath)
            base = raw_base
            self.raw_base = raw_base
            im = base.postprocess(
                    output_color=rawpy.ColorSpace.raw,
                    use_camera_wb=False,
                    highlight_mode=rawpy.HighlightMode.Ignore,
                    user_flip=0,
                    use_auto_wb=False,
                    user_wb=ext_wb,
                    no_auto_bright=False,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR
                    )
            #cv2.imwrite('rawImg.jpg', im)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # pretty much must be a raw format image
        except (LibRawFatalError, LibRawNonFatalError) as e:

            if imgPath != '':
                title = 'Error opening file'
                text = 'Corrupted or incompatible image file.'
                detail_text = (f"LibRawError opening: {imgPath}\nUsually this "
                               "indicates a corrupted or incompatible image."
                               f"\n{e}")
                self.userNotice(text, title, detail_text)
            raise  # Pass this up to the process function for halting
        return im

    def apply_corrections(self):
        """
        applies postprocessing to self.raw_base based on what was learned
        from the initial openImageFile object.
        """
        cc_avg_white = self.cc_avg_white
        if cc_avg_white:  # if a cc_avg_white value was found
            # get max channel value
            maxChan = max(cc_avg_white)
            # get position of max channel value
            maxPos = cc_avg_white.index(maxChan)
            # get the average channel value from all 3 channels
            avgChan = np.mean(cc_avg_white)
            #avgChan = np.partition(cc_avg_white, 1)[1]
            # divide all values by the max channel value
            cc_avg_white = [maxChan/x for x in cc_avg_white]

            # replace the max channel value with avgchannel value / itself 
            cc_avg_white[maxPos] = avgChan / maxChan
            r, g, b = cc_avg_white
            # adjust green channel for the 4-to-3 channel black magicks
            g = g/2
            wb = [r, g, b, g]
            use_camera_wb = False
        else:  # otherwise use as shot values
            use_camera_wb = True
            wb = [1, 1, 1, 1]

        rgb_cor = self.raw_base.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                                            #dcb_enhance=True,
                                            #use_auto_wb=True,
                                            use_camera_wb=use_camera_wb,
                                            user_wb=wb,
                                            #highlight_mode=rawpy.HighlightMode.Blend,
                                            output_color=rawpy.ColorSpace.sRGB,
                                            #output_bps=8,
                                            user_flip=self.flip_value,
                                            user_sat=None,
                                            auto_bright_thr=None,
                                            bright=1.0,
                                            exp_shift=None,
                                            chromatic_aberration=( 1, 1),
                                            #exp_preserve_highlights=1.0,
                                            no_auto_scale=False,
                                            gamma=None
                                            )
        self.raw_base.close()
        del self.raw_base
        self.raw_base = None  # be extra sure we free the ram
        self.im = rgb_cor

    def create_profile(self):
        """
        Called by user, or if no profiles are present on start up. Creates a
        settings profile.
        """
        # be sure we start with a clean slate
        self.reset_working_variables()
        # If folder monitoring is on turn it off
        try:
            if self.folder_watcher.is_monitoring:
                self.toggle_folder_monitoring()
        except AttributeError:
            # may be first time the program opened
            pass
        sd = SettingsWizard(parent=self)
        if sd.run_wizard():
            # if finish button pressed ... swap to main page.
            self.mainWindow.tabWidget.setCurrentIndex(0)

    def edit_current_profile(self):
        """
        calls the setupWizard to edit the currently selected profile
        """
        # be sure we start with a clean slate
        self.reset_working_variables()
        # If folder monitoring is on turn it off
        if self.folder_watcher.is_monitoring:
            self.toggle_folder_monitoring()

        selectedProfile = self.mainWindow.comboBox_profiles.currentText()
        # first get the profiles
        profiles = self.get('profiles', {})
        # then get the selected profiles
        profile = profiles.get(selectedProfile, {})
        profile['profileName'] = selectedProfile
        sd = SettingsWizard(wiz_dict=profile, parent=self)
        if sd.run_wizard():
            # if finish button pressed ... swap to main page.
            self.mainWindow.tabWidget.setCurrentIndex(0)

    def delete_current_profile(self):
        """
        removes the currently selected profile (after asking)
        """
        # identify the selected profile
        selectedProfile = self.mainWindow.comboBox_profiles.currentText()
        if selectedProfile == '':
            # if it is empty, exit early.
            return

        mb_title = f'PERMANENTLY remove profile?'
        mb_text = f'PERMANENTLY remove the profile named {selectedProfile}?'
        mb = self.userAsk(text=mb_text, title=mb_title)
        if mb:
            profiles = self.get('profiles', {})
            profiles.pop(selectedProfile, None)
            self.setValue('profiles', profiles)
            # update the profile options
            self.populate_profile_list()

    def update_profile_settings(self):
        """
        called upon load or after changes are made using the settings wizard.
        Updates the appropriate classes with potentially new information.
        """
        self.reset_working_variables()
        selected_profile = self.mainWindow.comboBox_profiles.currentText()
        # first get the profiles
        profiles = self.get('profiles', {})
        # then get the selected profiles
        profile = profiles.get(selected_profile, {})
        # store the profile as a class variable
        self.profile = profile
        ###
        # initalize the folder_watcher using current user inputs
        ###
        self.setup_Folder_Watcher()
        self.setup_Output_Handler()
        ####
        # set up bcRead
        ###
        patterns = self.profile.get('patterns', '')
        backend = self.profile.get('bcBackend', 'zbar')
        self.bcRead = bcRead(patterns, backend, parent=self.mainWindow)
        # populate self.static_exif
        self.scaleRead = ScaleRead(parent=self.mainWindow)
        ###
        # misc feature classes
        ###
        self.eqRead = eqRead(parent=self.mainWindow, equipmentDict=self.profile.get('equipmentDict', {}))
        ###
        # setup metadata reading/writing class
        ###
        self.metaRead = MetaRead(parent=self.mainWindow)
        self.update_metadata()

    def update_currently_selected_profile(self, selected_profile=False):
        """
        used to store the currently selected profile in settings
        """
        if not selected_profile:
            selected_profile = self.mainWindow.comboBox_profiles.currentText()
        self.setValue('selected_profile', selected_profile)
        self.populateQComboBoxSettings(self.mainWindow.comboBox_profiles,
                                       selected_profile)
        self.update_profile_settings()

    def populate_profile_list(self):
        """
        populates the profiles QCombobox with the stored profiles.
        """
        # identify the target comboBox
        profile_comboBox = self.mainWindow.comboBox_profiles
        # clear current items in the list
        profile_comboBox.clear() 
        # populate the list
        nameList = list(self.get('profiles', {}).keys())
        # if the list of profile names is empty, force the wizard.
        if nameList == []:
            # assign 'results' to _ so that it waits on the wizard to finish.
            _ = self.create_profile()
            return
        profile_comboBox.addItems(nameList)
        previously_selected_profile = self.get('selected_profile', False)
        if not previously_selected_profile:
            # if somehow has profiles and no previously selected, use 1st found
            previously_selected_profile = nameList[0]
        self.mainWindow.comboBox_profiles.setCurrentText(previously_selected_profile)
        # update it in settings
        self.update_currently_selected_profile()

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
        # give app a moment to update
        app.processEvents()
        
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
        # give app a moment to update
        app.processEvents()
        
        reply = msg.exec_()
        return reply

    # Functions related to the saving and retrieval of preference settings
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

    def populateSettings(self):
        """
        Uses self.settings to populate the preferences widget's selections
        """

        # Many fields are connected to save on 'changed' signals
        # block those emissions while populating values.
        children = self.mainWindow.tabWidget.findChildren(QObject)
        [x.blockSignals(True) for x in children]

        # populate technican_list
        self.populate_technician_list()
        # populate the profile list #
        self.populate_profile_list()

        # QComboBox
        comboBox_profiles = self.get('selected_profile', '')
        self.populateQComboBoxSettings( self.mainWindow.comboBox_profiles, comboBox_profiles)

        # radiobutton
        value_DarkTheme = self.get('value_DarkTheme', False)
        self.mainWindow.value_DarkTheme.setChecked(value_DarkTheme)
        value_LightTheme = self.get('value_LightTheme', True)
        self.mainWindow.value_LightTheme.setChecked(value_LightTheme)

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
        comboBox_profiles = self.mainWindow.comboBox_profiles.currentText()
        self.settings.setValue('selected_profile', comboBox_profiles)

        # radiobutton
        value_DarkTheme = self.mainWindow.value_DarkTheme.isChecked()
        self.settings.setValue('value_DarkTheme', value_DarkTheme)
        value_LightTheme = self.mainWindow.value_LightTheme.isChecked()
        self.settings.setValue('value_LightTheme', value_LightTheme)

        # cleanup, after changes are made, should re-initalize some classes
        self.setup_Folder_Watcher()
        self.setup_Output_Handler()
        self.update_metadata()

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
    #app.setStyle("plastique")
    #app.setStyle("fusion")
    if w.get('value_DarkTheme', False):
        w.setStyleSheet(darkorange.getStyleSheet())

    w.show()
    sys.exit(app.exec_())
