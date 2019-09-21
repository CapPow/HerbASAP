"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QWizard, QSizePolicy, QWhatsThis,
                             QMessageBox, QDialog, QApplication)
from PyQt5.QtCore import Qt
from ui.settingsWizardUI import Ui_Wizard
from ui.imageDialogUI import Ui_Dialog_image

from libs.eqRead import eqRead

import numpy as np
import cv2
import re
import os
import rawpy
from rawpy import LibRawNonFatalError, LibRawFatalError


class ImageDialog(QDialog):
    def __init__(self, img_array_object):
        super().__init__()
        self.init_ui(img_array_object)

    def init_ui(self, img_array_object):
        mb = Ui_Dialog_image()
        mb.setupUi(self)

        height, width = img_array_object.shape[0:2]
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(img_array_object.copy(),
                            width, height, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        mb.label_Image.setPixmap(pixmap)


class Canvas(QtWidgets.QLabel):

    def __init__(self, im, parent=None):
        super().__init__()
        #pixmap = QtGui.QPixmap(600, 300)
        #self.setPixmap(pixmap)
        self.parent = parent
        self.setObjectName("canvas")
        self.backDrop, self.correction = self.genPixBackDrop(im)
        self.setPixmap(self.backDrop)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        # the box_size, is the point of the entire canvas.
        # The scale corrected larger dim of the annotated rect
        self.scaled_begin = (0, 0)
        self.scaled_end = (0, 0)

    def genPixBackDrop(self, im):
        # if it is oriented in landscape, rotate it
        h, w = im.shape[0:2]
        if h < w:
            im = np.rot90(im, 3)  # 3 or 1 would be equally appropriate
            h, w = w, h  # swap the variables after rotating
        bytesPerLine = 3 * w
        # odd bug here, must use .copy() to avoid a mem error.
        # see: https://stackoverflow.com/questions/48639185/pyqt5-qimage-from-numpy-array
        qImg = QtGui.QImage(im.copy(), w, h, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        width = self.parent.wiz.crc_canvas_cont.width()
        height = self.parent.wiz.crc_canvas_cont.height()
        pixmap = pixmap.scaled(width, height,
                               QtCore.Qt.KeepAspectRatio,
                               Qt.FastTransformation)
        # corrections are doubled due to display image bieng opened at half res
        h_correction = (h) / height
        w_correction = (w) / width
        correction = (w_correction, h_correction)
        return pixmap, correction

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.drawPixmap(self.rect(), self.backDrop)
        # set brush to a lime green
        br = QtGui.QBrush(QtGui.QColor('#03EA00'))
        qp.setBrush(br)
        qp.drawRect(QtCore.QRect(self.begin, self.end))

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
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
        self.updateBoxSize()

    def updateBoxSize(self):
        # qpoint is formatted as (xpos, ypos)
        # x col, y row
        # save the scale corrected start / end points
        x_corr, y_corr = self.correction
        b_x = self.begin.x()
        b_y = self.begin.y()
        e_x = self.end.x()
        e_y = self.end.y()
        # determine the start (s) and end (e) points, adjusted for scale
        sb_x, sb_y = int(b_x * x_corr), int(b_y * y_corr)
        self.scaled_begin = (sb_x, sb_y)
        se_x, se_y= int(e_x * x_corr), int(e_y * y_corr)
        self.scaled_end = (se_x, se_y)
        # determine the scaled lengths of x & y lines
        x_len = abs(sb_x - se_x)
        y_len = abs(sb_y - se_y)
        # set the max value among the x, y lengths to the box_size
        box_size = max(x_len, y_len)
        
        # You must have atleast 25 pixels to ride this ride.
        box_size= int(box_size * 1.1)
        # send the box_size over for correction and storage as partition_size
        self.parent.save_corrected_partition_size(box_size)
        

class SettingsWizard(QWizard):

    def __init__(self, wiz_dict={}, parent=None):
        super().__init__()
        self.init_ui()
        self.parent = parent
        # a container to hold and pass along the user entries
        self.wiz_dict = wiz_dict
        # pass out the wiz_dict values handed in
        self.populate_wiz_fields()
        self.img = None
        self.ext = None
        self.canvas = None
        self.profile_saved = False
        self.partition_size = 125 # a default value in case they never set one.
        self.colorCheckerDetection = False # has the crc been tested?
        self.working = False # used to hold "next" buttons until finished doing something.

    def init_ui(self):
        self.wiz = Ui_Wizard()
        self.wiz.setupUi(self)
        self.eqRead = eqRead(parent=self.wiz)
        # attach update_profile_details to the next button
        self.button(QWizard.NextButton).clicked.connect(self.update_profile_details)
        self.button(QWizard.HelpButton).clicked.connect(self.enter_whatsthis_mode)
        # Reimpliment isComplete() for appropriate pages
        # intro_page
        def isComplete():
            if (self.working) or (self.img is None):
                res = False
            else:
                res = True
                
            return res
        self.wiz.intro_page.isComplete = isComplete
        # ccRead_setup_page1
        def isComplete():
            if self.working:
                res = False
            else:
                res = True
            return res
        # assign it
        self.wiz.ccRead_setup_page1.isComplete = isComplete
        # path_setup_page
        def isComplete():
            res = False
            inputPath = self.wiz.lineEdit_inputPath.text()
            if os.path.isdir(inputPath) and inputPath != '':
                res = True
                if self.wiz.group_saveProcessedJpg.isChecked() and not os.path.isdir(self.wiz.lineEdit_pathProcessedJpg.text()):
                    res = False
                if self.wiz.group_keepUnalteredRaw.isChecked() and not os.path.isdir(self.wiz.lineEdit_pathProcessedJpg.text()):
                    res = False
                if self.wiz.group_saveProcessedTIFF.isChecked() and not os.path.isdir(self.wiz.lineEdit_pathProcessedTIFF.text()):
                    res = False
            return res
        # assign it
        self.wiz.path_setup_page.isComplete = isComplete
        # final_page
        def isComplete():
            return self.profile_saved
        # assign it
        self.wiz.final_page.isComplete = isComplete

    def enter_whatsthis_mode(self):
        QWhatsThis.enterWhatsThisMode()

    def is_nameAvailable(self, profile_name):
        """
        determines if the given profile_name is unique
        """
        if len(profile_name) > 2:
            profiles = self.parent.get('profiles', {})
            if profile_name in profiles.keys():
                formatted_result = "Profile name is NOT unique!"
            else:
                formatted_result = "Profile name is available"
            
            self.wiz.label_nameAvailable.setText(formatted_result)

    def emit_completeChanged(self):
        currID = self.currentId()
        QApplication.processEvents()
        if currID == 0:
            self.wiz.intro_page.completeChanged.emit()
        elif currID == 1:
            self.wiz.path_setup_page.completeChanged.emit()
        elif currID == 4:
            self.wiz.ccRead_setup_page1.completeChanged.emit()
        elif currID == 8:
            self.wiz.final_page.completeChanged.emit()

    def setFolderPath(self):
        """ Called by all of the "Browse" buttons in Image Location Tab.
        Assigns selected folder name to the associated lineEdit.
        Uses hacky methods to lineEdit associated to button."""

        # this only works with strict object naming conventions.
        # get name of button pressed
        buttonPressed = self.sender().objectName().split('_')[-1]
        # use string methods to get the associated lineEdit name
        # use eval method to convert string to variable.
        targetField = eval(f'self.wiz.lineEdit_{buttonPressed}')
        targetDir = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Select a folder:', QtCore.QDir.homePath(),
                QtWidgets.QFileDialog.ShowDirsOnly)
        if targetDir != '':
            targetField.setText(targetDir)

    def userNotice(self, text, title='', detailText=None):
        """ a general user notice """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(text)
        if detailText is not None:
            msg.setDetailedText(detailText)
        msg.setWindowTitle(title)
        reply = msg.exec_()
        return reply

    def userAsk(self, text, title='', detailText=None):
        """ a general user dialog with yes / cancel options"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText(text)
        msg.setWindowTitle(title)
        if detailText:
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

    def openDisplayImage(self):
        """
        Called when pushButton_selectExampleImage is pressed, requests a path
        to return a 'raw' image file and attempts to useit to initiate a Canvas
        object. The raw image is opened using default postProcessing params.
        While not totally accurate is used for displaying examples and
        performing tests in setupWizard.
        """
        imgPath = self.ask_img_path()
        if imgPath == '':
            return
        try:  # use rawpy to convert raw to openCV
            self.working = True
            self.emit_completeChanged()
            with rawpy.imread(imgPath) as raw:
                im = raw.postprocess(output_color=rawpy.ColorSpace.raw,
                                     use_auto_wb=True,
                                     demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR
                                     )
                # identify the destination for the draw-able canvas object
                ccRead_layout = self.wiz.crc_canvas_layout
                # create the draw-able canvas object with the image object.
                if self.canvas:
                    self.canvas.close()
                self.canvas = Canvas(im, self)
                # add it to the destination
                ccRead_layout.addWidget(self.canvas)
                # enable test buttons
                self.wiz.pushButton_testbcRead.setEnabled(True)
                self.wiz.pushButton_testBlurDetection.setEnabled(True)
                self.wiz.pushButton_testeqRead.setEnabled(True)
               
                # store class variables concerning the file selected
                self.img = im  # the opened image object
                self.imgPath = imgPath  # the path to the selected file
                file_name, self.ext = os.path.splitext(imgPath)
                base_file_name = os.path.basename(imgPath)
                self.wiz.label_loadedExampleFile.setText(f'{base_file_name} Loaded')
                self.wiz.label_loadedExampleFile_1.setText(f'{base_file_name} Loaded')
            
            # determine the imaging equipment, and generate correction
            self.populate_lenses()
            self.gen_distort_corrections()
            
            self.working = False
            # inform the UI the loading has completed
            self.emit_completeChanged()

        # pretty much must be a raw format image
        except (LibRawFatalError, LibRawNonFatalError) as e:
            if imgPath != '':
                title = 'Error opening file'
                text = 'Corrupted or incompatible image file.'
                detail_text = (f"LibRawError opening: {imgPath}\nUsually this "
                               "indicates a corrupted or incompatible image."
                               "\n{e}")
                self.userNotice(text, title, detail_text)
                
            self.wiz.pushButton_testbcRead.setEnabled(False)
            self.wiz.pushButton_testBlurDetection.setEnabled(False)
            self.wiz.pushButton_testeqRead.setEnabled(False)
            self.wiz.label_loadedExampleFile.setText('')
            self.wiz.label_loadedExampleFile_1.setText('')

    def ask_img_path(self):
        # ask the user for an example image file
        imgPath, _ = QtWidgets.QFileDialog.getOpenFileName(None,
                                                    "Open an Example Image")
        return imgPath

    ###
    # barcode reading related functions
    ###
    def update_CatalogNumber_Preview(self):
        """
        called after changes are made to the catalog number related barcode
        input fields present on bcRead_setup_page.
        """

        dupNamingPolicy = self.wiz.comboBox_dupNamingPolicy.currentText()
        if dupNamingPolicy == 'LOWER case letter':
            suffix = 'a'
        elif dupNamingPolicy == 'UPPER case letter':
            suffix = 'A'
        elif dupNamingPolicy == 'Number with underscore':
            suffix = '_1'
        elif dupNamingPolicy == 'OVERWRITE original':
            suffix = ''
        # update bc pattern preview
        prefix = self.wiz.lineEdit_catalogNumberPrefix.text()
        digits = int(self.wiz.spinBox_catalogDigits.value())
        if digits == 0:
            startingNum = '(anything)'
        else:
            startingNum = ''.join(str(x) for x in list(range(digits)))
            startingNum = startingNum.zfill(digits)  # fill in leading zeroes
        # update dup naming preview
        dupPreviewText = f'{prefix}{startingNum}{suffix}'
        self.wiz.label_dupPreviewDisplay.setText(dupPreviewText)  # set it

    def add_pattern(self):
        """
        called when toolButton_addPattern is pressed
        """
        prefix = self.wiz.lineEdit_catalogNumberPrefix.text()
        digits = int(self.wiz.spinBox_catalogDigits.value())
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
        listWidget_patterns = self.wiz.listWidget_patterns
        listWidget_patterns.addItem(collRegEx)
        item_pos = listWidget_patterns.count() - 1
        item = listWidget_patterns.item(item_pos)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.set_bc_pattern()

    def remove_pattern(self):
        """
        called when toolButton_removePattern is pressed
        """
        listWidget_patterns = self.wiz.listWidget_patterns
        selection = listWidget_patterns.currentRow()
        item = listWidget_patterns.takeItem(selection)
        item = None
        self.set_bc_pattern()

    def retrieve_bc_patterns(self):
        """
        harvests all pattern strings in the listWidget_patterns and returns
        them as a unique "|" joined set
        """
        listWidget_patterns = self.wiz.listWidget_patterns
        # is there really no way to get everything from a listWidget?
        patterns = listWidget_patterns.findItems('', Qt.MatchContains)
        patterns = "|".join(set(x.text() for x in patterns))
        if patterns == '':
            # if the pattern is empty string, pass a fully permissive pattern
            patterns = '^(.*)'
        return patterns

    def set_bc_pattern(self):
        """ harvests all pattern strings in the listWidget_patterns, joins
        them and sends them to self.bcRead.compileRegexPattern which in turn
        sets the bcRead.rePattern attribute."""
        patterns = self.retrieve_bc_patterns()
        try:
            self.wiz_dict['rePattern'] = re.compile(patterns)
        except re.error:
            notice_title = 'Regex Pattern Error'
            notice_text = f'Warning, improper regex pattern.'
            detail_text = f'Regex patterns failed to compile.\n{patterns}'
            self.userNotice(notice_text, notice_title, detail_text)

    def fill_patterns(self, joinedPattern):
        """
        Populates the listWidget_patterns with saved patterns. To be used when
        editing existing profile
        """
        patterns = self.retrieve_bc_patterns()
        patterns = joinedPattern.split('|')
        listWidget_patterns = self.wiz.listWidget_patterns
        listWidget_patterns.clear()
        for i, pattern in enumerate(patterns):
            # save the item
            listWidget_patterns.addItem(pattern)
            # now that it exists set the flag as editable
            item = listWidget_patterns.item(i)
            item.setFlags(item.flags() | Qt.ItemIsEditable)

    def test_bcRead(self):
        """
        used to test the bcRead functions
        """
        self.update_CatalogNumber_Preview()
        patterns = self.retrieve_bc_patterns()
        self.fill_patterns(patterns)
        backend = self.wiz.comboBox_bcBackend.currentText()
        from libs.bcRead import bcRead
        self.bcRead = bcRead(patterns, backend, parent=self.wiz)
        grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        bcResults = str(self.bcRead.decodeBC(grey))
        if bcResults == '[]':
            bcResults = 'FAILED, no codes found.'
        self.wiz.label_bcRead_results.setText(f'results: {bcResults}')

    ###
    # blur detection related functions
    ###
    def test_blurDetect(self):
        """
        used to test the blurDetect functions
        """
        from libs.blurDetect import blurDetect
        blurDetect = blurDetect()
        blur_threshold = self.wiz.doubleSpinBox_blurThreshold.value()
        grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur_results = blurDetect.blur_check(grey, blur_threshold, True)
        formatted_results = f'Laplacian Norm: {round(blur_results.get("lapNorm", 00), 3)}\nIs Blurry: {blur_results.get("isblurry", "")}'
        self.wiz.label_blurDetect_results.setText(formatted_results)

    ###
    # CRC detection related functions
    ###
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

    def save_corrected_partition_size(self, box_size):
        """
        given a box_size from the 'draw-able' canvas, stores a corrected
        partition size as a class variable (to this class)
        """
        self.working = True
        self.emit_completeChanged()
        # resize is necessary for the correction
        original_size, reduced_img = self.scale_images_with_info(self.img)
        # determine a correction value to det! a partition_size, from box_size
        ow, oh = original_size
        nh, nw = reduced_img.shape[0:2]
        correction_val = nh / oh
        corrected_box_size = int(round(box_size * correction_val,0))
        # keep a floor partition_size of 125 
        partition_size = max([125, corrected_box_size])
        print(f'using a {partition_size} partition_size')
        self.partition_size = partition_size
        self.wiz.pushButton_testCRCDetection.setEnabled(True)
        # force a test after drawing the box
        self.test_crcDetection()
        self.working = False
        self.emit_completeChanged()

    def test_crcDetection(self):
        """
        used to test basic crc detection
        """
        from libs.ccRead import ColorchipRead
        
        self.working = True
        self.emit_completeChanged()
        try:
            colorchipDetect = ColorchipRead(parent=self.wiz)
            crc_type = self.wiz.comboBox_crcType.currentText()
            if crc_type == "ISA ColorGauge Nano":  # aka small crc
                original_size, reduced_img = self.scale_images_with_info(self.img)
                self.cc_position, self.cropped_cc, self.cc_crop_time = colorchipDetect.process_colorchip_small(reduced_img,
                                                                                                             original_size,
                                                                                                             stride_style='quick',
                                                                                                             high_precision=True,
                                                                                                             partition_size=self.partition_size)
            else:
                self.cc_position, self.cropped_cc, self.cc_crop_time = colorchipDetect.process_colorchip_big(self.img)
            
            mb = ImageDialog(self.cropped_cc)
            if mb.exec():
                formatted_result = f"Detected CRC in {round(self.cc_crop_time, 3)} seconds."
                self.wiz.pushButton_test_scaleDet.setEnabled(True)
                self.wiz.pushButton_test_whiteBalance.setEnabled(True)
                self.colorCheckerDetection = True # successfully tested crc
            else:
                formatted_result = 'TEST FAILED'
                self.wiz.pushButton_test_scaleDet.setEnabled(False)
                self.wiz.pushButton_test_whiteBalance.setEnabled(False)

        except Exception as e:  # really any exception here should be handled.
            notice_title = 'Error locating CRC'
            notice_text = 'Failed to locate the CRC. Verify the CRC type '
            ' selection and try drawing a more precise box.'
            detail_text = ('The following exception was rasied:'
                           f'\n{e}')
            self.userNotice(notice_text, notice_title, detail_text)
            formatted_result = 'TEST FAILED'
            self.wiz.pushButton_test_scaleDet.setEnabled(False)
            self.wiz.pushButton_test_whiteBalance.setEnabled(False)
        
        self.wiz.label_crcDetection_results.setText(formatted_result)
        self.working = False
        self.emit_completeChanged()
        
    def test_scaleRead(self):
        """
        used to test scaleReading based on CRC
        """
        from libs.scaleRead import ScaleRead
        scaleRead = ScaleRead(parent=self.wiz)
        
        try:
            x1, y1, x2, y2 = self.cc_position
            full_res_cc = self.img[y1:y2, x1:x2].copy()
            # lookup the patch area and seed function
            # if setting not found, use isa nano settings
            patch_mm_area, seed_func, to_crop = scaleRead.scale_params.get(
                    self.wiz.comboBox_crcType.currentText(),
                    (10.0489, scaleRead.det_isa_nano_seeds, True))
            ppmm, ppmm_uncertainty = scaleRead.find_scale(
                    full_res_cc,
                    patch_mm_area,
                    seed_func,
                    to_crop
                    )

            if ppmm > 0:
                formatted_result = (f"pixel:mm scale = {ppmm}\n"
                                    f"scale uncertainty = {ppmm_uncertainty}")
            else:
                raise Exception('unable to make sufficient measurements to safely infer pixel:mm scale.')

        except Exception as e:  # really any exception here should be handled.
            notice_title = 'Error Determining Scale'
            notice_text = 'Failed to determine scale from CRC patches'
            detail_text = ('Failed attempting to determine a pixel:mm scale'
                           f'\n{e}')
            self.userNotice(notice_text, notice_title, detail_text)
            formatted_result = 'TEST FAILED'
        
        self.wiz.label_scaleDet_results.setText(formatted_result)

    def test_wbDet(self):
        """
        used to test the determination of an average 'white' RGB value from CRC
        """
        from libs.ccRead import ColorchipRead
        colorchipDetect = ColorchipRead(parent=self.wiz)
        try:
            cc_avg_white, cc_black_point = colorchipDetect.predict_color_chip_whitevals(self.cropped_cc)
            if isinstance(cc_avg_white, list):
                formatted_results = f"avg white RGB from CRC = {cc_avg_white}"

        except Exception as e:  # really any exception here should be handled.
            notice_title = 'Error Determining White Balance Value'
            notice_text = 'Failed to calculate average white RGB value from CRC patches'
            detail_text = '\n{e}'
            self.userNotice(notice_text, notice_title, detail_text)
            formatted_result = 'TEST FAILED'

        self.wiz.label_whiteBalance_results.setText(formatted_results)

        ###
        # eqRead test results
        ###
    def test_eqRead(self):
        """
        Used to determine if the equipment can be properly detected in order
        for eqRead to properly make equipment corrections.
        """
        try:
            self.gen_distort_corrections()
            self.eqRead.lensCorrect(self.img)
            formatted_result = 'Success generating correction matrix!'
            self.wiz.label_eqRead_results.setText(formatted_result)
            self.wiz.checkBox_lensCorrection.setChecked(True)
        except Exception as e:  # really any exception here should be handled.
            notice_title = 'Error Determining Equipment'
            notice_text = 'Equipment determination / correction failed'
            detail_text = ('Failed while testing features necessary to correct'
                           'equipment based distortions. '
                           f'\n{e}')
            self.userNotice(notice_text, notice_title, detail_text)
            formatted_result = 'TEST FAILED'
            self.wiz.label_eqRead_results.setText(formatted_result)
            self.wiz.checkBox_lensCorrection.setChecked(True)

    def populate_lenses(self):
        """
        fills the lens selection qcombobox with options.
        """
        self.equipmentDict = self.eqRead.detImagingEquipment(self.imgPath)
        models = self.equipmentDict.get('cams', [''])
        # update the cam model text
        model = models[0]
        print(model)
        formatted_result = f'{model}'
        self.wiz.label_camModel.setText(formatted_result)
        lenses = self.equipmentDict.get('lenses', [''])
        # populate the lens model qcombo box
        self.populateQComboBoxSettings(self.wiz.comboBox_lensModel, lenses)
        

    def gen_distort_corrections(self):
        """
        attempts to generate the distortion correction matrix
        """
        
        selected_lens = self.wiz.comboBox_lensModel.currentText()
        if selected_lens is "Not Available (no lens corrections are available)":
            # if no lenses selected, stop early and uncheck "lensCorrection"
            self.wiz.checkBox_lensCorrection.setChecked(False)
            return
        # select the 'best' guess
        self.selectQComboBoxSettings(self.wiz.comboBox_lensModel, selected_lens)
        self.equipmentDict['lens'] = selected_lens
        self.equipmentDict['cam'] = self.equipmentDict.get('cams', [''])[0]
        imgShape = self.img.shape[0:2]
        # because we don't know the proper rotation, assume most images will be taller than wide
        # If this is wrong, the first process of each image will take a little longer
        height = max(imgShape)
        width = min(imgShape)
        self.equipmentDict['height'] = height
        self.equipmentDict['width'] = width
        # focal distance should be in meters
        focalDistance = self.wiz.doubleSpinBox_focalDistance.value()
        focalDistance = round(focalDistance / 100, 5)
        self.equipmentDict['focalDistance'] = focalDistance
        # reinitalizing eqRead also calls setmod
        self.eqRead = eqRead(parent=self.wiz, equipmentDict = self.equipmentDict)

    def get_selected_lens(self):
        """
        called when the lensfunpy object is needed from user's lens selection.
        """
        current_index= self.wiz.comboBox_lensModel.currentIndex()
        if current_index == 0:
            return None
        else:
            current_index =- 1
            lens = self.equipmentDict['lenses'][current_index]
            return lens

    def selectQComboBoxSettings(self, obj, value):
        """ sets a QComboBox based on a string value. Presumed to be a more
        durable method. obj is the qComboBox object, and value is a string
        to search for"""
        self.wiz.comboBox_lensModel.blockSignals(True)
        index = obj.findText(value)
        obj.setCurrentIndex(index)
        self.wiz.comboBox_lensModel.blockSignals(False)

    def populateQComboBoxSettings(self, qbox_object, value_list):
        """
        populates the obj qbox_object with the value_list.
        """
        self.wiz.comboBox_lensModel.blockSignals(True)
        # identify the target comboBox
        profile_comboBox = qbox_object
        # clear current items in the list
        profile_comboBox.clear()
        # clean the value_list of empty strings
        value_list = [str(x) for x in value_list if x != '']
        # insert the "Not available text at top of list" then populate it
        value_list.insert(0, "Not Available (no lens corrections are available)")
        # if the list of profile names is empty, force the wizard.
        profile_comboBox.addItems(value_list)
        self.wiz.comboBox_lensModel.blockSignals(False)

    def convertCheckState(self, stringState):
        """
        given a string either "true" or "false" returns the proper
        Qt.CheckState
        """
        if str(stringState).lower() != 'true':
            return Qt.Unchecked
        else:
            return Qt.Checked

    def populate_wiz_fields(self):
        """
        Called when the class is initalized, populates the appropriate fields
        """
        # make it easier to access the input dict
        d = self.wiz_dict
        # make it easier to access the objects
        w = self.wiz

        if d == {}:
            # if the dict is empty wrap up early.
            return

        # class variables:
        # if partition_size was handed it (i.e., it is an edit profile event)
        partition_size = d.get('partition_size', None)
        if partition_size:
            self.wiz.pushButton_testCRCDetection.setEnabled(True)
            self.partition_size = partition_size

        colorCheckerDetection = d.get('colorCheckerDetection', False)
        self.colorCheckerDetection = colorCheckerDetection
        # populate listWidget_patterns        
        self.fill_patterns(d.get('patterns', ''))

        # QComboBox
        comboBox_colorCheckerPosition = d.get('colorCheckerPosition', 'Upper right')
        self.selectQComboBoxSettings( w.comboBox_colorCheckerPosition, comboBox_colorCheckerPosition)
        comboBox_bcBackend = d.get('bcBackend', 'zbar')
        self.selectQComboBoxSettings( w.comboBox_bcBackend, comboBox_bcBackend)
        comboBox_dupNamingPolicy = d.get('dupNamingPolicy', 'LOWER case letter')
        self.selectQComboBoxSettings( w.comboBox_dupNamingPolicy, comboBox_dupNamingPolicy)
        comboBox_crcType = d.get('crcType', 'ISA ColorGauge Nano')
        self.selectQComboBoxSettings( w.comboBox_crcType, comboBox_crcType)
        comboBox_colorCheckerPosition = d.get('colorCheckerPosition', 'Upper right')
        self.selectQComboBoxSettings( w.comboBox_colorCheckerPosition, comboBox_colorCheckerPosition)
        
        # QLineEdit
        lineEdit_profileName = d.get('profileName', '')
        w.lineEdit_profileName.setText(lineEdit_profileName)
        lineEdit_inputPath = d.get('inputPath', '')
        w.lineEdit_inputPath.setText(lineEdit_inputPath)
        lineEdit_pathProcessedJpg = d.get('pathProcessedJpg', '')
        w.lineEdit_pathProcessedJpg.setText(lineEdit_pathProcessedJpg)
        lineEdit_pathUnalteredRaw = d.get('pathUnalteredRaw', '')
        w.lineEdit_pathUnalteredRaw.setText(lineEdit_pathUnalteredRaw)
        lineEdit_catalogNumberPrefix = d.get('catalogNumberPrefix', '')
        w.lineEdit_catalogNumberPrefix.setText(lineEdit_catalogNumberPrefix)
        
        # QPlainTextEdit
        plainTextEdit_collectionName = d.get('collectionName', '')
        w.plainTextEdit_collectionName.setPlainText(plainTextEdit_collectionName)
        plainTextEdit_collectionURL = d.get('collectionURL', '')
        w.plainTextEdit_collectionURL.setPlainText(plainTextEdit_collectionURL)
        plainTextEdit_contactEmail = d.get('contactEmail', '')
        w.plainTextEdit_contactEmail.setPlainText(plainTextEdit_contactEmail)
        plainTextEdit_contactEmail = d.get('contactEmail', '')
        w.plainTextEdit_contactEmail.setPlainText(plainTextEdit_contactEmail)
        plainTextEdit_contactName = d.get('contactName', '')
        w.plainTextEdit_contactName.setPlainText(plainTextEdit_contactName)
        plainTextEdit_copywriteLicense = d.get('copywriteLicense', '')
        w.plainTextEdit_copywriteLicense.setPlainText(plainTextEdit_copywriteLicense)
        
        # QCheckBox
        checkBox_blurDetection = self.convertCheckState(d.get('blurDetection', 'true'))
        w.checkBox_blurDetection.setChecked(checkBox_blurDetection)
        checkBox_scaleDetermination = self.convertCheckState(d.get('scaleDetermination', 'true'))
        w.checkBox_scaleDetermination.setChecked(checkBox_scaleDetermination)
        checkBox_performWhiteBalance = self.convertCheckState(d.get('performWhiteBalance', 'true'))
        w.checkBox_performWhiteBalance.setChecked(checkBox_performWhiteBalance)
        checkBox_performWhiteBalance = self.convertCheckState(d.get('performWhiteBalance', 'true'))
        w.checkBox_performWhiteBalance.setChecked(checkBox_performWhiteBalance)
        checkBox_lensCorrection = self.convertCheckState(d.get('lensCorrection', 'true'))
        w.checkBox_lensCorrection.setChecked(checkBox_lensCorrection)

        # QGroupbox (checkstate)
        group_saveProcessedJpg = self.convertCheckState(d.get('saveProcessedJpg', 'true'))
        w.group_saveProcessedJpg.setChecked(group_saveProcessedJpg)
        group_keepUnalteredRaw = self.convertCheckState(d.get('keepUnalteredRaw', 'false'))
        w.group_keepUnalteredRaw.setChecked(group_keepUnalteredRaw)
        group_renameByBarcode = self.convertCheckState(d.get('renameByBarcode', 'true'))
        w.group_renameByBarcode.setChecked(group_renameByBarcode)
        group_verifyRotation = self.convertCheckState(d.get('verifyRotation', 'true'))
        w.group_verifyRotation.setChecked(group_verifyRotation)
        
        # QSpinBox
        spinBox_catalogDigits = int(d.get('catalogDigits', 0))
        w.spinBox_catalogDigits.setValue(spinBox_catalogDigits)

        # QDoubleSpinBox
        doubleSpinBox_blurThreshold = float(d.get('blurThreshold', 0.045))
        w.doubleSpinBox_blurThreshold.setValue(doubleSpinBox_blurThreshold)
        doubleSpinBox_focalDistance = float(d.get('focalDistance', 25.5))
        w.doubleSpinBox_focalDistance.setValue(doubleSpinBox_focalDistance)
        

    def build_profile_dict(self):
        """
        Builds a settings profile dict from the appropriate fields, storing as
        a class variable "wiz_dict"
        """

        self.wiz_dict = {
                # class variable(s)
                "colorCheckerDetection": self.colorCheckerDetection,
                "equipmentDict":self.equipmentDict,
                # self.wiz.path_setup_page
                "inputPath": self.wiz.lineEdit_inputPath.text(),  # input monitor folder path
                "saveProcessedJpg": self.wiz.group_saveProcessedJpg.isChecked(),  # checked = saveJPG
                "pathProcessedJpg": self.wiz.lineEdit_pathProcessedJpg.text(),  # output jpg folder path
                "keepUnalteredRaw": self.wiz.group_keepUnalteredRaw.isChecked(), #  checked = move unaltered
                "pathUnalteredRaw": self.wiz.lineEdit_pathUnalteredRaw.text(),  # unaltered raw folder path
                
                # self.wiz.bcRead_setup_page
                "renameByBarcode": self.wiz.group_renameByBarcode.isChecked(),  # checked = rename by barcode
                "bcBackend": self.wiz.comboBox_bcBackend.currentText(),  # bc back end comboBox
                "catalogNumberPrefix": self.wiz.lineEdit_catalogNumberPrefix.text(),  # prefix
                "catalogDigits": self.wiz.spinBox_catalogDigits.value(),  # digits
                "dupNamingPolicy": self.wiz.comboBox_dupNamingPolicy.currentText(),  # dup naming policy
                "patterns": self.retrieve_bc_patterns(),  # pattern list
                
                # self.wiz.blurDetect_setup_page
                "blurDetection": self.wiz.checkBox_blurDetection.isChecked(),  # checked = perform blur detection
                "blurThreshold": self.wiz.doubleSpinBox_blurThreshold.value(),  # blur threshold
                
                # self.wiz.ccRead_setup_page1
                "crcType": self.wiz.comboBox_crcType.currentText(),  # crcType options
                # self.wiz.crc_canvas_layout # canvas with attributes related to partition size: self.scaled_begin, self.scaled_end
                "partition_size": self.partition_size,
                
                # self.wiz.ccRead_setup_page2
                "scaleDetermination": self.wiz.checkBox_scaleDetermination.isChecked(),  # checked = perform scale Det!
                "verifyRotation": self.wiz.group_verifyRotation.isChecked(),  # checked = verify / correct image rotation
                "colorCheckerPosition": self.wiz.comboBox_colorCheckerPosition.currentText(),  # combobox with CRC position options
                "performWhiteBalance": self.wiz.checkBox_performWhiteBalance.isChecked(),  # checked = white balance images
                
                # self.wiz.eqRead_setup_page
                "lensCorrection": self.wiz.checkBox_lensCorrection.isChecked(),  # checked = lens corrections
                "focalDistance": self.wiz.doubleSpinBox_focalDistance.value(),  # approx focal distance (cm)
                
                # self.wiz.metaRead_setup_page
                "collectionName": self.wiz.plainTextEdit_collectionName.toPlainText(),
                "collectionURL": self.wiz.plainTextEdit_collectionURL.toPlainText(),
                "contactEmail": self.wiz.plainTextEdit_contactEmail.toPlainText(),
                "contactName": self.wiz.plainTextEdit_contactName.toPlainText(),
                "copywriteLicense": self.wiz.plainTextEdit_copywriteLicense.toPlainText()
                }

    def update_profile_details(self):
        """
        updates known details about the profile in the final page.
        """
        self.build_profile_dict()
        # list the profile details on the final page.
        formatted_profile = "\n".join([f"{k}: {v}" for k, v in self.wiz_dict.items() if k != 'undistCoords'])
        self.wiz.label_profileDetails.setText(formatted_profile)

    def save_inputs(self):
        """
        Builds a settings profile dict from the appropriate fields and saves it
        to the parent settings
        """
        # buid most recent profile dict
        self.build_profile_dict()
        # identify the profiles stored in parent app's settings.
        profiles = self.parent.get('profiles', {})
        # verify the name is unique & if not check if okay to overwrite.
        profile_name = self.wiz.lineEdit_profileName.text()
        if len(profile_name) > 2:
            to_save = True
            if profile_name in profiles.keys():
                mb_title = f"Overwrite {profile_name}?"
                mb_text = (f"A profile named {profile_name} already exists!"
                           " OVERWRITE it with these settings?")
                userAgree = self.userAsk(text=mb_text, title=mb_title)
                if userAgree:
                    to_save = True
                    formatted_result = "Profile Saved!"
                else:
                    to_save = False
                    formatted_result = ""
            else:
                to_save = True
                formatted_result = "Profile Saved!"
        else:
            # profile name should be > 2 characters ...
            formatted_result = "Profile name should have a minimum of 3 characters"
        if to_save:
            # actually save the profile details
            profiles[profile_name] = self.wiz_dict
            self.parent.setValue('profiles', profiles)
            # set this as the current profile
            # update qComboBox on "mainapp"
            self.parent.populate_profile_list()
            self.parent.update_currently_selected_profile(profile_name)
            self.update_profile_details()  # update the profile details
            # reset availability text from entry field.
            self.wiz.label_nameAvailable.setText('')
            
            # set the wizard final page to "completed"
            self.profile_saved = True

        self.wiz.label_saveProfile_Results.setText(formatted_result)
        # see if it is appropriate to enable the finish button
        self.emit_completeChanged()

    def run_wizard(self):
        wiz_window = self.exec()

