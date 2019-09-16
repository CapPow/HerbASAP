"""
    HerbASAP - Herbarium Application for Specimen Auto-Processing
    performs post processing steps on raw format images of natural history
    specimens. Specifically designed for Herbarium sheet images.
"""
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWizard, QSizePolicy, QMessageBox, QDialog
from PyQt5.QtCore import Qt
from ui.settingsWizardUI import Ui_Wizard
from ui.imageDialogUI import Ui_Dialog_image

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
        self.box_size = 0

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
        width = self.width()
        height = self.height()
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
        self.box_size = max(x_len, y_len)
        
        if self.box_size > 25:
            self.parent.wiz.pushButton_testCRCDetection.setEnabled(True)
        else:
            self.parent.wiz.pushButton_testCRCDetection.setEnabled(False)
        print(f'box_size = {self.box_size}')

class SettingsWizard(QWizard):

    def __init__(self, wiz_dict={}):
        super().__init__()
        self.init_ui()
        # a container to hold and pass along the various user entries
        self.wiz_dict = wiz_dict
        
        self.img = None
        self.ext = None
        self.canvas = None

    def init_ui(self):
        self.wiz = Ui_Wizard()
        self.wiz.setupUi(self)

        # assign the mandatory fields as appropriate.

    def populate_wiz_fields():
        """
        populates the various wizard input fields, useful when editing an
        existing provile
        """

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
                # store class variables concerning the file selected
                self.img = im  # the opened image object
                self.imgPath = imgPath  # the path to the selected file
                file_name, self.ext = os.path.splitext(imgPath)
                base_file_name = os.path.basename(imgPath)
                self.wiz.label_loadedExampleFile.setText(f'{base_file_name} Loaded')

        # pretty much must be a raw format image
        except (LibRawFatalError, LibRawNonFatalError) as e:
            if imgPath != '':
                title = 'Error opening file'
                text = 'Corrupted or incompatible image file.'
                detail_text = (f"LibRawError opening: {imgPath}\nUsually this "
                               "indicates a corrupted or incompatible image."
                               "\n{e}")
                self.userNotice(text, title, detail_text)
            return  # Pass this up to the process function for halting

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
        return patterns

    def set_bc_pattern(self):
        """ harvests all pattern strings in the listWidget_patterns, joins
        them and sends them to self.bcRead.compileRegexPattern which in turn
        sets the bcRead.rePattern attribute."""
        patterns = self.retrieve_bc_patterns()
        if patterns == '':
            # if the pattern is empty string, pass a fully permissive pattern
            patterns = '^(.*)'
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
        from libs.bcRead import bcRead
        
        self.update_CatalogNumber_Preview()
        patterns = self.retrieve_bc_patterns()
        self.fill_patterns(patterns)
        backend = self.wiz.comboBox_bcBackend.currentText()

        bcRead = bcRead(patterns, backend, parent=self.wiz)
        grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        bcResults = str(bcRead.decodeBC(grey))
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

    def test_crcDetection(self):
        """
        used to test basic crc detection
        """
        try:
            from libs.ccRead import ColorchipRead, ColorChipError
            colorchipDetect = ColorchipRead(parent=self.wiz)
            crc_type = self.wiz.comboBox_crcType.currentText()
            if crc_type == "ISA ColorGauge Nano":  # aka small crc
                original_size, reduced_img = self.scale_images_with_info(self.img)
                # determine a correction value to det! a partition_size, from box_size
                ow, oh = original_size
                nh, nw = reduced_img.shape[0:2]
                correction_val = nh / oh
                box_size = self.canvas.box_size
                corrected_box_size = int(round(box_size * correction_val,0))
                # keep a floor partition_size of 125 
                partition_size = max([125, corrected_box_size])
                print(f'using a {partition_size} partition_size')
                self.cc_position, self.cropped_cc, self.cc_crop_time = colorchipDetect.process_colorchip_small(reduced_img,
                                                                                                             original_size,
                                                                                                             stride_style='quick',
                                                                                                             high_precision=True,
                                                                                                             partition_size=partition_size)
            else:
                self.cc_position, self.cropped_cc, self.cc_crop_time = colorchipDetect.process_colorchip_big(self.img)
            
            mb = ImageDialog(self.cropped_cc)
            if mb.exec():
                formatted_result = f"Detected CRC in {round(self.cc_crop_time, 3)} seconds."
                self.wiz.pushButton_test_scaleDet.setEnabled(True)
                self.wiz.pushButton_test_whiteBalance.setEnabled(True)                              
            else:
                formatted_result = 'TEST FAILED'
                self.wiz.pushButton_test_scaleDet.setEnabled(False)
                self.wiz.pushButton_test_whiteBalance.setEnabled(False)

        except Exception as e:  # really any exception here should be handled.
            notice_title = 'Error Determining Color Reference Location'
            notice_text = 'Critical Error: Image was NOT processed!'
            detail_text = ('While attempting to determine the color chip '
                           'location the following exception was rasied:'
                           f'\n{e}')
            self.userNotice(notice_text, notice_title, detail_text)
            formatted_result = 'TEST FAILED'
            self.wiz.pushButton_test_scaleDet.setEnabled(False)
            self.wiz.pushButton_test_whiteBalance.setEnabled(False)
        
        self.wiz.label_crcDetection_results.setText(formatted_result)

        # self.wiz.intro_Page.pushButton_selectExampleImage # open example image file
        
        # path_setup_page
        # lineEdit_inputPath # input monitor folder path
        # pushButton_inputPath # input monitor browse button
        # group_saveProcessedJpg # checked = saveJPG
        # lineEdit_pathProcessedJpg # output jpg folder path
        # pushButton_pathProcessedJpg #output browse button
        # group_keepUnalteredRaw # checked = move unaltered
        # lineEdit_pathUnalteredRaw # unaltered raw folder path
        # pushButton_pathUnalteredRaw # unaltered raw browse button
        
        # bcRead_setup_page
        # group_renameByBarcode # checked = rename by barcode
        # comboBox_bcBackend # bc back end comboBox
        # lineEdit_catalogNumberPrefix # prefix
        # spinBox_catalogDigits # digits
        # comboBox_dupNamingPolicy # dup naming policy
        # label_dupPreviewDisplay # label preview
        # listWidget_patterns # pattern list
        # toolButton_addPattern # add pattern button
        # toolButton_removePattern # remove button
    
        # blurDetect_setup_page
        # checkBox_blurDetection # checked = perform blur detection
        # doubleSpinBox_blurThreshold # blur threshold
        
        # ccRead_setup_page1
        # comboBox_crcType # crcType options
        # crc_canvas_layout # canvas with attributes related to partition size: self.scaled_begin, self.scaled_end
        
        # ccRead_setup_page2
        # group_scaleDetermination # checked = perform scale Det!
        # group_verifyRotation # checked = verify / correct image rotation
        # comboBox_colorCheckerPosition # combobox with CRC position options
        # groupBox_performWhiteBalance # checked = white balance images
    
        # eqRead_setup_page
        # checkBox_lensCorrection # checked = lens corrections
        # doubleSpinBox_focalDistance # approx focal distance (cm)
    
        # metaRead_setup_page
        # plainTextEdit_collectionName
        # plainTextEdit_collectionURL
        # plainTextEdit_contactEmail
        # plainTextEdit_contactName
        # plainTextEdit_copywriteLicense

    def harvest_inputs(self):
        """
        iterates through the various 
        """
        
        #self.wiz_dict
        
        # self.wiz.path_setup_page
        # self.wiz.path_setup_page.lineEdit_inputPath.text() # input monitor folder path
        # self.wiz.path_setup_page.group_saveProcessedJpg.isChecked() # checked = saveJPG
        # self.wiz.path_setup_page.lineEdit_pathProcessedJpg.text() # output jpg folder path
        # self.wiz.path_setup_page.group_keepUnalteredRaw.isChecked() # checked = move unaltered
        # self.wiz.path_setup_page.lineEdit_pathUnalteredRaw.text() # unaltered raw folder path

        # self.wiz.bcRead_setup_page
        # self.wiz.bcRead_setup_page.group_renameByBarcode.isChecked() # checked = rename by barcode
        # self.wiz.bcRead_setup_page.comboBox_bcBackend.currentText() # bc back end comboBox
        # self.wiz.bcRead_setup_page.lineEdit_catalogNumberPrefix.text() # prefix
        # self.wiz.bcRead_setup_page.spinBox_catalogDigits.value() # digits
        # self.wiz.bcRead_setup_page.comboBox_dupNamingPolicy.currentText() # dup naming policy
        # self.wiz.bcRead_setup_page.listWidget_patterns # pattern list
        
        # self.wiz.blurDetect_setup_page
        # self.wiz.blurDetect_setup_page.checkBox_blurDetection.isChecked() # checked = perform blur detection
        # self.wiz.blurDetect_setup_page.doubleSpinBox_blurThreshold.value() # blur threshold
        
        # self.wiz.ccRead_setup_page1
        # self.wiz.ccRead_setup_page1.comboBox_crcType.currentText() # crcType options
        # self.wiz.ccRead_setup_page1.crc_canvas_layout # canvas with attributes related to partition size: self.scaled_begin, self.scaled_end
        
        # self.wiz.ccRead_setup_page2
        # self.wiz.ccRead_setup_page2.group_scaleDetermination.isChecked() # checked = perform scale Det!
        # self.wiz.ccRead_setup_page2.group_verifyRotation.isChecked() # checked = verify / correct image rotation
        # self.wiz.ccRead_setup_page2.comboBox_colorCheckerPosition.currentText() # combobox with CRC position options
        # self.wiz.ccRead_setup_page2.groupBox_performWhiteBalance.isChecked() # checked = white balance images

        # self.wiz.eqRead_setup_page
        # self.wiz.eqRead_setup_page.checkBox_lensCorrection.isChecked() # checked = lens corrections
        # self.wiz.eqRead_setup_page.doubleSpinBox_focalDistance.value() # approx focal distance (cm)
        
        # metaRead_setup_page
        # self.wiz.eqRead_setup_page.plainTextEdit_collectionName.text()
        # self.wiz.eqRead_setup_page.plainTextEdit_collectionURL.text()
        # self.wiz.eqRead_setup_page.plainTextEdit_contactEmail.text()
        # self.wiz.eqRead_setup_page.plainTextEdit_contactName.text()
        # self.wiz.eqRead_setup_page.plainTextEdit_copywriteLicense.text()

    
    def run_wizard(self):
        wiz_window = self.exec()


