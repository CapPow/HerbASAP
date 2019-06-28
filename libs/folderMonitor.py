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
# imports here
import string
from os import path
from shutil import move as shutil_move
import glob
#import piexif

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from PyQt5 import QtCore
import cv2


class New_Image_Emitter(QtCore.QObject):
    new_image_signal = QtCore.pyqtSignal(object)


class Event_Handler(PatternMatchingEventHandler):
    """
    Watchdog based Class to handle when new files are detected in the monitored
    folder.
    """
    def __init__(self, parent, *args, emitter=None, **kwargs):
        super(Event_Handler, self).__init__(*args, **kwargs)
        PatternMatchingEventHandler.__init__(self, *args, **kwargs)
        self._emitter = emitter
        self.parent = parent

    def on_any_event(self, event):
        event_type = event.event_type
        img_path = event.src_path
        if event_type in ['created'
                          'renamed',
                          'modified',
                          'moved']:

            if event_type in ['renamed',
                              'moved']:
                img_path = event.dest_path
            if img_path not in self.parent.existing_files:
                print(f'{event.src_path} file was modified')
                # update the list of known files
                self.parent.existing_files.append(img_path)
                self._emitter.new_image_signal.emit(img_path)
        elif event.event_type in ['deleted', 'moved']:
        # if the user removes the file from a monitored directory...
            self.parent.existing_files.remove(img_path)
        else:
            pass


class Folder_Watcher:
    def __init__(self, input_folder_path=None, raw_image_patterns=None):
        self.watch_dir = input_folder_path
        # identify currently present files        
        self.existing_files = []
        for files in raw_image_patterns:
            globPattern = path.join(input_folder_path, files)
            self.existing_files.extend(glob.glob(globPattern))
        self.emitter = New_Image_Emitter()
        self.observer = Observer()
        self.event_handler = Event_Handler(
                parent=self,
                emitter=self.emitter,
                patterns=raw_image_patterns,
                ignore_patterns=['*.tmp'],
                ignore_directories=True)

    def run(self):
        self.observer.schedule(self.event_handler, self.watch_dir)
        self.observer.start()
        #self.observer.join()


class Save_Output_Handler:
    """
    Class to handle storing the processed images, using names and formats
    detemrined by user preferences.
    """
    def __init__(self, output_map, dupNamingPolicy):
        self.output_map = output_map
        # establish self.suffix_lookup according to dupNamingPolicy
        # given an int (count of how many files have exact matching names,
        # returns an appropriate file name suffix)
        if dupNamingPolicy == 'append LOWER case letter':
            self.suffix_lookup = lambda x: {n+1: ch for n, ch in enumerate(string.ascii_lowercase)}.get(x)
        elif dupNamingPolicy == 'append UPPER case letter':
            self.suffix_lookup = lambda x: {n+1: ch for n, ch in enumerate(string.ascii_uppercase)}.get(x)
        elif dupNamingPolicy == 'append Number with underscore':
            self.suffix_lookup = lambda x: f'_{x}'
        elif dupNamingPolicy == 'OVERWRITE original image with newest':
            self.suffix_lookup = lambda x: ''
        else:
            self.suffix_lookup = False
        
        print(self.output_map)
        
    def save_output_images(self, im, orig_img_path, im_base_names, meta_data=None):
        """
        Function that saves processed images to the appropriate format and
        locations.
        :param im: Processed Image array to be saved.
        :type im: cv2 Array
        :param im_base_names: the destination file(s) base names. Usually a
        catalog number. Passed in as a list of strings.
        :type im_base_names: list
        :param meta_data: Optional, metadata dictionary organized with
        keys as destination metadata tag names, values as key value.
        :type meta_data: dict
        """
        output_map = self.output_map
        print(output_map)
        for obj, location, ext in output_map:
            if obj:
                to_rename = False
                # flag for when the file should be moved instead of cv2 written
                if not ext:
                    ext = path.splitext(orig_img_path)[-1]
                    to_rename = True
                for bc in im_base_names:
                    fileQty = len(glob.glob(f'{location}//{bc}*{ext}'))
                    if fileQty > 0:
                        new_file_suffix = self.suffix_lookup(fileQty)
                        new_file_base_name = f'{bc}{new_file_suffix}'
                        new_file_name = f'{location}//{new_file_base_name}{ext}'
                    else:
                        new_file_name = f'{location}//{bc}{ext}'
                    # TODO add in the metadata handling code
                    # piexif's transplant function may be useful if the exif is dumped in
                    # the saving process.
                    # See https://piexif.readthedocs.io/en/latest/functions.html#transplant
                    # also, add in this program's name  and version  to proper tag for processing documentation
                    # save outputs
                    if to_rename:
                        shutil_move(orig_img_path, new_file_name)
                    else:
                        cv2.imwrite(new_file_name, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

