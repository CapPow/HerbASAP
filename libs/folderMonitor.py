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
import time
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
    def __init__(self, parent, watch_dir, emitter=None, *args, **kwargs):
        super(Event_Handler, self).__init__(*args, **kwargs)
        PatternMatchingEventHandler.__init__(self, *args, **kwargs)
        self._emitter = emitter
        self.parent = parent
        self.watch_dir = watch_dir
        self.last_item = None
        self._emitEvents = ['created', 'renamed', 'modified', 'moved']
        self._removeEvents = ['deleted', 'moved']

    def on_any_event(self, event):
        """
        attempts to handle the event based on the event type and destination
        """
        event_type = event.event_type
        img_path = event.src_path
        # be sure the destination path is the focus
        if event_type in ['renamed', 'moved']:
            img_path = event.dest_path
        # if it is leaving the self.watch_dir, set self.last_item = None
        if (path.dirname(img_path) != self.watch_dir) or event.event_type in self._removeEvents:
            # if a file is moved out out self.watch_dir:
            if img_path == self.last_item:
                self.last_item = None
            pass
        # if the event is an emit event and has not been seen emit it.
        if (event_type in self._emitEvents) and (img_path != self.last_item):
        #(img_path not in self.parent.existing_files):
            self._emitter.new_image_signal.emit(img_path)
        # remember this was the last object seen.
        self.last_item = img_path
        return


class Folder_Watcher:
    def __init__(self, input_folder_path=None, raw_image_patterns=None):
        self.watch_dir = input_folder_path
        self.emitter = New_Image_Emitter()
        self.event_handler = Event_Handler(
                parent=self,
                watch_dir=self.watch_dir,
                emitter=self.emitter,
                patterns=raw_image_patterns,
                ignore_directories=True)
        self.is_monitoring = False

    def run(self):
        if self.is_monitoring:
            pass
        else:
            self.observer = Observer(timeout=0.2)
            self.is_monitoring = True
            self.observer.schedule(self.event_handler, self.watch_dir)
            self.observer.start()

    def stop(self):
        while (self.observer.event_queue.unfinished_tasks != 0):
                time.sleep(1)
        self.observer.stop()
        self.observer.join(timeout=.1)
        self.is_monitoring = False
