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


# current thread types
# bc_worker
# blur_worker
# eq_worker
from PyQt5.QtCore import (QObject, QRunnable, pyqtSignal, pyqtSlot, QThread,
                          QEventLoop)
from PyQt5.QtWidgets import QApplication
import traceback
import sys


class Job:
    """

    """
    def __init__(self, job_name, job_data, job_function):
        self.job_name = job_name
        self.job_data = job_data
        self.job_function = job_function


class BCWorkerData:
    """
        object that represents the data needed to run a bc_worker
    """
    def __init__(self, grey_image):
        self.grey_image = grey_image


class BlurWorkerData:
    """
        object that represents the data needed to run a blur_worker
    """
    def __init__(self, grey_image, blur_threshold, return_details):
        self.grey_image = grey_image
        self.blur_threshold = blur_threshold
        self.return_details = return_details


class EQWorkerData:
    """
        object that represents the data needed to run a eq_worker
    """
    def __init__(self, im, img_path, m_distance):
        self.im = im
        self.img_path = img_path
        self.mDistance = m_distance

class SaveWorkerData:
    """
        object that represents the data needed to run a save_worker
    """
    def __init__(self, new_file_name, im):
        self.im = im
        self.new_file_name = new_file_name

class BossSignalData:
    """

    """
    def __init__(self, is_worker_signal, signal_data):
        self.isWorkerSignal = is_worker_signal
        self.signal_data = signal_data  # this can be a message or an instance of WorkerSignalData


class BossSignals(QObject):
    """

    """
    boss_started = pyqtSignal(BossSignalData)
    boss_closed = pyqtSignal(BossSignalData)
    job_started = pyqtSignal(BossSignalData)
    job_finished = pyqtSignal(BossSignalData)
    job_error = pyqtSignal(BossSignalData)
    job_result = pyqtSignal(BossSignalData)
    job_progress = pyqtSignal(BossSignalData)
    clear_to_save = pyqtSignal()


class Boss(QThread):
    """

    """
    def __init__(self, thread_pool):
        super(Boss, self).__init__()
        self.__thread_pool = thread_pool
        self.__job_queue = []
        self.__should_run = True
        self.__is_bc_worker_running = False
        self.__is_blur_worker_running = False
        self.__is_eq_worker_running = False
        # self.args = args
        # self.kwargs = kwargs
        self.signals = BossSignals()

    def request_job(self, job):
        """

        """
        if job.job_name == 'bc_worker' and job.job_data is not None and job.job_function is not None:
            self.__is_bc_worker_running = True
            bc_worker = Worker(job.job_function, job.job_data.grey_image)
            bc_worker.set_worker_name('bc_worker')
            bc_worker.signals.started.connect(self.worker_started_handler)
            bc_worker.signals.error.connect(self.worker_error_handler)
            bc_worker.signals.result.connect(self.worker_result_handler)
            bc_worker.signals.finished.connect(self.worker_finished_handler)
            self.__thread_pool.start(bc_worker)  # start bc_worker thread
        elif job.job_name == 'blur_worker' and job.job_data is not None and job.job_function is not None:
            self.__is_blur_worker_running = True
            blur_worker = Worker(job.job_function, job.job_data.grey_image, job.job_data.blur_threshold, job.job_data.return_details)
            blur_worker.set_worker_name('blur_worker')
            blur_worker.signals.started.connect(self.worker_started_handler)
            blur_worker.signals.error.connect(self.worker_error_handler)
            blur_worker.signals.result.connect(self.worker_result_handler)
            blur_worker.signals.finished.connect(self.worker_finished_handler)
            self.__thread_pool.start(blur_worker)  # start blur_worker thread
        elif job.job_name == 'eq_worker' and job.job_data is not None and job.job_function is not None:
            self.__is_eq_worker_running = True
            eq_worker = Worker(job.job_function, job.job_data.im, job.job_data.img_path, job.job_data.mDistance)
            eq_worker.set_worker_name('eq_worker')
            eq_worker.signals.started.connect(self.worker_started_handler)
            eq_worker.signals.error.connect(self.worker_error_handler)
            eq_worker.signals.result.connect(self.worker_result_handler)
            eq_worker.signals.finished.connect(self.worker_finished_handler)
            self.__thread_pool.start(eq_worker)  # start eq_worker thread
            # in this case, job_data should be None
        elif job.job_name == 'save_worker' and job.job_data is not None and job.job_function is not None:
            save_worker = Worker(job.job_function, job.job_data.new_file_name, job.job_data.im)
            save_worker.set_worker_name('save_worker')
            save_worker.signals.started.connect(self.worker_started_handler)
            save_worker.signals.error.connect(self.worker_error_handler)
            save_worker.signals.result.connect(self.worker_result_handler)
            save_worker.signals.finished.connect(self.worker_finished_handler)
            self.__thread_pool.start(save_worker)  # start eq_worker thread
        else:
            print('no my son')

    def program_closing(self, prgm_is_closing):
        """

        """
        self.__should_run = prgm_is_closing
        boss_closed_data = BossSignalData(False, 'boss thread is stopping...')
        self.signals.boss_closed.emit(boss_closed_data)

    def worker_result_handler(self, worker_signal_data):
        """
            single handler for all result signals to be routed through
        """
        job_result_data = BossSignalData(True, worker_signal_data)
        self.signals.job_result.emit(job_result_data)

    def worker_started_handler(self, worker_signal_data):
        """
            single handler for all started signals to be routed through
        """
        job_started_data = BossSignalData(True, worker_signal_data)
        self.signals.job_started.emit(job_started_data)
        QApplication.processEvents()

    def worker_finished_handler(self, worker_signal_data):
        """
            single handler for all finished signals to be routed through
        """
        job_finished_data = BossSignalData(True, worker_signal_data)
        if worker_signal_data.worker_name == 'bc_worker':
            self.__is_bc_worker_running = False
        elif worker_signal_data.worker_name == 'blur_worker':
            self.__is_blur_worker_running = False
        elif worker_signal_data.worker_name == 'eq_worker':
            self.__is_eq_worker_running = False
        self.signals.job_finished.emit(job_finished_data)
        # check if any other workers in operation. If not, 'get the lights'
        if not any([self.__is_eq_worker_running,
                    self.__is_blur_worker_running,
                    self.__is_bc_worker_running]):
            self.signals.clear_to_save.emit()

    def worker_error_handler(self, worker_signal_data):
        """
            single handler for all finished signals to be routed through
        """
        job_error_data = BossSignalData(True, worker_signal_data)
        self.signals.job_error.emit(job_error_data)


class WorkerSignalData:
    """

    """
    def __init__(self, worker_name, signal_data):
        self.worker_name = worker_name
        self.signal_data = signal_data


class WorkerErrorData:
    """

    """
    def __init__(self, caught_exception, exctype, value, format_exc):
        self.exception = caught_exception
        self.exctype = exctype
        self.value = value
        self.format_exc = format_exc


class WorkerSignals(QObject):
    """

    """
    started = pyqtSignal(WorkerSignalData)
    finished = pyqtSignal(WorkerSignalData)
    error = pyqtSignal(WorkerSignalData)
    result = pyqtSignal(WorkerSignalData)
    progress = pyqtSignal(WorkerSignalData)
    new_image_signal = pyqtSignal(WorkerSignalData)


class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    see: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.worker_name = ''
        # https://www.riverbankcomputing.com/static/Docs/PyQt4/qrunnable.html#autoDelete
        self.setAutoDelete(True)

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    def set_worker_name(self, worker_name):
        self.worker_name = worker_name

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
            started_data = WorkerSignalData(self.worker_name, f'{self.worker_name}: started')
            self.signals.started.emit(started_data)
        except Exception as e:
            # traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            worker_error_data = WorkerErrorData(e, exctype, value, traceback.format_exc())
            error_data = WorkerSignalData(self.worker_name, worker_error_data)
            self.signals.error.emit(error_data)
            # NOTE This makes the program more fragile, should be removed as we get closer to release
            # added for now because useful tracebacks were being obfuscated
            raise
        else:
            result_data = WorkerSignalData(self.worker_name, result)
            self.signals.result.emit(result_data)  # Return the result of the processing
        finally:
            finished_data = WorkerSignalData(self.worker_name, f'{self.worker_name}: finished')
            self.signals.finished.emit(finished_data)  # Done

