# current thread types
# bc_worker
# blur_worker
# eq_worker
# save_worker
from PyQt5.QtCore import (QObject, QRunnable, pyqtSignal, pyqtSlot, QThread)
import traceback
import sys
from libs.bcRead import bcRead
from libs.eqRead import eqRead
from libs.blurDetect import blurDetect


class Job:
    def __init__(self, job_name, job_data, job_function):
        self.job_name = job_name
        self.job_data = job_data
        self.job_function = job_function


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    see: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    new_image_signal = pyqtSignal(object)


class BossSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    see: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/

    Supported signals are:

    job_started
        bool indicating whether or not a job was started and the Worker name

    job_finished
        indicates job finished, returns results from Worker, and the Worker name

    job_error
        `tuple` (exctype, value, traceback.format_exc() ) and the Worker name

    job_result
        `object` data returned from processing, anything and the Worker name

    job_progress
        `int` indicating % progress and the Worker name

    """
    boss_started = pyqtSignal()
    boss_closed = pyqtSignal()
    job_started = pyqtSignal()
    job_finished = pyqtSignal()
    job_error = pyqtSignal(tuple)
    job_result = pyqtSignal(object)
    job_progress = pyqtSignal(int)


class BCWorkerData:
    """

    """
    def __init__(self, grey_image):
        self.grey_image = grey_image


class BlurWorkerData:
    """

    """
    def __init__(self, grey_image, blur_threshold):
        self.grey_image = grey_image
        self.blur_threshold = blur_threshold


class EQWorkerData:
    """

    """
    def __init__(self, im, img_path, m_distance):
        self.im = im
        self.img_path = img_path
        self.mDistance = m_distance


class Boss(QThread):
    """

    """
    def __init__(self):
        super(Boss, self).__init__()
        self.__sleep_time = 2  # measured in seconds
        self.__job_queue = []
        self.__should_run = True
        self.__is_bc_worker_running = False
        self.__is_blur_worker_running = False
        self.__is_eq_worker_running = False
        # self.args = args
        # self.kwargs = kwargs
        self.signals = BossSignals()

    def request_job(self, job):
        self.__job_queue.append(job)  # append job to queue (now last element)

    def run(self):
        print('started __boss_function')
        while self.__should_run:
            if len(self.__job_queue) > 0:
                print('job_queue len is: ' + str(len(self.__job_queue)))
                job = self.job_queue.pop(0)  # pop first element
                """
                    if the job is save_worker, then we will append it to the end of the queue
                    otherwise, the job is OK to run
                    1. if job is save_worker, we want to wait to schedule it until other threads complete
                        a. all threads complete == all __is_$_running are False
                    ! - ensure this doesn't overfill memory, not sure how Python will deal with popping / appending
                        big elements onto the queue. if the job data is re-created each time we may have memory
                        issues
                """
                if job.job_name == 'save_worker':
                    if (not self.__is_eq_worker_running and not self.__is_blur_worker_running
                                                        and not self.__is_bc_worker_running):
                        print('running save_worker job')
                        self.spawn_thread(job)
                    else:
                        self.__job_queue.append(job)
                else:
                    print('running job' + job.job_name)
                    self.spawn_thread(job)
            else:
                pass
            self.sleep(self.__sleep_time)  # https://doc.qt.io/qt-5/qthread.html#sleep
        print('__boss_function exiting')

    def __spawn_thread(self, job):
        if job.job_name == 'bc_worker' and job.job_data is not None and job.job_function is not None:
            self.__is_bc_worker_running = True
            bc_worker = Worker(job.job_function, job.job_data.grey)
            bc_worker.signals.result.connect(self.handle_bc_result)
            bc_worker.signals.finished.connect(self.alert_bc_finished)
            self.threadPool.start(bc_worker)  # start bc_worker thread
        elif job.job_name == 'blur_worker' and job.job_data is not None and job.job_function is not None:
            self.__is_blur_worker_running = True
            blur_worker = Worker(job.job_function, job.job_data.grey_image, job.job_data.blur_threshold)
            blur_worker.signals.result.connect(self.handle_blur_result)
            blur_worker.signals.finished.connect(self.alert_blur_finished)
            self.threadPool.start(blur_worker)  # start blur_worker thread
        elif job.job_name == 'eq_worker' and job.job_data is not None and job.job_function is not None:
            self.__is_eq_worker_running = True
            eq_worker = Worker(job.job_function, job.job_data.im, job.job_data.img_path, job.job_data.mDistance)
            eq_worker.signals.result.connect(self.handle_eq_result)
            eq_worker.signals.finished.connect(self.alert_eq_finished)
            self.threadPool.start(eq_worker)  # start eq_worker thread
        # in this case, job_data should be None
        elif job.job_name == 'save_worker' and job.job_data is None and job.job_function is not None:
            # wait on bcWorker
            save_worker = Worker(job.job_function)
            # save_worker.signals.finished.connect(job.job_data[1])
            self.threadPool.start(save_worker)  # start save_worker thread
        else:
            return 'no my son'

    def program_closing(self, prgm_is_closing):
        print('boss thread closing')
        self.__should_run = prgm_is_closing


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
        # https://www.riverbankcomputing.com/static/Docs/PyQt4/qrunnable.html#autoDelete
        self.setAutoDelete(True)

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

