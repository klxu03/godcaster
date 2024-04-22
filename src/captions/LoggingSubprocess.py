import argparse
import os
import sys
import threading
import subprocess
import time
import select
import logging
import signal
from queue import Queue

RESET = "\x1b[0m"
YELLOW = "\x1b[1;33m"
WHITE = "\x1b[1;37m"
GRAY = "\x1b[0;37m"
CYAN = "\x1b[0;36m"
LIGHT_CYAN = "\x1b[1;36m"
LIGHT_RED = "\x1b[1;31m"


def get_format(color):
    """
    Default formatting for loggers used by LoggingSubprocess.

    The timestamp is excluded because its rarely useful during development,
    and many of the processes' output already includes one.
    """
    return (  
            ' %(asctime)s' 
            ' %(name)-10s'
            ' %(levelname)-6s'
            ' %(message)s'
            )


class MaxLevelFilter(object):
    """
    This filter for loggers / log handlers sets a maximum logging level.

    The logging levels on loggers / log handlers only take a minimum level.
    This acts as a complement to logging levels and creates an inclusive upper
    bound on the logging levels.
    """

    def __init__(self, level):
        self.__level = level

    def filter(self, record):
        return record.levelno <= self.__level


def get_logger(name, logging_level=logging.DEBUG, color=YELLOW):
    """
    returns a standard logging.Logger with useful default behavior.

    All criticals, errors, exceptions, and warnings will be displayed on the
    console in red. Everything else down to the "logging_level" argument passed
    will show up in the console in the "color" argument passed.
    """
    time_format = '%b %d %H:%M:%S'

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        get_format(color), datefmt=time_format))
    log_handler.addFilter(MaxLevelFilter(logging.INFO))
    logger.addHandler(log_handler)

    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        get_format(LIGHT_RED), datefmt=time_format))
    log_handler.setLevel(logging.WARNING)
    logger.addHandler(log_handler)
    return logger


class LoggingSubprocess(threading.Thread):
    """
    Creates a subprocess as per subprocess.Popen, but with built-in logging.

    The subprocess is spawned from a separate thread that monitors stdout and
    stderr, and feeds their output into a logger.
    """

    def __init__(self, command, shell=True, working_directory=None,
                 stdout_log_level=logging.INFO,
                 stderr_log_level=logging.ERROR):
        """
        :param command: A list wherein the first element is a command and
            subsequent elements are its arguments. (same as subprocess.Popen)
        :param shell:  execute the command as on the shell?
            (same as subprocess.Popen)
        :param logger: A logging.Logger which will receive output from stderr
            and stdout of the subprocess.
        :param working_directory: The current working directory to use for the
            subprocess. (same as cwd for subprocess.Popen)
        :param stdout_log_level: stdout messages will be logged at this level.
        :param stderr_log_level: stderr messages will be logged at this level.
        """
        self.__stdout = stdout_log_level
        self.__stderr = stderr_log_level

        self._command = command
        self._shell = shell
        threading.Thread.__init__(self, name=command)
        self.setDaemon(False)
        self.subprocess = None
        self._stdin_queue = Queue()
        self._cwd = working_directory

    def input(self, message):
        """Safely pass a message to the subprocess' stdin. Non-blocking."""
        self._stdin_queue.put_nowait(message)

    def run(self):
        """Don't call this. It will run when you call .start()"""
        try:
            my_env = os.environ.copy()
            self.subprocess = subprocess.Popen(self._command,
                                               shell=self._shell,
                                               stdin=subprocess.PIPE,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               cwd=self._cwd,
                                               bufsize=0,
                                               preexec_fn=os.setsid,
                                               env=my_env
                                               )
            self.logger = get_logger(f"Inference PID {self.subprocess.pid}")
            self.logger.debug('starting...')

            log_level = {self.subprocess.stdout: self.__stdout,
                         self.subprocess.stderr: self.__stderr}

            def check_io():
                ready_to_read, ready_to_write, _ = \
                    select.select([self.subprocess.stdout,
                                   self.subprocess.stderr],
                                  [self.subprocess.stdin], [], .1)
                for io in ready_to_read:
                    line = io.readline().decode()
                    if len(line) > 2:
                        self.logger.log(level=log_level[io], msg=line)
                for io in ready_to_write:
                    if not self._stdin_queue.empty():
                        msg = self._stdin_queue.get_nowait()
                        io.write(msg)
                        # self.logger.debug('put "'+msg+ '" on stdin')

            # keep checking stdout/stderr until the process exits
            while self.subprocess.poll() is None:
                check_io()
                # self.logger.debug('still running')

            for x in range(20):  # log anything left in the pipe
                check_io()

            self.logger.debug('process finished with code ' +
                              str(self.subprocess.poll()))

        except Exception as exc:
            self.logger.exception(exc)
            if self.subprocess is not None:
                self.subprocess.kill()

    def stop(self, timeout=2.0):
        """Blocking. This will kill the subprocess but not the thread."""
        if self.subprocess is None:
            # raise Exception(
            self.logger.error(
                "Trying to stop a subprocess that was never created")
            return
        if self.subprocess.poll():
            return
        os.killpg(self.subprocess.pid, signal.SIGTERM)
        self.logger.debug('terminating process')
        timer = 0
        while self.subprocess.poll() is None and timer < timeout:
            time.sleep(.1)
            timer += .1

        if self.subprocess.poll() is None:
            os.killpg(self.subprocess.pid, signal.SIGKILL)
            self.logger.debug('killing process')

    def join(self, timeout=None):
        """
        Blocking. Works like Thread.join().

        This will kill the subprocess if given a timeout.
        """
        if timeout is not None:
            self.stop(timeout - .2)
        elif self.subprocess is not None:
            self.subprocess.wait()
        threading.Thread.join(self, timeout=.2)
