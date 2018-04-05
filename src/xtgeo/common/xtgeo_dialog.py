# -*- coding: utf-8 -*-
"""Module for basic XTGeo dialog"""

# =============================================================================
# Message and dialog handler in xtgeo. It works together the logging module,
# But, also I need stuff to work together with existing
# Perl and C libraries...
#
# How it should works:
# Enviroment variable XTG_VERBOSE_LEVEL will steer the output from lowelevel
# C routines; normally they are quiet
# XTG_VERBOSE_LEVEL is undefined: xtg.say works to screen
# XTG_VERBOSE_LEVEL > 1 starts to print C messages
# XTG_VERBOSE_LEVEL < 0 skip also xtg.say
#
# XTG_LOGGING_LEVEL is for Python logging (string, as INFO)
# XTG_LOGGING_FORMAT is for Python logging (number, 0 ,1, 2, ...)
#
# The system here is:
# syslevel is the actual level when code is executed:
#
# -1: quiet dialog, no warnings only errors and critical
# 0 : quiet dialog, only warnings and errors will be displayed
# JRIV
# =============================================================================


import os
import sys
import inspect
import logging
import xtgeo
import cxtgeo
import cxtgeo.cxtgeo as _cxtgeo
import timeit

UNDEF = _cxtgeo.UNDEF
UNDEF_LIMIT = _cxtgeo.UNDEF_LIMIT
VERYLARGENEGATIVE = _cxtgeo.VERYLARGENEGATIVE
VERYLARGEPOSITIVE = _cxtgeo.VERYLARGEPOSITIVE


class _BColors:
    # local class for ANSI term color commands
    # bgcolors:
    # 40=black, 41=red, 42=green, 43=yellow, 44=blue, 45=pink, 46 cyan

    HEADER = '\033[1;96m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARN = '\033[93;43m'
    ERROR = '\033[93;41m'
    CRITICAL = '\033[1;91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class XTGeoDialog(object):
    """System for handling dialogs and messages in XTGeo.

    This module cooperates with Python logging module.

    """

    def __init__(self):

        # a number, for C routines
        envsyslevel = os.environ.get('XTG_VERBOSE_LEVEL')

        # a string, for Python logging:
        logginglevel = os.environ.get('XTG_LOGGING_LEVEL')

        # a number, for format, 1 is simple, 2 is more info etc
        loggingformat = os.environ.get('XTG_LOGGING_FORMAT')

        if envsyslevel is None:
            self._syslevel = 0
        else:
            self._syslevel = int(envsyslevel)

        if logginglevel is None:
            self._logginglevel = 'CRITICAL'
        else:
            self._logginglevel = str(logginglevel)

        if loggingformat is None:
            self._lformatlevel = 1
        else:
            self._lformatlevel = int(loggingformat)

    @staticmethod
    def UNDEF():
        return UNDEF

    @staticmethod
    def UNDEF_LIMIT():
        return UNDEF_LIMIT

    @property
    def syslevel(self):
        return self._syslevel

    # for backward compatibility (to be phased out)
    def get_syslevel(self):
        return self._syslevel

    @property
    def logginglevel(self):
        """Will return a logging level property, e.g. logging.CRITICAL"""
        ll = logging.CRITICAL
        if self._logginglevel == 'INFO':
            ll = logging.INFO
        elif self._logginglevel == 'WARNING':
            ll = logging.WARNING
        elif self._logginglevel == 'DEBUG':
            ll = logging.DEBUG

        return ll

    @property
    def loggingformatlevel(self):
        return self._lformatlevel

    @property
    def loggingformat(self):
        """Returns the format string to be used in logging"""

        if self._lformatlevel <= 1:
            self._lformat = '%(name)44s %(funcName)44s '\
                + '%(levelname)8s: \t%(message)s'
        else:
            self._lformat = '%(asctime)s Line: %(lineno)4d %(name)44s '\
                + '[%(funcName)40s()]'\
                + '%(levelname)8s:'\
                + '\t%(message)s'

        return self._lformat

    @syslevel.setter
    def syslevel(self, mylevel):
        if mylevel >= 0 and mylevel < 5:
            self._syslevel = mylevel
        else:
            print('Invalid range for syslevel')

        envsyslevel = os.environ.get('XTG_VERBOSE_LEVEL')

        if envsyslevel is None:
            pass
        else:
            # print('Logging overridden by XTG_VERBOSE_LEVEL = {}'
            #       .format(envsyslevel))
            self._syslevel = int(envsyslevel)

    @staticmethod
    def print_xtgeo_header(appname, appversion):
        """Prints a XTGeo banner for an app to STDOUT."""

        cur_version = 'Python ' + str(sys.version_info[0]) + '.'
        cur_version += str(sys.version_info[1]) + '.' \
            + str(sys.version_info[2])

        app = appname + ' (version ' + str(appversion) + ')'
        print('')
        print(_BColors.HEADER)
        print('#' * 79)
        print('#{}#'.format(app.center(77)))
        print('#' * 79)
        ver = 'XTGeo4Python version ' + xtgeo.__version__
        ver = ver + ' (CXTGeo v. ' + cxtgeo.__version__ + ')'
        print('#{}#'.format(ver.center(77)))
        print('#{}#'.format(cur_version.center(77)))
        print('#' * 79)
        print(_BColors.ENDC)
        print('')

    def basiclogger(self, name):
        """Initiate the logger by some default settings."""

        format = self.loggingformat
        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(self.logginglevel)  # root logger!
        logging.captureWarnings(True)

        return logging.getLogger(name)

    @staticmethod
    def functionlogger(name):
        """Get the logger for functions (not top level)."""

        logger = logging.getLogger(name)
        logger.addHandler(logging.NullHandler())
        return logger

    def testsetup(self):
        """Basic setup for XTGeo testing (private; only relevant for tests)"""

        path = 'TMP'
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

        try:
            bigtest = int(os.environ['XTG_BIGTEST'])
            bigtest = True
            print('<< Big tests enabled by XTG_BIGTEST env >>')
        except Exception:
            bigtest = False
            print('<< Big tests disabled as XTG_BIGTEST not set >>')

        testpath = '../xtgeo-testdata'
        try:
            testpath = str(os.environ['XTG_BIGTEST'])
            print('<< Test data path by XTG_TESTDATA env: >>'.format(testpath))
        except Exception:
            print('<< No env XTG_TESTDATA - test data path default: {} >>'
                  .format(testpath))

        self.test_env = True
        self.tmpdir = path
        self.bigtest = bigtest
        self.testpath = testpath

        return True

    @staticmethod
    def timer(*args):
        """Without args; return the time, with a time as arg return the
        difference.
        """
        time1 = timeit.default_timer()

        if len(args) > 0:
            return time1 - args[0]
        else:
            return time1

    def insane(self, string):
        level = 4
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def trace(self, string):
        level = 3
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def debug(self, string):
        level = 2
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def speak(self, string):
        level = 1
        idx = 1

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    info = speak

    def say(self, string):
        level = -5
        idx = 3

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def warn(self, string):
        level = 0
        idx = 6

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    warning = warn

    def error(self, string):
        level = -8
        idx = 8

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def critical(self, string):
        level = -9
        idx = 9

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)
        raise SystemExit('STOP!')

    def get_callerinfo(self, caller, frame):
        the_class = self._get_class_from_frame(frame)

        # just keep the last class element
        x = str(the_class)
        x = x.split('.')
        the_class = x[-1]

        self._caller = caller
        self._callclass = the_class

        return (self._caller, self._callclass)

# =============================================================================
# Private routines
# =============================================================================

    def _get_class_from_frame(self, fr):
        args, _, _, value_dict = inspect.getargvalues(fr)
        # we check the first parameter for the frame function is
        # named 'self'
        if len(args) and args[0] == 'self':
            instance = value_dict.get('self', None)
            if instance:
                # return its class
                return getattr(instance, '__class__', None)
        # return None otherwise
        return None

    def _output(self, idx, level, string):

        prefix = ''
        endfix = ''

        if idx == 0:
            prefix = '++'
        elif idx == 1:
            prefix = '**'
        elif idx == 3:
            prefix = '>>'
        elif idx == 6:
            prefix = _BColors.WARN + '##'
            endfix = _BColors.ENDC
        elif idx == 8:
            prefix = _BColors.ERROR + '!#'
            endfix = _BColors.ENDC
        elif idx == 9:
            prefix = _BColors.CRITICAL + '!!'
            endfix = _BColors.ENDC

        prompt = False
        if level <= self._syslevel:
            prompt = True

        if prompt:
            if self._syslevel <= 1:
                print('{} {}{}'.format(prefix, string, endfix))
            else:
                ulevel = str(level)
                if (level == -5):
                    ulevel = 'M'
                if (level == -8):
                    ulevel = 'E'
                if (level == -9):
                    ulevel = 'W'
                print('{0} <{1}> [{2:23s}->{3:>33s}] {4}{5}'
                      .format(prefix, ulevel, self._callclass,
                              self._caller, string, endfix))


# =============================================================================
# MAIN, for initial testing. Run from current directory
# =============================================================================
def main():

    xtg = XTGeoDialog()

    xtg.speak('Level 1 text')
    xtg.debug('Level 2 text debug should not show')

    xtx = XTGeoDialog()

    # can use both class and instacne her (since this is a classmethod)
    print('Syslevel (instance) is {}'.format(xtx.syslevel))

    xtx.speak('Level 1 speak text')
    xtx.info('Level 1 info text')

    mynumber = 2233.2293939
    xtx.say('My number is {0:6.2f}'.format(mynumber))

    xtg.syslevel(2)

    print('Syslevel is ' + str(xtg.syslevel))

    xtg.debug('Level 2 (debug) text should show now')
    xtg.say('Say hello ...')

    xtg.error('Errors are always shown as long as level > -9')


if __name__ == '__main__':
    main()
