import inspect
import logging
import os

FORMAT = '%(asctime)-15s - %(funcfile)s [%(func)s] - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('allrgb')


def auto_log(message, level='info'):
    """
    Auto log with function and file name.

    :param message: log message
    :param level: (optional) log level. Default is 'info'
    :return: None
    """

    # https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    assert level in ['debug', 'info', 'warning', 'error', 'critical']

    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    eval('logger.' + level)(message, extra={'func': func.co_name,
                                            'funcfile': os.path.basename(func.co_filename)})
