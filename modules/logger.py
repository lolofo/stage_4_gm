import logging
import sys
import os
from os import path

# Colored log:https://medium.com/geekculture/view-your-logs-in-colour-c49f7b90347b
from tqdm import tqdm

LOG_CONSOLE = 'console_'
LOG_FILE = 'file'

# exporting variable
log = logging.getLogger(LOG_CONSOLE)


class ColorFormatter(logging.Formatter):
    # Inspired: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

    # codes
    # Color code:https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
    _COLOR = dict(
        bright_blue='\u001b[34;1m', bright_green='\u001b[32;1m', grey='\x1b[38;21m', yellow='\x1b[33;21m',
        red="\u001b[31m",
        bold_red='\x1b[31;1m', yellow2='\x1B[33m', blue='\x1b[34m'
    )
    _STYLE = dict(bold='\u001b[1m', underline='\u001b[4m', reversed='\u001b[7m')
    _RESET = '\x1b[0m'

    COLOR_MAP = {
        logging.DEBUG: _COLOR['bright_green'],
        logging.INFO: _COLOR['blue'],
        logging.WARNING: _COLOR['yellow2'],
        logging.ERROR: _COLOR['red'],
        logging.CRITICAL: _COLOR['bold_red']
    }

    def __init__(self, color=True):
        self.verbose = '%(levelname)8s'
        self.location = '%(filename)s:%(funcName)s:%(lineno)d'
        self.msg = '%(message)s'
        self.FORMAT = f'%(asctime)s | {self.verbose} {self.location} {self.msg}'
        self._colorize = color

    def _bold(self, text):
        return '\x1b[1m' + text + self.reset

    def _italic(self, text):
        return

    def _color(self, text, level_code):
        return self.COLOR_MAP[level_code] + text + self._RESET

    def format(self, record):
        # replace only the level

        log_fmt = self.FORMAT
        if self._colorize:
            # treat verbose level
            log_fmt = log_fmt.replace(self.verbose, self._color(self.verbose, record.levelno))
            log_fmt = log_fmt.replace(self.msg, self._color(self.msg, record.levelno))

            # treat filename
            log_fmt = log_fmt.replace(self.location,
                                      f"{self._STYLE['bold']} {self._STYLE['underline']} {self.location} {self._RESET}")

        formatter = logging.Formatter(log_fmt, datefmt='%d-%m-%Y %H:%M:%S')
        return formatter.format(record)


def init_logging(color: bool = True,
                 oar_id=None,
                 cache_path: str = '_out'):
    global log
    log.propagate = False
    log.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(ColorFormatter(color=color))

    if oar_id is not None:
        os.makedirs(path.join(cache_path, 'logger'), exist_ok=True)
        log_path = path.join(cache_path, 'logger', f'{oar_id}.{LOG_CONSOLE}.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ColorFormatter(color=color))
        if not color:
            file_handler.setFormatter(
                logging.Formatter(f'%(asctime)s |%(levelname)8s| (%(filename)s:%(funcName)s:%(lineno)d) %(message)s')
            )
        log.addHandler(file_handler)
    else :
        log.addHandler(sh)
