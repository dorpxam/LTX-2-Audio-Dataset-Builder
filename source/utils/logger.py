import logging
import re
import pathlib
from typing import Optional
from .console import Console

class MXPFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        self.console = Console()

    def format(self, record: logging.LogRecord) -> str:
        log_line = super().format(record)
        return self.console.string(f"[a]{log_line}[ ]")

class FileCleanerFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        clean_msg = re.sub(r'\[[a-z ]\]', '', msg)
        return clean_msg

def get_logger(name: Optional[str] = None):
    return logging.getLogger("MXP")

def setup_mxp_logging(audio_path: Optional[str] = None, level: int = logging.INFO):
    logger = logging.getLogger("MXP")
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(MXPFormatter())
    logger.addHandler(sh)

    if audio_path:
        log_filename = pathlib.Path(audio_path).with_suffix('.log')
        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setFormatter(FileCleanerFormatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)

    return logger