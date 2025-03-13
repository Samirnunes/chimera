from logging import INFO, StreamHandler, getLogger
from sys import stdout

logger = getLogger("chimera")
logger.setLevel(INFO)
logger.addHandler(StreamHandler(stdout))
