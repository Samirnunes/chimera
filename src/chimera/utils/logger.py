from logging import INFO, FileHandler, StreamHandler, getLogger
from sys import stdout

status_logger = getLogger("chimera_status")
status_logger.setLevel(INFO)
status_logger.addHandler(StreamHandler(stdout))

time_logger = getLogger("chimera_time")
time_logger.setLevel(INFO)
time_logger.addHandler(FileHandler("chimera_time.log"))
