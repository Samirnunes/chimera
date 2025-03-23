from logging import INFO, FileHandler, getLogger

_STATUS_FILE = "chimera_status.log"
_TIME_FILE = "chimera_time.log"

status_logger = getLogger("chimera_status")
status_logger.setLevel(INFO)
status_logger.addHandler(FileHandler(_STATUS_FILE))

time_logger = getLogger("chimera_time")
time_logger.setLevel(INFO)
time_logger.addHandler(FileHandler(_TIME_FILE))
