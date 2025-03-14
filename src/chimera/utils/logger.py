from logging import INFO, FileHandler, getLogger

status_logger = getLogger("chimera_status")
status_logger.setLevel(INFO)
status_logger.addHandler(FileHandler("chimera_status.log"))

time_logger = getLogger("chimera_time")
time_logger.setLevel(INFO)
time_logger.addHandler(FileHandler("chimera_time.log"))
