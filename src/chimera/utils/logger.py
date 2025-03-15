from logging import INFO, FileHandler, getLogger
from typing import Dict, List

_STATUS_FILE = "chimera_status.log"
_TIME_FILE = "chimera_time.log"

status_logger = getLogger("chimera_status")
status_logger.setLevel(INFO)
status_logger.addHandler(FileHandler(_STATUS_FILE))

time_logger = getLogger("chimera_time")
time_logger.setLevel(INFO)
time_logger.addHandler(FileHandler(_TIME_FILE))


def parse_times_file() -> Dict:
    times: Dict[str, Dict] = {}
    workers_times: Dict[str, List[float]] = {}
    masters_times: Dict[str, List[float]] = {}
    with open(_TIME_FILE, "r") as f:
        for line in f:
            line_split = line.split("=")
            name = line_split[0].strip()
            time = float(line_split[1].strip().removesuffix(" s"))
            endpoint = name.split(" ")[0]
            if "worker" in name:
                if endpoint not in workers_times:
                    workers_times[endpoint] = []
                workers_times[endpoint].append(time)
            elif "master" in name:
                if endpoint not in masters_times:
                    masters_times[endpoint] = []
                masters_times[endpoint].append(time)
    times["worker"] = workers_times
    times["master"] = masters_times
    return times
