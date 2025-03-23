from typing import Dict, List


def parse_times_file(filepath: str) -> Dict:
    times: Dict[str, Dict] = {}
    workers_times: Dict[str, List[float]] = {}
    masters_times: Dict[str, List[float]] = {}
    with open(filepath, "r") as f:
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
