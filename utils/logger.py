import logging
import os
import time
from typing import List, Dict
import json


class Logger:
    def __init__(self, workspace: str, level: str):
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self._workspace: str = workspace

        self._level: int = logging.INFO if level == "INFO" else logging.DEBUG
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        self._logger = logging.getLogger("message")
        self._logger.setLevel(self._level)

        file_handler = logging.FileHandler(os.path.join(self._workspace, "ouput.log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self._level)
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(self._level)
        self._logger.addHandler(stream_handler)
        self._time: float = time.time()
        self._save_path: str = os.path.join(self._workspace, "metric.txt")

    def report_time(self, name: str):
        current: float = time.time()
        self._write(name + " - time: {}".format(current - self._time))

    def report_metric(self, name: str, metric: Dict):
        self.report_delimiter()
        self.report_time(name)
        keys: List = list(metric.keys())
        for key in keys:
            self._write("\t- {}: {}".format(key, metric[key]))
        self.report_delimiter()
        self.report_newline()

    def write(self, metric: Dict):
        with open(self._save_path, 'a', encoding='utf=8') as f:
            f.write(json.dumps(metric))
            f.write("\n")

    def report_delimiter(self):
        self._write("-" * 33)

    def report_newline(self):
        self._write("")

    def _write(self, message: str):
        if self._level == logging.INFO:
            self._logger.info(message)
            return
        self._logger.debug(message)
