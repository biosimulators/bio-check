# -- worker models -- #
from dataclasses import dataclass
from typing import *

import numpy as np

from worker.shared_worker import BaseClass


@dataclass
class BiosimulationsReportOutput(BaseClass):
    dataset_label: str
    data: np.ndarray


@dataclass
class BiosimulationsRunOutputData(BaseClass):
    report_path: str
    data: List[BiosimulationsReportOutput]


