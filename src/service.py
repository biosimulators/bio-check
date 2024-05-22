from dataclasses import dataclass
from typing import List, Dict
import abc

import numpy as np

from src import _BaseClass


@dataclass
class BiosimulationsSpeciesOutput:
    dataset_label: str
    data: np.ndarray


@dataclass
class BiosimulationsRunOutputData:
    report_path: str
    data: list[BiosimulationsSpeciesOutput]


@dataclass
class ProjectsQuery(_BaseClass):
    project_ids: List[str]
    project_data: Dict


@dataclass
class ArchiveFiles(_BaseClass):
    run_id: str
    project_name: str
    files: List[Dict]


class RestService(abc.ABC):
    @classmethod
    @abc.abstractmethod
    async def _search_source(cls, query: str) -> ProjectsQuery:
        pass

    @classmethod
    @abc.abstractmethod
    async def fetch_files(cls, query: str) -> ProjectsQuery:
        pass

    @classmethod
    @abc.abstractmethod
    async def extract_data(cls, query: str, ) -> ProjectsQuery:
        pass
