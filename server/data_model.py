from typing import *

from pydantic import BaseModel as _BaseModel, ConfigDict


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ArchiveUploadResponse(BaseModel):
    filename: str
    content: str
    path: str


class UtcSpeciesComparison(BaseModel):
    mse: Dict
    proximity: Dict
    output_data: Dict

