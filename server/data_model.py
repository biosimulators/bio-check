from typing import *

from pydantic import BaseModel


class ArchiveUploadResponse(BaseModel):
    filename: str


