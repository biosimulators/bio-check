from src import BaseModel


class EntryPoint(BaseModel):
    file_path: str


class OMEXArchive(EntryPoint):
    pass


class ModelFile(EntryPoint):
    pass


class SBMLFile(ModelFile):
    pass


class CellMLFile(ModelFile):
    pass

