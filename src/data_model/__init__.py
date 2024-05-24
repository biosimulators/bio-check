from dataclasses import dataclass, asdict
from pydantic import BaseModel as _BaseModel, ConfigDict


@dataclass(frozen=True)
class _BaseClass:
    def to_dict(self):
        return asdict(self)


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
