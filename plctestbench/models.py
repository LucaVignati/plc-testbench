from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNNING = "RUNNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class DBPlatform(Enum):
    MONGODB = "MONGODB"
    TINYDB = "TINYDB"


@dataclass
class Run:

    _id: str = None

    workers: list = None

    nodes: list = None

    status: RunStatus = None

    creator: str = None

    created: str = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Run":
        return cls(
            _id=data.get("_id"),
            workers=data.get("workers"),
            nodes=data.get("nodes"),
            status=RunStatus(data.get("status")) if data.get("status") else None,
            creator=data.get("creator"),
            created=data.get("created"),
        )


@dataclass
class TestbenchConfiguration:

    root_folder: str = None

    db_platform: DBPlatform = DBPlatform.TINYDB

    db_port: str = None

    db_ip: str = None

    db_username: str = None

    db_password: str = None


@dataclass
class User:

    email: str

    first_name: str

    last_name: str

    locale: str

    image_url: str

    @classmethod
    def get_default_user(self) -> "User":
        return User("default@default.com", "default", "default", "it_IT", "")
