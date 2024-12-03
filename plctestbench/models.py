from dataclasses import dataclass, asdict
from typing import Any
from enum import Enum


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


@dataclass
class TestbenchSettings:

    root_folder: str = None

    db_platform: DBPlatform = DBPlatform.TINYDB

    db_port: str = None

    db_ip: str = None

    db_username: str = None

    db_password: str = None
