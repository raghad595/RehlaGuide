from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings

DB_PATH = get_settings().database_path
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create any missing tables. Never drops or clears existing tables/rows."""
    from app import models  # noqa: F401  (registers tables on SQLModel.metadata)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
