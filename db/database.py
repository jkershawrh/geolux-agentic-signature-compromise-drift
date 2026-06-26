from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATA_DIR = Path(__file__).parent.parent / "data"


class Base(DeclarativeBase):
    pass


def get_database_url(db_path: str | None = None) -> str:
    if db_path == ":memory:":
        return "sqlite:///:memory:"
    path = db_path or os.environ.get("ASC_DATABASE_PATH", str(DATA_DIR / "asc.db"))
    return f"sqlite:///{path}"


def create_db_engine(db_path: str | None = None):
    url = get_database_url(db_path)
    engine = create_engine(url, echo=False)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def init_db(engine) -> None:
    from db.models import Base  # noqa: F811

    Base.metadata.create_all(engine)


def get_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
