import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use DATA_DIR for persistent storage (Railway volume mount)
_data_dir = Path(os.getenv("DATA_DIR", "."))
_data_dir.mkdir(parents=True, exist_ok=True)
_db_path = _data_dir / "sql_app.db"

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
