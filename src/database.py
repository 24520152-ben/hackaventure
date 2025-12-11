import os
from typing import Generator
from sqlmodel import create_engine, SQLModel, Session

# Postgres configuration
HEATMAP_DATABASE = os.environ.get('DATABASE_URL')
engine = create_engine(
    url=HEATMAP_DATABASE, # type: ignore
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,
    connect_args={'sslmode': 'require'},
)

def create_database_and_table():
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator:
    with Session(engine) as session:
        yield session