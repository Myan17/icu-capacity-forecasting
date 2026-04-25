"""Test configuration — patch DB to SQLite so tests run without Postgres."""
from __future__ import annotations

import os

# Set DATABASE_URL before any app module is imported.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.main import app
from fastapi.testclient import TestClient


SQLALCHEMY_DATABASE_URL = "sqlite://"


@pytest.fixture(scope="session", autouse=True)
def override_db():
    """Replace the Postgres engine with an in-memory SQLite engine for all tests."""
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _get_test_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()
