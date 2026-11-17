"""Fixtures globales de pytest."""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(scope="session")
def qapp():
    """QApplication único, headless (offscreen), para tests de la GUI PyQt6."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def seed_42_rng() -> np.random.Generator:
    """Generador determinístico con seed 42."""
    return np.random.default_rng(42)


@pytest.fixture
def seed() -> int:
    """Seed canónico de tests."""
    return 42
