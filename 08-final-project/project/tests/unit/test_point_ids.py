"""Unit-тесты детерминированных ID точек Qdrant."""

from __future__ import annotations

import uuid

from app.services.vectorstore import point_id_for


def test_same_input_same_id() -> None:
    assert point_id_for("deploy.md", 3) == point_id_for("deploy.md", 3)


def test_different_chunks_different_ids() -> None:
    ids = {
        point_id_for("deploy.md", 0),
        point_id_for("deploy.md", 1),
        point_id_for("other.md", 0),
    }
    assert len(ids) == 3


def test_id_is_valid_uuid() -> None:
    value = point_id_for("deploy.md", 0)
    assert str(uuid.UUID(value)) == value
