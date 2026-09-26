"""Idempotency-Key 幂等创建接口测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        c._test_main = main  # 供用例访问 store / 数据库路径
        yield c


_BASE = {
    "measure_heights": [1] * 11,
    "silence_seconds": [1, 5, 9, 5, 9, 5, 5, 1, 5, 5, 0],
    "page_height": 3,
    "min_turn_time": 1,
}

_KEY = {"Idempotency-Key": "retry-key-001"}


def _plan_rows(client):
    db_path = client._test_main.DB_PATH
    with sqlite3.connect(db_path) as conn:
        plans = conn.execute("SELECT id FROM plans").fetchall()
        keys = conn.execute("SELECT key, plan_id FROM idempotency_keys").fetchall()
    return plans, keys


def test_same_key_same_content_returns_same_plan_without_new_record(client):
    first = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert first.status_code == 201, first.text
    body1 = first.json()

    # 网络重试：同键同内容 -> 同一已存方案 ID、创建时间与结果
    second = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert second.status_code == 201, second.text
    body2 = second.json()
    assert body2["id"] == body1["id"]
    assert body2["created_at"] == body1["created_at"]
    assert body2["result"] == body1["result"]
    assert body2["request"] == body1["request"]

    # 不新增记录：plans 与键绑定各只有一行
    plans, keys = _plan_rows(client)
    assert len(plans) == 1
    assert keys == [("retry-key-001", body1["id"])]

    # GET 能查询到复用结果
    got = client.get(f"/api/v1/plans/{body1['id']}")
    assert got.status_code == 200
    assert got.json()["result"] == body1["result"]
    assert got.json()["created_at"] == body1["created_at"]


def test_same_key_different_content_returns_409(client):
    first = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert first.status_code == 201, first.text

    changed = dict(_BASE, page_height=4)
    conflict = client.post("/api/v1/plans", json=changed, headers=_KEY)
    assert conflict.status_code == 409, conflict.text
    assert "Idempotency-Key" in conflict.json()["detail"]

    # 冲突不写入任何记录
    plans, keys = _plan_rows(client)
    assert len(plans) == 1
    assert len(keys) == 1

    # 同一键仍可重试原内容
    retry = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert retry.status_code == 201
    assert retry.json()["id"] == first.json()["id"]


def test_no_key_creates_new_plan_every_time(client):
    first = client.post("/api/v1/plans", json=_BASE)
    second = client.post("/api/v1/plans", json=_BASE)
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    plans, keys = _plan_rows(client)
    assert len(plans) == 2
    assert keys == []


def test_different_keys_same_content_create_separate_plans(client):
    first = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "key-a"}
    )
    second = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "key-b"}
    )
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    plans, keys = _plan_rows(client)
    assert len(plans) == 2
    assert len(keys) == 2


def test_object_field_order_irrelevant_for_content_identity(client):
    first = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert first.status_code == 201, first.text

    # 手工构造字段顺序不同的原始 JSON：对象字段顺序不影响内容标识
    reordered = {
        "forbidden_breaks": [],
        "min_turn_time": _BASE["min_turn_time"],
        "required_breaks": [],
        "page_height": _BASE["page_height"],
        "silence_seconds": _BASE["silence_seconds"],
        "measure_heights": _BASE["measure_heights"],
    }
    raw = json.dumps(reordered)
    second = client.post(
        "/api/v1/plans",
        content=raw,
        headers={**_KEY, "Content-Type": "application/json"},
    )
    assert second.status_code == 201, second.text
    assert second.json()["id"] == first.json()["id"]
    plans, _ = _plan_rows(client)
    assert len(plans) == 1


def test_array_order_is_significant_for_content_identity(client):
    payload = dict(_BASE, required_breaks=[5, 8])
    first = client.post("/api/v1/plans", json=payload, headers=_KEY)
    assert first.status_code == 201, first.text

    # 数组顺序保留：[5, 8] 与 [8, 5] 视为不同内容
    reordered = dict(_BASE, required_breaks=[8, 5])
    conflict = client.post("/api/v1/plans", json=reordered, headers=_KEY)
    assert conflict.status_code == 409, conflict.text


def test_planning_failure_does_not_consume_key(client):
    # 该请求规划失败（位置 1 禁翻导致无合法方案），不占键
    bad = {
        "measure_heights": [5, 5],
        "silence_seconds": [9, 0],
        "page_height": 6,
        "min_turn_time": 1,
        "forbidden_breaks": [1],
    }
    failed = client.post("/api/v1/plans", json=bad, headers=_KEY)
    assert failed.status_code == 422, failed.text
    plans, keys = _plan_rows(client)
    assert plans == [] and keys == []

    # 同一键换用可规划的内容：正常创建
    ok = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert ok.status_code == 201, ok.text
    # 重试仍返回同一方案
    retry = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert retry.json()["id"] == ok.json()["id"]


def test_validation_failure_does_not_consume_key(client):
    invalid = dict(_BASE, measure_heights=[])
    resp = client.post("/api/v1/plans", json=invalid, headers=_KEY)
    assert resp.status_code == 422, resp.text
    _, keys = _plan_rows(client)
    assert keys == []

    ok = client.post("/api/v1/plans", json=_BASE, headers=_KEY)
    assert ok.status_code == 201, ok.text


def test_empty_or_blank_key_rejected_with_400(client):
    for bad_value in ("", "   "):
        resp = client.post(
            "/api/v1/plans",
            json=_BASE,
            headers={"Idempotency-Key": bad_value},
        )
        assert resp.status_code == 400, resp.text
    plans, keys = _plan_rows(client)
    assert plans == [] and keys == []


def test_overlong_key_rejected_with_400(client):
    resp = client.post(
        "/api/v1/plans",
        json=_BASE,
        headers={"Idempotency-Key": "k" * 256},
    )
    assert resp.status_code == 400, resp.text
    ok = client.post(
        "/api/v1/plans",
        json=_BASE,
        headers={"Idempotency-Key": "k" * 255},
    )
    assert ok.status_code == 201, ok.text


def test_concurrent_same_key_creates_single_plan(client):
    """并发相同键同内容：所有请求拿到同一方案，库里只有一份记录。"""
    main = client._test_main
    assert main.store is not None
    request_dump = dict(_BASE, required_breaks=[], forbidden_breaks=[])
    result_dump = {
        "page_count": 4,
        "shortest_turn_silence": 5.0,
        "breaks": [3, 6, 9],
        "pages": [],
    }
    outcomes: list[tuple[str, bool]] = []
    errors: list[Exception] = []

    def worker():
        try:
            outcomes.append(
                main.store.save_idempotent(
                    "race-key", "hash-of-content", request_dump, result_dump
                )
            )
        except Exception as exc:  # pragma: no cover - 失败时由断言暴露
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    plan_ids = {plan_id for plan_id, _ in outcomes}
    assert len(plan_ids) == 1
    # 恰好一个请求真正创建，其余全部复用
    assert sum(1 for _, created in outcomes if created) == 1
    plans, keys = _plan_rows(client)
    assert len(plans) == 1
    assert keys == [("race-key", plan_ids.pop())]


def test_concurrent_same_key_different_content_single_winner(client):
    """并发相同键不同内容：只有一个内容能绑定成功，其余报冲突且不写记录。"""
    main = client._test_main
    assert main.store is not None
    outcomes: list[object] = []

    def worker(content_hash: str):
        try:
            outcomes.append(
                main.store.save_idempotent(
                    "race-key-2", content_hash, dict(_BASE), {"pages": []}
                )
            )
        except Exception as exc:
            outcomes.append(exc)

    from app.storage import IdempotencyConflictError

    threads = [
        threading.Thread(target=worker, args=(f"hash-{i % 3}",)) for i in range(15)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [o for o in outcomes if not isinstance(o, Exception)]
    conflicts = [o for o in outcomes if isinstance(o, IdempotencyConflictError)]
    # 恰好一个请求真正创建；其余同内容请求复用同一方案，不同内容全部冲突
    assert sum(1 for _, created in successes if created) == 1
    assert len({plan_id for plan_id, _ in successes}) == 1
    assert len(successes) + len(conflicts) == 15
    assert len(conflicts) > 0
    plans, keys = _plan_rows(client)
    assert len(plans) == 1
    assert len(keys) == 1
