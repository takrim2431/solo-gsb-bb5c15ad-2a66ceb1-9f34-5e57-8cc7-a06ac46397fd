"""POST /api/v1/plans 的 Idempotency-Key 幂等语义端到端测试。"""

from __future__ import annotations

import importlib
import sqlite3
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from app.storage import canonical_request_json, content_fingerprint


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _db_counts(tmp_path) -> tuple[int, int]:
    conn = sqlite3.connect(tmp_path / "data" / "plans.db")
    try:
        plans = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        keys = conn.execute("SELECT COUNT(*) FROM idempotency_keys").fetchone()[0]
    finally:
        conn.close()
    return plans, keys


_BASE = {
    "measure_heights": [1] * 11,
    "silence_seconds": [1, 5, 9, 5, 9, 5, 5, 1, 5, 5, 0],
    "page_height": 3,
    "min_turn_time": 1,
}

_BASE_VARIANT = dict(_BASE, page_height=4)


# ------------------------- 内容标识的规范化性质


def test_canonical_json_ignores_object_key_order_but_keeps_array_order():
    a = {"b": 1, "a": [1, 2, 3]}
    b = {"a": [1, 2, 3], "b": 1}
    assert canonical_request_json(a) == canonical_request_json(b)
    assert content_fingerprint(a) == content_fingerprint(b)

    reversed_arr = {"b": 1, "a": [3, 2, 1]}
    assert content_fingerprint(a) != content_fingerprint(reversed_arr)


# ------------------------- 未带键：原方式


def test_no_key_creates_new_plan_every_time(client, tmp_path):
    r1 = client.post("/api/v1/plans", json=_BASE)
    r2 = client.post("/api/v1/plans", json=_BASE)
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]
    assert _db_counts(tmp_path) == (2, 0)


@pytest.mark.parametrize("header_value", ["", "   "])
def test_empty_or_blank_key_treated_as_absent(client, tmp_path, header_value):
    r1 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": header_value}
    )
    r2 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": header_value}
    )
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]
    assert _db_counts(tmp_path) == (2, 0)


# ------------------------- 同键同内容：复用


def test_same_key_same_content_returns_stored_plan_without_new_record(
    client, tmp_path
):
    r1 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "key-1"}
    )
    assert r1.status_code == 201, r1.text
    first = r1.json()

    # 网络重试：同键同内容
    r2 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "key-1"}
    )
    assert r2.status_code == 200, r2.text
    second = r2.json()

    assert second["id"] == first["id"]
    assert second["created_at"] == first["created_at"]
    assert second["result"] == first["result"]
    assert second["request"] == first["request"]
    assert _db_counts(tmp_path) == (1, 1)

    # GET 查询到的即复用的同一份结果
    got = client.get(f"/api/v1/plans/{first['id']}").json()
    assert got["id"] == first["id"]
    assert got["created_at"] == first["created_at"]
    assert got["result"] == first["result"]

    # 再来一次仍是同一份
    r3 = client.post(
        "/api/v1/plans", json=_BASE, headers={"idempotency-key": "key-1"}
    )
    assert r3.status_code == 200
    assert r3.json()["id"] == first["id"]
    assert _db_counts(tmp_path) == (1, 1)


def test_replay_works_when_json_object_fields_reordered(client, tmp_path):
    r1 = client.post(
        "/api/v1/plans",
        json={
            "min_turn_time": 1,
            "page_height": 3,
            "silence_seconds": _BASE["silence_seconds"],
            "measure_heights": _BASE["measure_heights"],
        },
        headers={"Idempotency-Key": "key-reorder"},
    )
    assert r1.status_code == 201, r1.text

    r2 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "key-reorder"}
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["id"] == r1.json()["id"]
    assert _db_counts(tmp_path) == (1, 1)


def test_key_whitespace_is_trimmed(client, tmp_path):
    r1 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "  pad-key  "}
    )
    r2 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "pad-key"}
    )
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    assert _db_counts(tmp_path) == (1, 1)


def test_different_keys_create_separate_plans(client, tmp_path):
    r1 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "k-a"}
    )
    r2 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "k-b"}
    )
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]
    assert _db_counts(tmp_path) == (2, 2)


# ------------------------- 同键不同内容：409


def test_same_key_different_content_returns_409_and_keeps_first_plan(
    client, tmp_path
):
    r1 = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "dup"}
    )
    assert r1.status_code == 201, r1.text
    first_id = r1.json()["id"]

    conflict = client.post(
        "/api/v1/plans", json=_BASE_VARIANT, headers={"Idempotency-Key": "dup"}
    )
    assert conflict.status_code == 409, conflict.text
    assert first_id in conflict.json()["detail"] or "幂等键" in conflict.json()["detail"]

    # 冲突请求未新增方案/绑定，原方案仍可查
    assert _db_counts(tmp_path) == (1, 1)
    got = client.get(f"/api/v1/plans/{first_id}")
    assert got.status_code == 200
    assert got.json()["request"]["page_height"] == 3

    # 同键重试首次的内容仍可复用
    replay = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "dup"}
    )
    assert replay.status_code == 200
    assert replay.json()["id"] == first_id


def test_array_order_is_significant(client, tmp_path):
    payload_a = dict(_BASE, required_breaks=[2, 5])
    payload_b = dict(_BASE, required_breaks=[5, 2])
    r1 = client.post(
        "/api/v1/plans", json=payload_a, headers={"Idempotency-Key": "arr"}
    )
    assert r1.status_code == 201, r1.text
    r2 = client.post(
        "/api/v1/plans", json=payload_b, headers={"Idempotency-Key": "arr"}
    )
    assert r2.status_code == 409
    assert _db_counts(tmp_path) == (1, 1)


# ------------------------- 规划失败不占键


def test_planning_failure_does_not_occupy_key(client, tmp_path):
    # 单小节高度超过页高：规划失败 422
    invalid = {
        "measure_heights": [9],
        "silence_seconds": [0],
        "page_height": 5,
        "min_turn_time": 1,
    }
    bad = client.post(
        "/api/v1/plans", json=invalid, headers={"Idempotency-Key": "retry-me"}
    )
    assert bad.status_code == 422, bad.text
    assert _db_counts(tmp_path) == (0, 0)

    # 同键再提交一份合法的不同请求：键未被占用，正常创建
    ok = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": "retry-me"}
    )
    assert ok.status_code == 201, ok.text
    assert _db_counts(tmp_path) == (1, 1)


# ------------------------- 请求头约束


def test_overlong_key_rejected_with_400(client, tmp_path):
    long_key = "k" * 256
    resp = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": long_key}
    )
    assert resp.status_code == 400
    assert "Idempotency-Key" in resp.json()["detail"]
    assert _db_counts(tmp_path) == (0, 0)


def test_key_of_exact_max_length_accepted(client, tmp_path):
    key = "k" * 255
    resp = client.post(
        "/api/v1/plans", json=_BASE, headers={"Idempotency-Key": key}
    )
    assert resp.status_code == 201, resp.text


# ------------------------- 并发相同键：只建立一份方案


def test_concurrent_same_key_creates_exactly_one_plan(client, tmp_path):
    n = 12
    barrier = threading.Barrier(n)
    results: list[dict] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        resp = client.post(
            "/api/v1/plans",
            json=_BASE,
            headers={"Idempotency-Key": "concurrent-same"},
        )
        with results_lock:
            results.append(
                {"status": resp.status_code, "id": resp.json().get("id")}
            )

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(r["status"] for r in results)
    assert statuses == [200] * (n - 1) + [201]
    ids = {r["id"] for r in results}
    assert len(ids) == 1
    assert _db_counts(tmp_path) == (1, 1)


def test_concurrent_same_key_with_different_content(client, tmp_path):
    per_content = 6
    n = per_content * 2
    barrier = threading.Barrier(n)
    results: list[dict] = []
    results_lock = threading.Lock()

    def worker(payload: dict) -> None:
        barrier.wait()
        resp = client.post(
            "/api/v1/plans",
            json=payload,
            headers={"Idempotency-Key": "concurrent-conflict"},
        )
        with results_lock:
            results.append(
                {"status": resp.status_code, "id": resp.json().get("id")}
            )

    threads = [
        threading.Thread(target=worker, args=(_BASE,))
        for _ in range(per_content)
    ] + [
        threading.Thread(target=worker, args=(_BASE_VARIANT,))
        for _ in range(per_content)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = [r["status"] for r in results]
    # 恰好一个请求创建成功；与其同内容的后来者 200 复用，异内容 409
    assert statuses.count(201) == 1
    assert set(statuses) <= {200, 201, 409}
    assert statuses.count(409) >= 1
    created = [r for r in results if r["status"] == 201][0]
    assert all(
        r["id"] in (created["id"], None)
        for r in results
        if r["status"] in (200, 201)
    )
    assert _db_counts(tmp_path) == (1, 1)
