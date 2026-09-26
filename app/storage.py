"""本地持久化：SQLite 保存规划结果与幂等键绑定。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES plans(id)
);
"""

# 幂等键最大长度（字符），超出由接口层返回 400
IDEMPOTENCY_KEY_MAX_LENGTH = 255


def canonical_request_json(request: dict) -> str:
    """把**校验后的**请求体序列化为稳定 JSON 文本。

    对象（dict）按键名排序，故 JSON 字段书写顺序不影响标识；数组（list）
    顺序原样保留（如 required_breaks 的 [2,5] 与 [5,2] 视为不同内容）。
    入参来自 pydantic 校验后的模型，数值已统一为 int/float，不会出现
    NaN/Infinity（这类请求在到达存储前已被拒绝）。
    """
    return json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_fingerprint(request: dict) -> str:
    """校验后完整请求的稳定内容标识：规范化 JSON 的 SHA-256 十六进制摘要。"""
    payload = canonical_request_json(request).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class IdempotencyConflict(Exception):
    """同一幂等键已绑定内容指纹不同的方案：同键不同内容。"""

    def __init__(self, idempotency_key: str, existing_plan_id: str) -> None:
        self.idempotency_key = idempotency_key
        self.existing_plan_id = existing_plan_id
        super().__init__(
            f"幂等键 {idempotency_key!r} 已用于另一份不同内容的请求"
            f"（已存方案 {existing_plan_id}）：同一 Idempotency-Key 只能"
            "重试完全相同的请求，内容不同请更换新键。"
        )


class PlanStore:
    """线程安全的 SQLite 存储。每次操作使用独立连接，避免跨线程共享。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        # timeout：BEGIN IMMEDIATE 遇其他写事务时等待并发键的首个请求提交，
        # 等待后重查绑定即可复用，而不是报数据库忙
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    @staticmethod
    def _plan_row_to_record(row: tuple) -> dict:
        return {
            "id": row[0],
            "created_at": row[1],
            "request": json.loads(row[2]),
            "result": json.loads(row[3]),
        }

    def save(self, request: dict, result: dict) -> str:
        """未带幂等键的原有保存路径：只新增方案，返回新方案 ID。"""
        plan_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO plans (id, created_at, request_json, result_json)"
                " VALUES (?, ?, ?, ?)",
                (
                    plan_id,
                    created_at,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        return plan_id

    def lookup_idempotency_key(self, idempotency_key: str) -> Optional[dict]:
        """按键读取已绑定方案（快速重试路径，不加写锁）；无绑定返回 None。

        返回记录在 get() 结构基础上额外带 content_hash 字段。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT p.id, p.created_at, p.request_json, p.result_json,"
                "       k.content_hash"
                " FROM idempotency_keys k JOIN plans p ON p.id = k.plan_id"
                " WHERE k.idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        record = self._plan_row_to_record(row[:4])
        record["content_hash"] = row[4]
        return record

    def save_idempotent(
        self, idempotency_key: str, request: dict, result: dict
    ) -> tuple[dict, bool]:
        """在单个 SQLite 事务内原子完成「查重键绑定 → 插入方案 → 写入键绑定」。

        - 键不存在：新建方案并写入绑定，返回 (记录, False)；
        - 键已存在且内容指纹相同：不新增任何记录，返回 (已存记录, True)；
        - 键已存在但内容指纹不同：抛 IdempotencyConflict，事务回滚。

        BEGIN IMMEDIATE 在事务开始即取写锁，配合 idempotency_keys 的主键
        唯一约束，保证并发相同键只有一份方案落库：后来的请求在锁上等待，
        首个请求提交后重查即命中复用路径。调用方须在规划成功后再调用本
        方法，因此规划失败不会占用幂等键。
        """
        digest = content_fingerprint(request)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT plan_id, content_hash FROM idempotency_keys"
                    " WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    existing_plan_id, existing_hash = row
                    if existing_hash != digest:
                        raise IdempotencyConflict(
                            idempotency_key, existing_plan_id
                        )
                    plan_row = conn.execute(
                        "SELECT id, created_at, request_json, result_json"
                        " FROM plans WHERE id = ?",
                        (existing_plan_id,),
                    ).fetchone()
                    conn.execute("COMMIT")
                    return self._plan_row_to_record(plan_row), True

                plan_id = uuid.uuid4().hex
                created_at = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO plans (id, created_at, request_json, result_json)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        plan_id,
                        created_at,
                        json.dumps(request, ensure_ascii=False),
                        json.dumps(result, ensure_ascii=False),
                    ),
                )
                conn.execute(
                    "INSERT INTO idempotency_keys"
                    " (idempotency_key, plan_id, content_hash, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (idempotency_key, plan_id, digest, created_at),
                )
                conn.execute("COMMIT")
                record = {
                    "id": plan_id,
                    "created_at": created_at,
                    "request": request,
                    "result": result,
                }
                return record, False
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def get(self, plan_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at, request_json, result_json FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
        if row is None:
            return None
        return self._plan_row_to_record(row)
