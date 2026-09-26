"""本地持久化：SQLite 保存规划结果与幂等键绑定。"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class IdempotencyConflictError(Exception):
    """同一幂等键已绑定了不同内容的请求。"""


class PlanStore:
    """线程安全的 SQLite 存储。每次操作使用独立连接，避免跨线程共享。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self.init_db()

    def init_db(self) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)

    def save(self, request: dict, result: dict) -> str:
        plan_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(self._db_path) as conn:
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

    def get(self, plan_id: str) -> Optional[dict]:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, created_at, request_json, result_json FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "request": json.loads(row[2]),
            "result": json.loads(row[3]),
        }

    def get_idempotency_binding(self, key: str) -> Optional[dict]:
        """按幂等键查询已建立的绑定；无绑定返回 None。"""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT key, request_hash, plan_id, created_at"
                " FROM idempotency_keys WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": row[0],
            "request_hash": row[1],
            "plan_id": row[2],
            "created_at": row[3],
        }

    def save_idempotent(
        self, key: str, request_hash: str, request: dict, result: dict
    ) -> Tuple[str, bool]:
        """在单个事务内原子地保存方案并绑定幂等键。

        返回 ``(plan_id, created)``：
        - 键未绑定：同事务插入方案与键绑定，``created=True``；
        - 键已绑定且内容标识一致：不写入任何记录，返回已存方案 ID，
          ``created=False``；
        - 键已绑定但内容标识不同：抛出 :class:`IdempotencyConflictError`，
          不写入任何记录。

        事务以 ``BEGIN IMMEDIATE`` 开启（进入即取写锁），叠加键列主键约束，
        保证并发相同键的请求只会建立一份方案。
        """
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT request_hash, plan_id FROM idempotency_keys WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    conn.rollback()
                    if row[0] != request_hash:
                        raise IdempotencyConflictError(
                            f"Idempotency-Key 冲突：键 {key!r} 已绑定不同的请求内容，"
                            "请更换幂等键或去掉该请求头后重新创建"
                        )
                    return row[1], False
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
                    " (key, request_hash, plan_id, created_at) VALUES (?, ?, ?, ?)",
                    (key, request_hash, plan_id, created_at),
                )
                conn.commit()
                return plan_id, True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
