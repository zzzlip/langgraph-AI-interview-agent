"""SQLite 本地命令幂等存储。"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai_interview_worker.application.idempotency import IdempotencyDecision, IdempotencyDecisionType


@dataclass(frozen=True)
class _ProcessedCommandRow:
    command_event_id: str
    task_id: str
    attempt: int
    action: str
    status: str
    payload_hash: str
    updated_at: datetime


class SQLiteProcessedCommandStore:
    """单 Worker 实例的 SQLite 幂等栅栏。"""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def try_start(
        self,
        *,
        command_event_id: str,
        task_id: str,
        attempt: int,
        action: str = "",
        task_type: str,
        payload_hash: str,
        trace_id: str,
        now: datetime,
        timeout_seconds: int,
    ) -> IdempotencyDecision:
        """尝试建立 processing 栅栏，并返回 ACK/NACK 决策。"""
        with self._connect() as connection:
            event_row = self._find_by_event_id(connection, command_event_id)
            if event_row is not None:
                return self._decision_for_existing_row(event_row, payload_hash, now, timeout_seconds)

            task_attempt_row = self._find_by_task_attempt_action(connection, task_id, attempt, action)
            if task_attempt_row is not None:
                decision = self._decision_for_task_attempt_action_row(task_attempt_row, now, timeout_seconds)
                if not decision.should_process:
                    return decision
                self._replace_processing_row(
                    connection,
                    command_event_id=command_event_id,
                    task_id=task_id,
                    attempt=attempt,
                    action=action,
                    task_type=task_type,
                    payload_hash=payload_hash,
                    trace_id=trace_id,
                    now=now,
                )
                return decision

            self._insert_processing_row(
                connection,
                command_event_id=command_event_id,
                task_id=task_id,
                attempt=attempt,
                action=action,
                task_type=task_type,
                payload_hash=payload_hash,
                trace_id=trace_id,
                now=now,
            )
            return IdempotencyDecision(IdempotencyDecisionType.PROCESS, "new_command")

    def mark_completed(self, command_event_id: str, now: datetime) -> None:
        """终态反馈确认发布后标记 completed。"""
        self._mark_terminal(command_event_id, "completed", now)

    def mark_failed(self, command_event_id: str, now: datetime) -> None:
        """失败反馈确认发布后标记 failed。"""
        self._mark_terminal(command_event_id, "failed", now)

    def get_status(self, command_event_id: str) -> str | None:
        """读取命令当前本地状态，供测试和排查使用。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM processed_command WHERE command_event_id = ?",
                (command_event_id,),
            ).fetchone()
        return None if row is None else str(row["status"])

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_command (
                    command_event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    action TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(task_id, attempt, action)
                )
                """
            )
            if self._needs_action_schema_rebuild(connection):
                self._rebuild_action_schema(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _find_by_event_id(self, connection: sqlite3.Connection, command_event_id: str) -> _ProcessedCommandRow | None:
        row = connection.execute(
            """
            SELECT command_event_id, task_id, attempt, action, status, payload_hash, updated_at
            FROM processed_command
            WHERE command_event_id = ?
            """,
            (command_event_id,),
        ).fetchone()
        return _to_row(row)

    def _find_by_task_attempt_action(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        attempt: int,
        action: str,
    ) -> _ProcessedCommandRow | None:
        row = connection.execute(
            """
            SELECT command_event_id, task_id, attempt, action, status, payload_hash, updated_at
            FROM processed_command
            WHERE task_id = ? AND attempt = ? AND action = ?
            """,
            (task_id, attempt, action),
        ).fetchone()
        return _to_row(row)

    def _decision_for_existing_row(
        self,
        row: _ProcessedCommandRow,
        payload_hash: str,
        now: datetime,
        timeout_seconds: int,
    ) -> IdempotencyDecision:
        if row.payload_hash != payload_hash:
            return IdempotencyDecision(IdempotencyDecisionType.ACK_CONFLICT, "payload_hash_conflict")
        return self._decision_for_task_attempt_action_row(row, now, timeout_seconds)

    def _decision_for_task_attempt_action_row(
        self,
        row: _ProcessedCommandRow,
        now: datetime,
        timeout_seconds: int,
    ) -> IdempotencyDecision:
        if row.status in {"completed", "failed"}:
            return IdempotencyDecision(IdempotencyDecisionType.ACK_DUPLICATE, "terminal_duplicate")
        if _is_processing_timeout(row.updated_at, now, timeout_seconds):
            return IdempotencyDecision(IdempotencyDecisionType.PROCESS_RECOVERED, "processing_timeout")
        return IdempotencyDecision(IdempotencyDecisionType.NACK_REQUEUE, "processing_not_timeout")

    def _insert_processing_row(
        self,
        connection: sqlite3.Connection,
        *,
        command_event_id: str,
        task_id: str,
        attempt: int,
        action: str,
        task_type: str,
        payload_hash: str,
        trace_id: str,
        now: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO processed_command (
                command_event_id, task_id, attempt, action, task_type, status,
                payload_hash, trace_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'processing', ?, ?, ?, ?)
            """,
            (
                command_event_id,
                task_id,
                attempt,
                action,
                task_type,
                payload_hash,
                trace_id,
                now.isoformat(),
                now.isoformat(),
            ),
        )

    def _replace_processing_row(
        self,
        connection: sqlite3.Connection,
        *,
        command_event_id: str,
        task_id: str,
        attempt: int,
        action: str,
        task_type: str,
        payload_hash: str,
        trace_id: str,
        now: datetime,
    ) -> None:
        connection.execute(
            "DELETE FROM processed_command WHERE task_id = ? AND attempt = ? AND action = ?",
            (task_id, attempt, action),
        )
        self._insert_processing_row(
            connection,
            command_event_id=command_event_id,
            task_id=task_id,
            attempt=attempt,
            action=action,
            task_type=task_type,
            payload_hash=payload_hash,
            trace_id=trace_id,
            now=now,
        )

    def _mark_terminal(self, command_event_id: str, status: str, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE processed_command
                SET status = ?, updated_at = ?
                WHERE command_event_id = ?
                """,
                (status, now.isoformat(), command_event_id),
            )

    @staticmethod
    def _needs_action_schema_rebuild(connection: sqlite3.Connection) -> bool:
        columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(processed_command)").fetchall()}
        if "action" not in columns:
            return True
        for index in connection.execute("PRAGMA index_list(processed_command)").fetchall():
            if int(index["unique"]) != 1:
                continue
            index_name = str(index["name"])
            index_columns = [
                str(row["name"])
                for row in connection.execute(f"PRAGMA index_info({index_name})").fetchall()
            ]
            if index_columns == ["task_id", "attempt"]:
                return True
        return False

    @staticmethod
    def _rebuild_action_schema(connection: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(processed_command)").fetchall()}
        action_select = "action" if "action" in columns else "''"
        connection.execute("ALTER TABLE processed_command RENAME TO processed_command_old")
        connection.execute(
            """
            CREATE TABLE processed_command (
                command_event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                action TEXT NOT NULL DEFAULT '',
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(task_id, attempt, action)
            )
            """
        )
        connection.execute(
            f"""
            INSERT INTO processed_command (
                command_event_id, task_id, attempt, action, task_type, status,
                payload_hash, trace_id, created_at, updated_at
            )
            SELECT command_event_id, task_id, attempt, {action_select}, task_type, status,
                   payload_hash, trace_id, created_at, updated_at
            FROM processed_command_old
            """
        )
        connection.execute("DROP TABLE processed_command_old")


def _to_row(row: sqlite3.Row | None) -> _ProcessedCommandRow | None:
    if row is None:
        return None
    return _ProcessedCommandRow(
        command_event_id=str(row["command_event_id"]),
        task_id=str(row["task_id"]),
        attempt=int(row["attempt"]),
        action=str(row["action"]),
        status=str(row["status"]),
        payload_hash=str(row["payload_hash"]),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _is_processing_timeout(updated_at: datetime, now: datetime, timeout_seconds: int) -> bool:
    return (now - updated_at).total_seconds() > timeout_seconds
