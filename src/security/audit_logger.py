"""
src/security/audit_logger.py

Structured audit logger for TeleRAG-Agent.

Writes a JSON-lines audit trail for every agent invocation, capturing:
  - Timestamp, session ID, query (sanitized), query type
  - Security events (injection attempts, PII detected)
  - Agent outcomes (confidence, iterations, sources used)
  - Latency breakdown

Log file: logs/audit.jsonl (rotation at 10 MB)
"""

import json
import os
import sys
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import BASE_DIR

# ─── log directory ─────────────────────────────────────────────
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_PATH = LOG_DIR / "audit.jsonl"
MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ─── stdlib logger for console output ─────────────────────────
_console_logger = logging.getLogger("telerag.audit")
if not _console_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _console_logger.addHandler(_handler)
    _console_logger.setLevel(logging.INFO)


class AuditLogger:
    """
    Writes structured audit records to logs/audit.jsonl.

    Usage:
        logger = AuditLogger()
        session_id = logger.start_session(query="What is DRX?")
        logger.log_security_event(session_id, "pii_detected", {"count": 1})
        logger.log_completion(session_id, confidence=0.82, iterations=1)
    """

    def __init__(self, log_path: Path = AUDIT_LOG_PATH):
        self.log_path = log_path
        self._sessions: dict[str, dict] = {}

    def _rotate_if_needed(self):
        """Rotate log if it exceeds MAX_LOG_SIZE_BYTES."""
        if self.log_path.exists() and self.log_path.stat().st_size >= MAX_LOG_SIZE_BYTES:
            rotated = self.log_path.with_suffix(
                f".{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            self.log_path.rename(rotated)
            _console_logger.info(f"Audit log rotated to {rotated.name}")

    def _write(self, record: dict):
        """Append a JSON record to the audit log."""
        self._rotate_if_needed()
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            _console_logger.error(f"Failed to write audit log: {e}")

    def start_session(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: str = "anonymous",
    ) -> str:
        """
        Begin an audit session for one agent invocation.

        Returns:
            session_id: UUID string to pass to subsequent log calls.
        """
        sid = session_id or str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        self._sessions[sid] = {
            "session_id": sid,
            "start_time": ts,
            "start_epoch": time.time(),
            "user_id": user_id,
            "query_length": len(query),
            "events": [],
        }
        record = {
            "event": "session_start",
            "session_id": sid,
            "timestamp": ts,
            "user_id": user_id,
            "query_length": len(query),
            # Only store first 100 chars of query for privacy
            "query_preview": query[:100] + ("..." if len(query) > 100 else ""),
        }
        self._write(record)
        return sid

    def log_security_event(
        self,
        session_id: str,
        event_type: str,
        details: Optional[dict] = None,
    ):
        """
        Log a security event within a session.

        event_type examples:
          "injection_blocked"  — prompt injection detected and rejected
          "pii_detected"       — PII found in output and redacted
          "input_truncated"    — query was too long and truncated
          "query_rejected"     — query failed sanitization
        """
        ts = datetime.now(timezone.utc).isoformat()
        record = {
            "event": "security_event",
            "event_type": event_type,
            "session_id": session_id,
            "timestamp": ts,
            "details": details or {},
        }
        self._write(record)
        if session_id in self._sessions:
            self._sessions[session_id]["events"].append(event_type)
        _console_logger.warning(f"Security event [{event_type}] session={session_id[:8]}... details={details}")

    def log_retrieval(
        self,
        session_id: str,
        num_sources: int,
        top_score: float,
        iteration: int,
    ):
        """Log retrieval statistics for one RETRIEVE node execution."""
        record = {
            "event": "retrieval",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_sources": num_sources,
            "top_rerank_score": round(top_score, 4),
            "iteration": iteration,
        }
        self._write(record)

    def log_completion(
        self,
        session_id: str,
        confidence: float,
        iterations: int,
        query_type: str = "",
        sources_used: int = 0,
        needs_clarification: bool = False,
        answer_length: int = 0,
    ):
        """Log the final outcome of an agent invocation."""
        session = self._sessions.get(session_id, {})
        start_epoch = session.get("start_epoch", time.time())
        latency_ms = round((time.time() - start_epoch) * 1000, 1)
        ts = datetime.now(timezone.utc).isoformat()

        record = {
            "event": "session_complete",
            "session_id": session_id,
            "timestamp": ts,
            "latency_ms": latency_ms,
            "query_type": query_type,
            "confidence": round(confidence, 3),
            "iterations": iterations,
            "sources_used": sources_used,
            "needs_clarification": needs_clarification,
            "answer_length": answer_length,
            "security_events": session.get("events", []),
        }
        self._write(record)
        _console_logger.info(
            f"Session complete [{session_id[:8]}...] "
            f"latency={latency_ms:.0f}ms conf={confidence:.2f} iter={iterations}"
        )
        # Clean up
        self._sessions.pop(session_id, None)
        return record

    def log_error(self, session_id: str, error: str, error_type: str = "runtime_error"):
        """Log an exception that occurred during agent execution."""
        record = {
            "event": "error",
            "event_type": error_type,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error[:500],  # truncate very long stack traces
        }
        self._write(record)
        _console_logger.error(f"Error [{error_type}] session={session_id[:8]}...: {error[:100]}")


# ── module-level singleton ─────────────────────────────────────
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# ── smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    logger = AuditLogger()

    sid = logger.start_session("What is RRC reconfiguration in 5G NR?", user_id="test_user")
    print(f"Session started: {sid}")

    logger.log_security_event(sid, "input_truncated", {"original_length": 2500, "truncated_to": 2000})
    logger.log_retrieval(sid, num_sources=5, top_score=0.72, iteration=1)
    record = logger.log_completion(
        sid,
        confidence=0.82,
        iterations=1,
        query_type="spec_qa",
        sources_used=5,
        answer_length=350,
    )
    print(f"Completion logged. Latency: {record['latency_ms']}ms")
    print(f"Audit log written to: {AUDIT_LOG_PATH}")

    # Verify it was written
    with open(AUDIT_LOG_PATH) as f:
        lines = f.readlines()
    print(f"Total log entries: {len(lines)}")
