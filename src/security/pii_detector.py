"""
src/security/pii_detector.py

PII (Personally Identifiable Information) detector for TeleRAG-Agent.

Detects and redacts sensitive telecom-specific data from LLM outputs
before they are shown to the user or written to logs.

Detects:
  - IPv4 / IPv6 addresses
  - MAC addresses
  - IMSI (International Mobile Subscriber Identity)
  - IMEI (International Mobile Equipment Identity)
  - Cell IDs / eNB IDs / gNB IDs
  - Phone numbers
  - GPS coordinates
  - Generic secret-looking strings (bearer tokens, API keys)
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ─── PII pattern definitions ──────────────────────────────────

@dataclass
class PIIPattern:
    name: str
    pattern: re.Pattern
    replacement: str
    risk_level: str  # "high" | "medium" | "low"


_PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="IPv4",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        replacement="[IPv4_REDACTED]",
        risk_level="high",
    ),
    PIIPattern(
        name="IPv6",
        pattern=re.compile(
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
            r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
            r"\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b"
        ),
        replacement="[IPv6_REDACTED]",
        risk_level="high",
    ),
    PIIPattern(
        name="MAC_address",
        pattern=re.compile(
            r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b"
        ),
        replacement="[MAC_REDACTED]",
        risk_level="high",
    ),
    PIIPattern(
        name="IMSI",
        # 15-digit number starting with country code (2-3 digits) — strict context check
        pattern=re.compile(
            r"\bIMSI[:\s]*(\d{15})\b|\b(2[0-9]{2}[0-9]{2}[0-9]{10})\b",
            re.IGNORECASE,
        ),
        replacement="[IMSI_REDACTED]",
        risk_level="high",
    ),
    PIIPattern(
        name="IMEI",
        # 15-digit IMEI
        pattern=re.compile(r"\bIMEI[:\s]*(\d{15})\b", re.IGNORECASE),
        replacement="[IMEI_REDACTED]",
        risk_level="high",
    ),
    PIIPattern(
        name="Cell_ID",
        # Matches patterns like cellId=12345, cell_id: 67890, CELL_001
        pattern=re.compile(
            r"\b(?:cell[-_]?id|cellId|eCellId|nCellId)[:\s=]*([0-9A-Za-z_\-]{3,20})\b",
            re.IGNORECASE,
        ),
        replacement="[CELL_ID_REDACTED]",
        risk_level="medium",
    ),
    PIIPattern(
        name="eNB_gNB_ID",
        # eNB IDs like eNB=1234 or gNB_01
        pattern=re.compile(
            r"\b(?:eNB|gNB|gNodeB|eNodeB)[-_]?(?:id|ID)?[:\s=]*([0-9A-Za-z_\-]{2,15})\b",
            re.IGNORECASE,
        ),
        replacement="[NB_ID_REDACTED]",
        risk_level="medium",
    ),
    PIIPattern(
        name="Phone_number",
        pattern=re.compile(
            r"\b(?:\+?[1-9]\d{0,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        replacement="[PHONE_REDACTED]",
        risk_level="high",
    ),
    PIIPattern(
        name="GPS_coordinates",
        pattern=re.compile(
            r"\b(-?\d{1,3}\.\d{4,})[,\s]+(-?\d{1,3}\.\d{4,})\b"
        ),
        replacement="[GPS_REDACTED]",
        risk_level="medium",
    ),
    PIIPattern(
        name="Bearer_token",
        # Bearer tokens / API keys — long alphanumeric strings after key indicators
        pattern=re.compile(
            r"\b(?:Bearer|Token|ApiKey|api_key|Authorization)[:\s]+([A-Za-z0-9\-_\.]{20,})\b",
            re.IGNORECASE,
        ),
        replacement="[TOKEN_REDACTED]",
        risk_level="high",
    ),
]


@dataclass
class PIIDetectionResult:
    original_text: str
    redacted_text: str
    findings: list[dict]  # [{name, count, risk_level}, ...]
    has_pii: bool


def detect_and_redact(text: str, redact: bool = True) -> PIIDetectionResult:
    """
    Detect and optionally redact PII from text.

    Args:
        text:   Text to scan (model output, user query, log entry).
        redact: If True, replace detected PII with redaction placeholders.
                If False, only detect (return original text unchanged).

    Returns:
        PIIDetectionResult with findings and optionally redacted text.
    """
    findings: dict[str, int] = {}
    redacted = text

    for pii in _PII_PATTERNS:
        matches = pii.pattern.findall(redacted if redact else text)
        if matches:
            count = len(matches) if isinstance(matches[0], str) else len(matches)
            findings[pii.name] = {"count": count, "risk_level": pii.risk_level}
            if redact:
                redacted = pii.pattern.sub(pii.replacement, redacted)

    return PIIDetectionResult(
        original_text=text,
        redacted_text=redacted if redact else text,
        findings=[{"name": name, **info} for name, info in findings.items()],
        has_pii=len(findings) > 0,
    )


def redact(text: str) -> str:
    """Convenience function — detect and redact, return cleaned text."""
    return detect_and_redact(text, redact=True).redacted_text


def has_pii(text: str) -> bool:
    """Quick check — does this text contain any PII?"""
    return detect_and_redact(text, redact=False).has_pii


# ── smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "The gNB at IP 192.168.1.100 has cellId=12345 and IMSI: 310260000000001",
        "MAC address 00:1A:2B:3C:4D:5E detected on node gNB-01",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc123xyz789",
        "The RRC reconfiguration is defined in TS 38.331 clause 5.3.3",  # No PII
        "GPS location: 17.3850, 78.4867 for base station",
    ]

    for text in test_texts:
        result = detect_and_redact(text)
        print(f"\nInput:    {text[:80]}")
        print(f"Redacted: {result.redacted_text[:80]}")
        print(f"Findings: {result.findings}")
