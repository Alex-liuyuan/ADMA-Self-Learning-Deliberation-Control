"""Robust JSON Parser — enhanced LLM response parsing with retry and fallback.

Fixes the brittle JSON parsing that silently fails. Provides:
  1. Primary parse attempt (strict JSON)
  2. Repair attempt (fix common LLM JSON errors)
  3. Regex extraction fallback (extract key fields from text)
  4. Retry with repair prompt (one LLM call)
"""

from __future__ import annotations

import json
import re
from typing import Any

from debate_rl_v2.logging_config import get_logger

logger = get_logger("llm.json_parser")


class RobustJSONParser:
    """Multi-strategy JSON parser for LLM responses.

    Tracks parse failure rates as an LLM quality metric.
    """

    def __init__(self) -> None:
        self._total_attempts = 0
        self._direct_success = 0
        self._repair_success = 0
        self._regex_fallback = 0
        self._failures = 0

    def parse(
        self,
        text: str,
        expected_fields: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Parse JSON from LLM response text.

        Tries multiple strategies in order:
        1. Direct JSON parse
        2. Extract JSON block from markdown
        3. Repair common errors and retry
        4. Regex extraction of expected fields

        Parameters
        ----------
        text : str
            Raw LLM response text.
        expected_fields : list[str] | None
            Fields to extract via regex fallback.

        Returns
        -------
        parsed : dict | None
            Parsed JSON dict, or None if all strategies fail.
        """
        self._total_attempts += 1

        # Strategy 1: Direct parse
        result = self._try_direct(text)
        if result is not None:
            self._direct_success += 1
            return result

        # Strategy 2: Extract from markdown code block
        result = self._try_extract_block(text)
        if result is not None:
            self._direct_success += 1
            return result

        # Strategy 3: Repair common errors
        result = self._try_repair(text)
        if result is not None:
            self._repair_success += 1
            return result

        # Strategy 4: Regex extraction
        if expected_fields:
            result = self._try_regex(text, expected_fields)
            if result is not None:
                self._regex_fallback += 1
                logger.warning("JSON parse fell back to regex extraction")
                return result

        self._failures += 1
        logger.error("All JSON parse strategies failed for text: %s", text[:200])
        return None

    def _try_direct(self, text: str) -> dict[str, Any] | None:
        """Try direct JSON parse."""
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _try_extract_block(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from markdown code blocks."""
        # Match ```json ... ``` or ``` ... ```
        patterns = [
            r"```json\s*\n?(.*?)\n?\s*```",
            r"```\s*\n?(.*?)\n?\s*```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Outermost braces
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue
        return None

    def _try_repair(self, text: str) -> dict[str, Any] | None:
        """Try to repair common JSON errors."""
        # Find the JSON-like portion
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        json_text = text[start:end + 1]

        # Common repairs
        repairs = [
            # Trailing commas before closing braces
            (r",\s*}", "}"),
            (r",\s*]", "]"),
            # Single quotes → double quotes
            (r"'([^']*)'(?=\s*:)", r'"\1"'),
            (r":\s*'([^']*)'", r': "\1"'),
            # Unquoted keys
            (r"(\{|,)\s*([a-zA-Z_]\w*)\s*:", r'\1 "\2":'),
            # Python True/False/None → JSON
            (r"\bTrue\b", "true"),
            (r"\bFalse\b", "false"),
            (r"\bNone\b", "null"),
        ]

        for pattern, replacement in repairs:
            json_text = re.sub(pattern, replacement, json_text)

        try:
            result = json.loads(json_text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _try_regex(
        self,
        text: str,
        expected_fields: list[str],
    ) -> dict[str, Any] | None:
        """Extract expected fields via regex patterns."""
        result: dict[str, Any] = {}

        for field_name in expected_fields:
            # Try various patterns for each field
            patterns = [
                # "field": "value" or "field": value
                rf'"{field_name}"\s*:\s*"([^"]*)"',
                rf'"{field_name}"\s*:\s*([0-9.]+)',
                rf'"{field_name}"\s*:\s*(true|false)',
                # field: value (unquoted)
                rf"{field_name}\s*[:=]\s*([0-9.]+)",
                rf"{field_name}\s*[:=]\s*\"([^\"]*?)\"",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    val = match.group(1)
                    # Try to convert to appropriate type
                    if val in ("true", "True"):
                        result[field_name] = True
                    elif val in ("false", "False"):
                        result[field_name] = False
                    else:
                        try:
                            result[field_name] = float(val)
                        except ValueError:
                            result[field_name] = val
                    break

        return result if result else None

    def build_repair_prompt(self, original_text: str) -> str:
        """Build a prompt asking the LLM to fix its JSON output."""
        return (
            "你的上一次回复包含无效的JSON格式。请仅返回修正后的JSON，不要包含其他文字。\n\n"
            f"原始回复:\n{original_text[:500]}\n\n"
            "请返回有效的JSON格式。"
        )

    @property
    def stats(self) -> dict[str, Any]:
        total = max(self._total_attempts, 1)
        return {
            "total_attempts": self._total_attempts,
            "direct_success": self._direct_success,
            "repair_success": self._repair_success,
            "regex_fallback": self._regex_fallback,
            "failures": self._failures,
            "success_rate": (total - self._failures) / total,
            "failure_rate": self._failures / total,
        }

    def reset_stats(self) -> None:
        self._total_attempts = 0
        self._direct_success = 0
        self._repair_success = 0
        self._regex_fallback = 0
        self._failures = 0
