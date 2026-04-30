"""Generic schema-based parser factory for arbitrary RoleDefinitions.

Given a RoleDefinition.output_schema, produces a parser function that
extracts and validates the expected fields from LLM JSON output.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def make_schema_parser(output_schema: dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a parser function from a RoleDefinition.output_schema.

    Parameters
    ----------
    output_schema : dict
        Maps field_name -> type_spec. type_spec is either a string
        ("string", "number", "boolean", "integer") or a dict with
        a "type" key.

    Returns
    -------
    parser : callable
        (raw_dict) -> validated_dict with defaults for missing fields.
    """
    # Pre-compute defaults
    defaults: dict[str, Any] = {}
    for key, spec in output_schema.items():
        dtype = spec if isinstance(spec, str) else spec.get("type", "string")
        if dtype == "string":
            defaults[key] = ""
        elif dtype == "number":
            defaults[key] = 0.5
        elif dtype == "boolean":
            defaults[key] = False
        elif dtype == "integer":
            defaults[key] = 0
        else:
            defaults[key] = ""

    def _parse(data: Dict[str, Any]) -> Dict[str, Any]:
        result: dict[str, Any] = {}
        for key, default_val in defaults.items():
            raw = data.get(key, default_val)
            # Type coercion
            spec = output_schema[key]
            dtype = spec if isinstance(spec, str) else spec.get("type", "string")
            try:
                if dtype == "number":
                    result[key] = float(raw)
                elif dtype == "integer":
                    result[key] = int(raw)
                elif dtype == "boolean":
                    result[key] = bool(raw)
                else:
                    result[key] = str(raw) if raw is not None else ""
            except (ValueError, TypeError):
                result[key] = default_val
        return result

    return _parse


def schema_to_expected_fields(output_schema: dict[str, Any]) -> list[str]:
    """Extract field names from output_schema for regex fallback."""
    return list(output_schema.keys())
