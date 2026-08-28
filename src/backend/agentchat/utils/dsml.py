import html
import json
import re
from typing import Any, Dict, List
from uuid import uuid4


_DSML_INVOKE_RE = re.compile(
    r'<[｜|]+DSML[｜|]+invoke\s+name="(?P<name>[^"]+)"[^>]*>'
    r'(?P<body>.*?)'
    r'(?:\\)?</[｜|]+DSML[｜|]+invoke\s*>',
    re.DOTALL,
)
_DSML_PARAMETER_RE = re.compile(
    r'<[｜|]+DSML[｜|]+parameter\s+name="(?P<name>[^"]+)"'
    r'(?P<attrs>[^>]*)>'
    r'(?P<value>.*?)'
    r'(?:\\)?</[｜|]+DSML[｜|]+parameter\s*>',
    re.DOTALL,
)
_DSML_TOOL_CALLS_RE = re.compile(
    r'<[｜|]+DSML[｜|]+tool(?:\\)?_calls\s*>'
    r'.*?'
    r'(?:\\)?</[｜|]+DSML[｜|]+tool(?:\\)?_calls\s*>',
    re.DOTALL,
)
_DSML_TOOL_CALLS_END_RE = re.compile(
    r'(?:\\)?</[｜|]+DSML[｜|]+tool(?:\\)?_calls\s*>',
    re.DOTALL,
)
_DSML_PREFIXES = ("<｜DSML", "<｜｜DSML", "<|DSML", "<||DSML")


class DSMLStreamSanitizer:
    """Hide leaked DSML blocks without delaying ordinary streamed text."""

    def __init__(self):
        self._buffer = ""
        self._inside_dsml = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        visible_parts: List[str] = []

        while self._buffer:
            if self._inside_dsml:
                end_match = _DSML_TOOL_CALLS_END_RE.search(self._buffer)
                if end_match is None:
                    return "".join(visible_parts)
                self._buffer = self._buffer[end_match.end():]
                self._inside_dsml = False
                continue

            marker_start = self._buffer.find("<")
            if marker_start < 0:
                visible_parts.append(self._buffer)
                self._buffer = ""
                break
            if marker_start > 0:
                visible_parts.append(self._buffer[:marker_start])
                self._buffer = self._buffer[marker_start:]

            if any(prefix.startswith(self._buffer) for prefix in _DSML_PREFIXES):
                break
            if any(self._buffer.startswith(prefix) for prefix in _DSML_PREFIXES):
                self._inside_dsml = True
                continue

            visible_parts.append(self._buffer[0])
            self._buffer = self._buffer[1:]

        return "".join(visible_parts)


def _parse_dsml_parameter(value: str, attrs: str) -> Any:
    value = html.unescape(value.strip().removesuffix("\\"))
    if re.search(r'\bstring="true"', attrs, re.IGNORECASE):
        return value

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def recover_dsml_tool_calls(content: Any) -> tuple[str, List[Dict[str, Any]]]:
    """Recover tool calls leaked by a model/provider as DSML message text."""
    if not isinstance(content, str) or "DSML" not in content:
        return content if isinstance(content, str) else "", []

    tool_calls: List[Dict[str, Any]] = []
    for invoke_match in _DSML_INVOKE_RE.finditer(content):
        tool_name = invoke_match.group("name").replace("\\_", "_").strip()
        if not tool_name:
            continue

        args: Dict[str, Any] = {}
        for parameter_match in _DSML_PARAMETER_RE.finditer(invoke_match.group("body")):
            parameter_name = parameter_match.group("name").replace("\\_", "_").strip()
            if parameter_name:
                args[parameter_name] = _parse_dsml_parameter(
                    parameter_match.group("value"),
                    parameter_match.group("attrs"),
                )

        tool_calls.append({
            "name": tool_name,
            "args": args,
            "id": f"call_dsml_{uuid4().hex}",
            "type": "tool_call",
        })

    cleaned_content = _DSML_TOOL_CALLS_RE.sub("", content).strip()
    return cleaned_content, tool_calls
