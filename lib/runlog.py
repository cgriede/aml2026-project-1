import json
from datetime import datetime, timezone
from pathlib import Path

import GLOBAL as G

RUNS_DIR = Path("runs")
JSONL = RUNS_DIR / "log.jsonl"
MD = RUNS_DIR / "log.md"

_HEAD = ("ts", "NOTE")


def append_run(metrics: dict, models: list) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **_globals(),
        **metrics,
        "models": {
            m.__class__.__name__: _jsonable(m.get_params())
            for m in models
        },
    }
    with JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    _write_md()


def _globals() -> dict:
    out = {}
    for k, v in vars(G).items():
        if k.isupper():
            out[k] = str(v) if isinstance(v, Path) else v
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _write_md() -> None:
    rows = []
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    keys = []
    seen = set()
    for k in _HEAD:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    extras = sorted({k for r in rows for k, v in r.items() if k not in seen and _is_scalar(v)})
    keys.extend(extras)

    def cell(r, k):
        v = r.get(k, "")
        if v is None:
            return ""
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(cell(r, k) for k in keys) + " |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
