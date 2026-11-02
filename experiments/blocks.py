"""Small HTML blocks for experiment notebooks. No extra packages."""

from __future__ import annotations

import html

import pandas as pd


class Block:
    def __init__(self, html_body: str, text: str):
        self._html = html_body
        self._text = text

    def _repr_html_(self) -> str:
        return self._html

    def __repr__(self) -> str:
        return self._text


_STYLE = """
<style>
.exp { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 14px; color: #1c1c1c; }
.exp h3 { font-size: 14px; font-weight: 600; margin: 0 0 8px; }
.exp table { border-collapse: collapse; }
.exp th { text-align: left; font-weight: 600; color: #444; }
.exp td, .exp th { padding: 5px 14px 5px 0; border-bottom: 1px solid #e6e6e6; vertical-align: middle; }
.exp td.num, .exp th.num { text-align: right; font-variant-numeric: tabular-nums; }
.exp .track { width: 132px; height: 8px; background: #f0f0f0; }
.exp .fill { height: 8px; background: #2f6f4e; }
</style>
"""


def report(df: pd.DataFrame, title: str, bar: str | None = None) -> Block:
    show = df.reset_index(drop=True)
    floats = [c for c in show.columns if pd.api.types.is_float_dtype(show[c])]
    ints = [c for c in show.columns if pd.api.types.is_integer_dtype(show[c])]
    draw = bar if bar and show[bar].min() >= 0 else None
    vmax = float(show[draw].max()) if draw else 1.0
    if vmax <= 0:
        vmax = 1.0

    def head(col: str) -> str:
        kind = "num" if col in floats or col in ints else ""
        return f"<th class='{kind}'>{html.escape(str(col))}</th>"

    heads = "".join(head(c) for c in show.columns)
    if draw:
        heads += "<th></th>"
    body = []
    for _, row in show.iterrows():
        cells = []
        for col in show.columns:
            val = row[col]
            if col in floats and pd.notna(val):
                cells.append(f"<td class='num'>{val:.3f}</td>")
            elif col in ints and pd.notna(val):
                cells.append(f"<td class='num'>{int(val)}</td>")
            else:
                cells.append(f"<td>{html.escape('' if pd.isna(val) else str(val))}</td>")
        if draw:
            width = max(0.0, float(row[draw])) / vmax * 100
            cells.append(
                f"<td><div class='track'><div class='fill' style='width:{width:.1f}%'></div></div></td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    page = (
        _STYLE
        + "<div class='exp'>"
        + f"<h3>{html.escape(title)}</h3>"
        + f"<table><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )
    return Block(page, show.to_string(index=False))
