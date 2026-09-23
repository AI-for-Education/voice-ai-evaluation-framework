"""Build a standalone, shareable HTML page for one IPA leaderboard."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egra_eval2.leaderboard_policy import has_required_eligibility_policy


TASKS = (
    ("passage_passage", "Passage", "corr"),
    ("syllables_grid", "Syllables · grid", "corr"),
    ("syllables_isolated", "Syllables · isolated", "accuracy"),
    ("nonwords_grid", "Non-words · grid", "corr"),
    ("nonwords_isolated", "Non-words · isolated", "accuracy"),
    ("letters_grid", "Letters · grid", "corr"),
    ("letters_isolated", "Letters · isolated", "accuracy"),
)


def _text(value: object, fallback: str = "Not recorded") -> str:
    rendered = str(value).strip() if value is not None else ""
    return html.escape(rendered or fallback, quote=True)


def _percent(value: object) -> str:
    try:
        return f"{float(str(value)):.2f}%"
    except (TypeError, ValueError):
        return "Not recorded"


def _outcome(value: object, kind: str) -> str:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return "Not recorded"
    return f"r {number:.4f}" if kind == "corr" else f"{number:.2f}%"


def _date_label(value: object) -> str:
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value or "Not recorded")


def _generated_label(value: object) -> str:
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %B %Y, %H:%M UTC")
    except (TypeError, ValueError):
        return str(value or "Not recorded")


def _safe_link(url: object, label: object) -> str:
    rendered = str(url or "").strip()
    if urlsplit(rendered).scheme not in {"http", "https"}:
        return _text(label)
    return (
        f'<a href="{_text(rendered)}" target="_blank" rel="noopener">'
        f"{_text(label)}</a>"
    )


def _public_system_name(row: dict[str, str]) -> str:
    """Return a reader-facing name without exposing internal identity fields."""
    return str(row.get("g2p_display_name", "")).strip()


def _public_scoring_route(row: dict[str, str]) -> str:
    """Describe the scoring route without local inventories or identity hashes."""
    system_name = _public_system_name(row)
    if str(row.get("native_output_units", "")).strip().lower() == "phoneme":
        recorded_route = str(row.get("hypothesis_route", "")).lower()
        if "no phoneme conversion" in recorded_route:
            return (
                f"native IPA — already {system_name}-compatible "
                "(no phoneme conversion)"
            )
        return f"native IPA → {system_name} IPA"
    return f"orthographic → {system_name}"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Leaderboard has no rows: {path}")
    required = {
        "rank",
        "model_label",
        "g2p_system_id",
        "g2p_display_name",
        "target_inventory",
        "global_per",
        "global_mer",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Leaderboard is missing columns: {', '.join(missing)}")
    identities = {row["g2p_system_id"] for row in rows}
    if len(identities) != 1:
        raise ValueError("HTML input must contain exactly one G2P system")
    return rows


def _detail_block(row: dict[str, str]) -> str:
    items = (
        ("Model artifact", row.get("model_artifact")),
        ("Inference setup", row.get("inference_setup")),
        ("Execution stack", row.get("execution_stack")),
        ("Native output", row.get("native_output_units")),
        ("Scoring route", _public_scoring_route(row)),
        ("Evidence", row.get("context_evidence")),
    )
    definitions = "".join(
        f"<dt>{_text(label)}</dt><dd>{_text(value)}</dd>" for label, value in items
    )
    return (
        '<details class="context"><summary>Technical context</summary>'
        f"<dl>{definitions}</dl></details>"
    )


def _task_cells(row: dict[str, str]) -> str:
    cells: list[str] = []
    for prefix, _label, outcome_kind in TASKS:
        per_value = row.get(f"{prefix}_per", "")
        mer_value = row.get(f"{prefix}_mer", "")
        outcome_value = row.get(f"{prefix}_{outcome_kind}", "")
        cells.extend(
            (
                f'<td class="result task-start" data-sort="{_text(per_value, "")}">{_text(_percent(per_value))}</td>',
                f'<td class="result" data-sort="{_text(mer_value, "")}">{_text(_percent(mer_value))}</td>',
                f'<td class="result" data-sort="{_text(outcome_value, "")}">{_text(_outcome(outcome_value, outcome_kind))}</td>',
            )
        )
    return "".join(cells)


def _table_rows(rows: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for row in rows:
        searchable = " ".join(
            str(row.get(key, ""))
            for key in (
                "model_label",
                "model_group",
                "model_name",
                "model_variant",
                "architecture",
                "execution_target",
                "decoding",
            )
        )
        searchable = f"{searchable} {_public_scoring_route(row)}".lower()
        model_name = _safe_link(
            row.get("official_model_url"), row.get("model_label")
        )
        rendered.append(
            f"""<tr data-search="{_text(searchable)}">
  <td class="rank rank-sticky" data-sort="{_text(row.get('rank'), '')}">{_text(row.get('rank'))}</td>
  <td class="configuration configuration-sticky" data-sort="{_text(row.get('model_label'), '').lower()}">
    <span class="group">{_text(row.get('model_group'))}</span>
    <strong>{model_name}</strong>
    {_detail_block(row)}
  </td>
  <td class="metric" data-sort="{_text(row.get('global_per'), '')}">{_text(_percent(row.get('global_per')))}</td>
  <td class="metric secondary" data-sort="{_text(row.get('global_mer'), '')}">{_text(_percent(row.get('global_mer')))}</td>
  {_task_cells(row)}
  <td data-sort="{_text(row.get('architecture'), '').lower()}">{_text(row.get('architecture'))}</td>
  <td data-sort="{_text(row.get('execution_target'), '').lower()}">{_text(row.get('execution_target'))}</td>
  <td data-sort="{_text(row.get('decoding'), '').lower()}">{_text(row.get('decoding'))}</td>
  <td class="date" data-sort="{_text(row.get('completed_at'), '')}">{_text(_date_label(row.get('completed_at')))}</td>
</tr>"""
        )
    return "\n".join(rendered)


def _table(rows: list[dict[str, str]]) -> str:
    grouped_headers = "".join(
        f'<th class="group-head" colspan="3">{_text(label)}</th>'
        for _prefix, label, _outcome_kind in TASKS
    )
    task_headers: list[str] = []
    column = 4
    for _prefix, _label, outcome_kind in TASKS:
        task_headers.extend(
            (
                f'<th class="task-start"><button data-column="{column}" data-type="number">PER</button></th>',
                f'<th><button data-column="{column + 1}" data-type="number">MER</button></th>',
                f'<th><button data-column="{column + 2}" data-type="number">{"r" if outcome_kind == "corr" else "Accuracy"}</button></th>',
            )
        )
        column += 3
    return f"""<div class="table-heading">
  <div><h2>IPA ranking</h2><p>Ranked by overall phoneme error rate (PER).</p></div>
  <span class="result-count" data-total="{len(rows)}">{len(rows)} results</span>
</div>
<div class="table-wrap">
  <table class="ranking-table">
    <thead>
      <tr>
        <th class="rank-sticky" rowspan="2"><button data-column="0" data-type="number">Rank</button></th>
        <th class="configuration-sticky" rowspan="2"><button data-column="1" data-type="text">Configuration</button></th>
        <th class="group-head" colspan="2">Overall</th>
        {grouped_headers}
        <th class="group-head" colspan="4">Run context</th>
      </tr>
      <tr>
        <th><button data-column="2" data-type="number">PER</button></th>
        <th><button data-column="3" data-type="number">MER</button></th>
        {''.join(task_headers)}
        <th><button data-column="25" data-type="text">Architecture</button></th>
        <th><button data-column="26" data-type="text">Execution target</button></th>
        <th><button data-column="27" data-type="text">Decoder</button></th>
        <th><button data-column="28" data-type="text">Completed</button></th>
      </tr>
    </thead>
    <tbody>{_table_rows(rows)}</tbody>
  </table>
</div>
<p class="empty" hidden>No results match the current search.</p>"""


def _document(
    rows: list[dict[str, str]],
    generated_at: object,
) -> str:
    first = rows[0]
    system_name = _public_system_name(first)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kiswahili ASR IPA leaderboard — {_text(system_name)}</title>
  <style>
    :root {{ --ink:#172033; --muted:#5d6675; --paper:#fff; --wash:#f3f6f8; --navy:#173b57; --teal:#087f78; --teal-soft:#e9f7f5; --line:#d8e0e6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--wash); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
    .page {{ width:min(1480px,calc(100% - 32px)); margin:28px auto; background:var(--paper); box-shadow:0 12px 36px rgba(23,59,87,.12); }}
    header {{ padding:48px 56px 40px; color:#fff; background:linear-gradient(125deg,var(--navy),#155b70 68%,var(--teal)); }}
    .eyebrow {{ margin:0 0 10px; color:#bfece5; font-size:.76rem; font-weight:760; letter-spacing:.13em; text-transform:uppercase; }}
    h1 {{ max-width:1000px; margin:0; font-size:clamp(2.1rem,4vw,3.5rem); line-height:1.06; }}
    header .lead {{ max-width:950px; margin:18px 0 0; color:#e8f4f5; font-size:1.04rem; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:22px; color:#d7eef0; font-size:.86rem; }}
    main {{ padding:34px 40px 54px; }}
    .controls {{ display:flex; align-items:end; justify-content:flex-end; margin-bottom:22px; }}
    .search {{ display:grid; gap:5px; width:min(480px,100%); color:var(--muted); font-size:.82rem; font-weight:700; }}
    .search input {{ width:100%; padding:11px 13px; color:var(--ink); background:#fff; border:1px solid #aeb9c2; border-radius:7px; font:inherit; }}
    .search input:focus {{ outline:3px solid rgba(8,127,120,.2); border-color:var(--teal); }}
    .guide {{ margin:0 0 30px; border:1px solid var(--line); border-radius:8px; background:#f7fafb; }}
    .guide summary {{ padding:14px 17px; color:var(--navy); font-weight:800; cursor:pointer; }}
    .guide[open] summary {{ border-bottom:1px solid var(--line); }}
    .guide div {{ padding:4px 20px 14px; color:var(--muted); }}
    .guide li {{ margin:10px 0; }}
    .guide strong {{ color:var(--navy); }}
    .table-heading {{ display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:12px; }}
    h2 {{ margin:0; color:var(--navy); font-size:1.55rem; }}
    .table-heading p {{ margin:4px 0 0; color:var(--muted); }}
    .result-count {{ padding:5px 10px; color:var(--teal); background:var(--teal-soft); border-radius:999px; font-size:.8rem; font-weight:760; white-space:nowrap; }}
    .table-wrap {{ overflow:auto; max-height:72vh; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; min-width:3600px; border-collapse:separate; border-spacing:0; }}
    thead {{ position:sticky; top:0; z-index:2; }}
    th {{ padding:0; color:#fff; background:var(--navy); border-right:1px solid rgba(255,255,255,.18); text-align:left; }}
    th.group-head {{ padding:9px 11px; text-align:center; background:#155b70; font-size:.75rem; letter-spacing:.03em; }}
    th button {{ width:100%; padding:12px 11px; color:inherit; background:transparent; border:0; font:inherit; font-size:.76rem; font-weight:780; text-align:left; cursor:pointer; }}
    th button::after {{ content:" ↕"; color:#bfece5; }}
    th button[data-direction="asc"]::after {{ content:" ↑"; }}
    th button[data-direction="desc"]::after {{ content:" ↓"; }}
    td {{ padding:12px 11px; vertical-align:top; border-bottom:1px solid var(--line); }}
    tbody tr:nth-child(even) {{ background:#fafcfd; }}
    tbody tr:hover {{ background:#edf9f6; }}
    a {{ color:var(--navy); }}
    .rank,.metric,.result,.date {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
    .rank {{ width:58px; min-width:58px; text-align:center; font-weight:760; }}
    .metric {{ color:var(--navy); font-size:1.02rem; font-weight:820; }}
    .metric.secondary {{ color:var(--teal); font-size:.92rem; }}
    .result {{ text-align:right; }}
    .task-start {{ border-left:2px solid #a8c7cf; }}
    .configuration {{ width:410px; min-width:410px; }}
    .rank-sticky,.configuration-sticky {{ position:sticky; z-index:3; background:#fff; }}
    .rank-sticky {{ left:0; }}
    .configuration-sticky {{ left:58px; box-shadow:5px 0 8px rgba(23,59,87,.08); }}
    thead .rank-sticky,thead .configuration-sticky {{ z-index:5; background:var(--navy); }}
    tbody tr:nth-child(even) .rank-sticky,tbody tr:nth-child(even) .configuration-sticky {{ background:#fafcfd; }}
    tbody tr:hover .rank-sticky,tbody tr:hover .configuration-sticky {{ background:#edf9f6; }}
    .configuration strong {{ display:block; margin:3px 0 4px; line-height:1.35; }}
    .group {{ display:inline-block; padding:2px 7px; color:var(--teal); background:var(--teal-soft); border-radius:999px; font-size:.7rem; font-weight:780; }}
    .context {{ margin-top:7px; color:var(--muted); font-size:.78rem; }}
    .context summary {{ color:var(--teal); font-weight:750; cursor:pointer; }}
    .context dl {{ display:grid; grid-template-columns:110px 1fr; gap:5px 10px; margin:8px 0 0; padding:10px; background:#f7fafb; border-radius:6px; }}
    .context dt {{ color:var(--navy); font-weight:740; }}
    .context dd {{ margin:0; }}
    .empty {{ padding:30px; color:var(--muted); text-align:center; }}
    [hidden] {{ display:none !important; }}
    @media (max-width:800px) {{ header {{ padding:38px 24px 32px; }} main {{ padding:28px 18px 44px; }} .controls {{ justify-content:stretch; }} .search {{ width:100%; }} }}
    @media print {{ body {{ background:#fff; font-size:9pt; }} .page {{ width:100%; margin:0; box-shadow:none; }} header {{ padding:28px 36px; print-color-adjust:exact; -webkit-print-color-adjust:exact; }} main {{ padding:24px 30px; }} .controls,.context {{ display:none !important; }} .table-wrap {{ overflow:visible; max-height:none; }} table {{ min-width:0; }} thead {{ position:static; }} th {{ print-color-adjust:exact; -webkit-print-color-adjust:exact; }} td {{ padding:7px; }} .configuration {{ min-width:0; }} }}
  </style>
</head>
<body>
  <article class="page">
    <header>
      <p class="eyebrow">AI for Education | Voice AI Evaluation Framework</p>
      <h1>Kiswahili ASR IPA leaderboard</h1>
      <p class="lead">A standalone ranking scored consistently against {_text(system_name)} IPA references.</p>
      <div class="meta"><span><strong>Scoring system:</strong> {_text(system_name)}</span><span><strong>Evidence generated:</strong> {_text(_generated_label(generated_at))}</span><span><strong>Configurations:</strong> {len(rows)}</span></div>
    </header>
    <main>
      <div class="controls"><label class="search">Search configurations<input id="search" type="search" placeholder="Model, architecture, target, decoder, or scoring route" autocomplete="off"></label></div>
      <details class="guide">
        <summary>How to read the scores</summary>
        <div><ul>
          <li><strong>PER</strong> is phoneme error rate: substitutions, deletions, and insertions divided by the number of reference phonemes. Lower is better.</li>
          <li><strong>MER</strong> compares learner-mistake sequences derived from the canonical prompt, human reference, and model hypothesis. Lower is better.</li>
          <li><strong>Correlation (r)</strong> measures agreement on correct-item counts for grid tasks; <strong>accuracy</strong> measures item-level correct/mistake agreement for isolated tasks. Higher is better.</li>
          <li>Error rates can exceed 100% when insertions outnumber the reference units.</li>
        </ul></div>
      </details>
      {_table(rows)}
    </main>
  </article>
  <script>
    const search = document.querySelector('#search');
    const body = document.querySelector('.ranking-table > tbody');
    const count = document.querySelector('.result-count');
    const empty = document.querySelector('.empty');
    search.addEventListener('input', () => {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      [...body.rows].forEach((row) => {{
        const show = !query || row.dataset.search.includes(query);
        row.hidden = !show;
        if (show) visible += 1;
      }});
      count.textContent = `${{visible}} results`;
      empty.hidden = visible !== 0;
    }});
    document.querySelectorAll('th button').forEach((button) => button.addEventListener('click', () => {{
      const direction = button.dataset.direction === 'asc' ? 'desc' : 'asc';
      document.querySelectorAll('th button').forEach((item) => item.removeAttribute('data-direction'));
      button.dataset.direction = direction;
      const column = Number(button.dataset.column);
      const factor = direction === 'asc' ? 1 : -1;
      const rows = [...body.rows].map((row, index) => ({{row, index}}));
      rows.sort((left, right) => {{
        const a = left.row.cells[column].dataset.sort || left.row.cells[column].textContent.trim();
        const b = right.row.cells[column].dataset.sort || right.row.cells[column].textContent.trim();
        let comparison;
        if (button.dataset.type === 'number') {{
          const aNumber = Number.parseFloat(a);
          const bNumber = Number.parseFloat(b);
          if (Number.isNaN(aNumber) && Number.isNaN(bNumber)) comparison = 0;
          else if (Number.isNaN(aNumber)) comparison = 1;
          else if (Number.isNaN(bNumber)) comparison = -1;
          else comparison = (aNumber - bNumber) * factor;
        }} else {{
          comparison = a.localeCompare(b) * factor;
        }}
        return comparison || left.index - right.index;
      }});
      rows.forEach((item) => body.appendChild(item.row));
    }}));
  </script>
</body>
</html>
"""


def build_html(
    leaderboards_root: Path, g2p_system_id: str, output: Path | None = None
) -> Path:
    metadata_path = leaderboards_root / "leaderboard_metadata.json"
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not has_required_eligibility_policy(metadata):
        raise ValueError(
            "Leaderboard metadata does not guarantee the required eligibility policy"
        )
    board = (
        metadata.get("leaderboards", {})
        .get("ipa_by_system", {})
        .get(g2p_system_id)
    )
    if not isinstance(board, dict):
        raise ValueError(f"G2P system not found in leaderboard metadata: {g2p_system_id}")
    csv_path = leaderboards_root / "ipa" / f"leaderboard_{g2p_system_id}.csv"
    rows = _load_rows(csv_path)
    if rows[0]["g2p_system_id"] != g2p_system_id:
        raise ValueError("Leaderboard metadata and CSV G2P identities do not match")
    expected_rows = board.get("rows")
    if expected_rows is not None and int(expected_rows) != len(rows):
        raise ValueError("Leaderboard metadata and CSV row counts do not match")
    output_path = output or csv_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        _document(
            rows,
            metadata.get("generated_at"),
        ),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaderboards-root",
        type=Path,
        default=Path("input_output_data/output/leaderboards"),
    )
    parser.add_argument("--g2p-system-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = build_html(
        args.leaderboards_root, args.g2p_system_id, output=args.output
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
