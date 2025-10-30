import argparse
import csv
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Explain Gallery</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    .row { display: flex; gap: 8px; }
    .col { flex: 1; }
    img { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }
    .meta { font-size: 12px; color: #555; margin-top: 6px; word-break: break-all; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .header h2 { margin: 0; font-size: 18px; }
    .badge { background: #eef; color: #223; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
  </style>
  </head>
<body>
  <div class="header">
    <h2>Explain Gallery</h2>
    <div class="badge">Total items: {{COUNT}}</div>
  </div>
  <div class="grid">
    {{CARDS}}
  </div>
</body>
</html>
"""


def build_card(overlay_path: str, composite_path: str, original_path: str, predicted: str, probs: list[str]) -> str:
    prob_text = ", ".join([f"p{i}={p}" for i, p in enumerate(probs)]) if probs else ""
    return f"""
    <div class=\"card\">
      <div class=\"row\">
        <div class=\"col\">
          <div class=\"meta\">Composite</div>
          <img src=\"{composite_path}\" alt=\"composite\" />
        </div>
        <div class=\"col\">
          <div class=\"meta\">Overlay</div>
          <img src=\"{overlay_path}\" alt=\"overlay\" />
        </div>
      </div>
      <div class=\"meta\"><strong>Predicted:</strong> {predicted} &nbsp; <strong>Probs:</strong> {prob_text}</div>
      <div class=\"meta\"><strong>Original:</strong> {original_path}</div>
    </div>
    """


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain_dir", type=str, default="/teamspace/studios/this_studio/artifacts/explain")
    parser.add_argument("--limit", type=int, default=0, help="optional limit of items")
    args = parser.parse_args()

    explain_dir = Path(args.explain_dir)
    csv_path = explain_dir / "occlusion_index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    cards = []
    for row in rows:
        overlay = Path(row["overlay_path"]).name
        composite = Path(row["composite_path"]).name
        original = row["original_path"]
        pred = row["predicted"]
        probs = [row.get("proba_0", ""), row.get("proba_1", ""), row.get("proba_2", "")]
        # Use relative paths so the HTML can be opened directly
        cards.append(build_card(overlay, composite, original, pred, probs))

    html = TEMPLATE.replace("{{CARDS}}", "\n".join(cards)).replace("{{COUNT}}", str(len(rows)))
    out_path = explain_dir / "index.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()



