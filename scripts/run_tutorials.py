"""Run the tutorials (examples/*.py, jupytext percent scripts) as notebooks and render them.

    python scripts/run_tutorials.py OUTDIR [NAME ...]

Writes OUTDIR/NAME.ipynb with every output, OUTDIR/NAME.html and OUTDIR/index.html (the site's
tutorials page: each notebook's title and introduction, linked). The notebooks run in OUTDIR
and import zebsdsolve the usual way: the installed wheel, or python/ on PYTHONPATH in a checkout.
A failing cell (each tutorial ends with checks against known answers) fails the run.

Needs jupytext, nbconvert, ipykernel, matplotlib and scipy.
"""
import html
import os
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>zebsdsolve tutorials</title>
<style>
  :root {{ --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --line: #e3e3e6; --accent: #0b63ce; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #141416; --fg: #ececef; --muted: #a0a3a8; --line: #2c2c31; --accent: #6aa8ff; }} }}
  body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 760px; margin: 0 auto; padding: 40px 16px 64px; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 4px; }}
  .lead {{ color: var(--muted); margin: 0 0 28px; }}
  a {{ color: var(--accent); }}
  article {{ border-top: 1px solid var(--line); padding: 20px 0; }}
  article h2 {{ font-size: 1.15rem; margin: 0 0 6px; }}
  article p {{ margin: 0 0 8px; color: var(--muted); }}
  .links a {{ margin-right: 16px; }}
</style>
</head>
<body>
<main>
<h1>zebsdsolve tutorials</h1>
<p class="lead">Run on a GitHub M1 runner against the released wheel. <a href="../">Open the app</a>.</p>
{items}
</main>
</body>
</html>
"""


def describe(nb):
    """A notebook's title (its first heading) and introduction (the first paragraph after it)."""
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        lines = cell.source.splitlines()
        title = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), None)
        if title:
            body = "\n".join(l for l in lines if not l.startswith("#")).strip()
            return title, body.split("\n\n")[0].replace("\n", " ")
    return None, ""


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[2:])
    sources = [p for p in sorted((ROOT / "examples").glob("*.py")) if not wanted or p.stem in wanted]
    items = []
    for src in sources:
        t = time.perf_counter()
        nb = jupytext.read(src)
        # the notebooks run in OUTDIR, which need not be inside the repository: say where its data is
        os.environ.setdefault("ZEBSDSOLVE_TESTDATA", str(ROOT / "testdata"))
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out)}})
        nbformat.write(nb, out / f"{src.stem}.ipynb")
        page, _ = HTMLExporter().from_notebook_node(nb)
        (out / f"{src.stem}.html").write_text(page, encoding="utf-8")
        title, intro = describe(nb)
        items.append(
            f'<article><h2>{html.escape(title or src.stem)}</h2><p>{html.escape(intro)}</p>'
            f'<p class="links"><a href="{src.stem}.html">Read</a><a href="{src.stem}.ipynb" download>Download the notebook</a></p></article>'
        )
        print(f"{src.stem}: {time.perf_counter() - t:.1f} s", flush=True)
    (out / "index.html").write_text(PAGE.format(items="\n".join(items)), encoding="utf-8")


if __name__ == "__main__":
    main()
