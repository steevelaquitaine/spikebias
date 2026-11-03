"""Replot figure panels from their saved source data, with the Nature figure guidelines (src/nodes/figure_style.py)

author: laquitainesteeve@gmail.com

For each panel notebook of notebooks/2_results_editorial_requests/ that follows the source data template (a `plot_source_data` function, the
source data saved to figures/8_source_data/<panel>/<panel>.xlsx and a cell that replots the panel from the reloaded xlsx), this script builds
and runs a short notebook with:
* the notebook's imports, constants and function definitions (no other statement, so that no analysis is re-run),
* `figure_style.apply()` (Arial, 5-7 pt, editable text, 300 dpi, a .pdf next to each .svg),
* the notebook's cells that reload the source data and replot the panel,
* the notebook's own savefig statement of the panel's .svg.
The figures therefore show exactly the data of the Source Data file. The analyses themselves are not re-run.

Usage (all template notebooks, or the panels given):

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/figures/replot_from_source_data.py [fig5b supp_fig7a ...]
"""

import argparse
import ast
import glob
import os
import re
import sys
import time

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = "notebooks/2_results_editorial_requests"
# calls that load or compute data: an assignment that makes one of them is not kept
HEAVY_CALLS = re.compile(r"load|read|compute|extract|fit|evaluate|train|run|simulate|save|^to_|download|slice|traces|predict|analyz|wave|pca|sort|"
                         r"get_config|Client|query|merge|concat|groupby|apply|subprocess|system", re.IGNORECASE)
SAFE_EXPR_CALLS = ("os.chdir", "chdir", "rcParams.update", "rc", "set_theme", "set_style", "set_context", "simplefilter", "filterwarnings",
                   "makedirs", "use", "sys.path.append", "sys.path.insert", "setrecursionlimit")


def _calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _name(call) -> str:
    f = call.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def keep_definitions(source: str) -> str:
    """Imports (in try/except), function and class definitions, constant assignments and setup calls of a cell"""
    lines = [l for l in source.split("\n") if not l.lstrip().startswith(("%", "!"))]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return ""
    kept = []
    for node in tree.body:
        seg = ast.get_source_segment("\n".join(lines), node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept.append("try:\n" + "\n".join("    " + l for l in seg.split("\n")) + "\nexcept Exception as e:\n    print('skipped import:', e)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kept.append(seg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if not any(HEAVY_CALLS.search(_name(c).split(".")[-1]) for c in _calls(node)):
                # constants may depend on data that is not loaded here: skip them if they fail
                kept.append("try:\n" + "\n".join("    " + l for l in seg.split("\n")) + "\nexcept Exception:\n    pass")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _name(node.value).endswith(SAFE_EXPR_CALLS):
                kept.append(seg)
    return "\n".join(kept)


def build(panel: str):
    """The replot notebook of a panel, or None if the notebook does not follow the template"""
    nb = nbformat.read(os.path.join(NOTEBOOKS, panel + ".ipynb"), 4)
    cells = [c.source for c in nb.cells if c.cell_type == "code"]
    if not any("def plot_source_data" in c for c in cells):
        return None
    reload_ix = [i for i, c in enumerate(cells) if "read_excel" in c]
    if not reload_ix:
        return None
    first_reload = reload_ix[-1]
    while first_reload - 1 in reload_ix:
        first_reload -= 1
    replot = [c for c in cells[first_reload:] if "read_excel" in c or ("plot_source_data(" in c and "def plot_source_data" not in c)]
    if not any("plot_source_data(" in c for c in replot):
        return None
    save = [l.strip() for c in cells for l in c.split("\n") if "savefig(" in l and ".svg" in l and not l.strip().startswith("#")]
    if not save:
        return None
    save = re.sub(r"^\s*(plt|fig|figure|ax\.figure)\.savefig\(", "fig.savefig(", save[0])
    defs = [keep_definitions(c) for c in cells[:first_reload]]
    setup = "from src.nodes import figure_style\nfigure_style.apply()"
    body = [d for d in defs if d.strip()]
    chdir = next((i for i, d in enumerate(body) if "chdir" in d), -1)
    body.insert(chdir + 1, setup)
    body += [re.sub(r"^%%time\n", "", c) for c in replot] + [save]
    out = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(b) for b in body])
    out.metadata = nb.metadata
    return out


def main(panels):
    sys.path.insert(0, os.getcwd())
    results = []
    for panel in panels:
        t0 = time.time()
        nb = build(panel)
        if nb is None:
            results.append((panel, "not a template notebook", 0))
            continue
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
            status = "ok"
        except Exception as e:
            status = "FAILED: " + str(e).strip().split("\n")[-1][:200]
        results.append((panel, status, round(time.time() - t0)))
        print(f"{panel}: {status} ({results[-1][2]} s)", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("panels", nargs="*", help="panel notebooks (default: all)")
    args = parser.parse_args()
    panels = args.panels or sorted(os.path.basename(f)[:-6] for f in glob.glob(os.path.join(NOTEBOOKS, "*fig*.ipynb")))
    main(panels)
