"""Merge the source data of every panel into Excel files of less than 30 MB, one sheet per figure (editor's request, Nature Communications)

author: laquitainesteeve@gmail.com

Input: figures/8_source_data/<panel>/*.xlsx (e.g. fig2a/fig2a.xlsx, supp_fig6c/supp_fig6c.xlsx), written by the notebooks of
notebooks/2_results_editorial_requests/. Output (the files requested by the editor, each below the 30 MB limit):
paper/Source_Data_1.xlsx: sheets "Fig. 2", ..., "Supplementary Fig. 9" except Supplementary Fig. 2;
paper/Source_Data_2.xlsx and paper/Source_Data_3.xlsx: the sheet "Supplementary Fig. 2" (926,000 rows, 40 MB), split by panel (a-c and d-g).
In each sheet, the panels follow each other from top to bottom: a title row ("Fig. 2a", followed by the file and the sheet when a panel has
several tables), the column names, the data and an empty row.

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/merge_source_data.py
"""

import glob
import math
import os
import re
import time

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font

SOURCE_DIR = "figures/8_source_data"
OUTPUTS = {1: os.path.join("paper", "Source_Data_1.xlsx"),    # the Source Data files requested by the editor, next to the manuscript
           2: os.path.join("paper", "Source_Data_2.xlsx"),
           3: os.path.join("paper", "Source_Data_3.xlsx")}
SPLIT_SHEET = ("supp_fig", 2)                                  # sheet too large for one file (40 MB): panels a-c in file 2, d-g in file 3
SPLIT_PANELS = {"a": 2, "b": 2, "c": 2, "d": 3, "e": 3, "f": 3, "g": 3}
MAX_ROWS = 1_048_576  # rows of an Excel sheet


def figure_of(panel: str):
    """('fig', 2, 'a') for fig2a, ('supp_fig', 6, 'c') for supp_fig6c"""
    m = re.fullmatch(r"(supp_fig|fig)(\d+)([a-z]?)", panel)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def sheet_name(kind: str, number: int) -> str:
    return f"Fig. {number}" if kind == "fig" else f"Supplementary Fig. {number}"


def panel_title(kind: str, number: int, letter: str) -> str:
    return sheet_name(kind, number) + letter


def clean(value):
    """Excel cannot store NaN or infinite values: write an empty cell"""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_book(output: str, selection: dict, t0: float):
    """Write one workbook: selection maps (kind, number) to a list of (letter, files)"""
    wb = Workbook(write_only=True)
    bold = Font(bold=True)
    for kind, number in sorted(selection, key=lambda k: (k[0] == "supp_fig", k[1])):
        ws = wb.create_sheet(sheet_name(kind, number))
        n_rows = 0
        for letter, files in sorted(selection[(kind, number)]):
            title = panel_title(kind, number, letter)
            tables = [(f, s, d) for f in files for s, d in pd.read_excel(f, sheet_name=None).items()]
            for f, s, d in tables:
                label = title if len(tables) == 1 else f"{title} ({os.path.basename(f)}, sheet {s})"
                cell = WriteOnlyCell(ws, value=label)
                cell.font = bold
                ws.append([cell])
                ws.append([str(c) for c in d.columns])
                for row in d.itertuples(index=False):
                    ws.append([clean(v) for v in row])
                ws.append([])
                n_rows += len(d) + 3
        assert n_rows <= MAX_ROWS, f"{sheet_name(kind, number)} has {n_rows} rows, more than an Excel sheet can hold"
        print(f"{sheet_name(kind, number)}: {len(selection[(kind, number)])} panels, {n_rows:,} rows ({time.time() - t0:.0f} s)", flush=True)
    wb.save(output)
    print(f"saved {output} ({os.path.getsize(output) / 1e6:.1f} MB) in {time.time() - t0:.0f} s", flush=True)


def main():
    t0 = time.time()
    panels = {}
    for folder in sorted(glob.glob(os.path.join(SOURCE_DIR, "*"))):
        key = figure_of(os.path.basename(folder))
        if key is None or not os.path.isdir(folder):
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
        if files:
            panels.setdefault(key[:2], []).append((key[2], files))

    books = {1: {k: v for k, v in panels.items() if k != SPLIT_SHEET}, 2: {}, 3: {}}
    for letter, files in panels[SPLIT_SHEET]:
        books[SPLIT_PANELS[letter]].setdefault(SPLIT_SHEET, []).append((letter, files))
    for n, selection in books.items():
        write_book(OUTPUTS[n], selection, t0)


if __name__ == "__main__":
    main()
