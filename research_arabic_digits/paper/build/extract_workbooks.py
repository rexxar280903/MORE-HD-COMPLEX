"""Ekstrak sheet ringkas dari workbook konsolidasi MADBase ke CSV untuk build paper.

Sumber (read-only): research_arabic_digits/research_data/consolidated/*.xlsx
Keluaran         : research_arabic_digits/paper/build/data/<track>__<sheet>.csv
Sheet raksasa (03/04 history, 11 test predictions) sengaja dilewati.
"""
import csv
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
RD = HERE.parents[1] / "research_data"
OUT = HERE / "data"
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
BOOKS = {"primary": "primary_240runs.xlsx", "ablation": "ablation_90runs.xlsx", "more": "more_reference_120runs.xlsx"}
SKIP = {"03_Clustering_History", "04_Supervised_History", "11_Test_Predictions"}


def col_index(ref):
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def targets(book):
    wb = ET.fromstring(book.read("xl/workbook.xml"))
    rels = {n.attrib["Id"]: n.attrib["Target"] for n in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for node in wb.find(f"{{{NS_MAIN}}}sheets"):
        t = rels[node.attrib[f"{{{NS_REL}}}id"]].lstrip("/")
        out[node.attrib["name"]] = t if t.startswith("xl/") else "xl/" + t
    return out


def shared_strings(book):
    try:
        root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")) for si in root]


def dump(book, target, sst, path, header_row=3):
    headers = None
    n = 0
    with book.open(target) as stream, path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for _e, row in ET.iterparse(stream, events=("end",)):
            if not row.tag.endswith("}row"):
                continue
            r = int(row.attrib.get("r", "0") or 0)
            cells = {}
            for c in row:
                if not c.tag.endswith("}c"):
                    continue
                t = c.attrib.get("t")
                if t == "inlineStr":
                    v = "".join(x.text or "" for x in c.iter() if x.tag.endswith("}t"))
                else:
                    vn = next((x for x in c if x.tag.endswith("}v")), None)
                    v = "" if vn is None else (vn.text or "")
                    if t == "s" and v:
                        v = sst[int(v)]
                cells[col_index(c.attrib.get("r", "A1"))] = v
            if r == header_row:
                headers = cells
                w.writerow([headers[i] for i in sorted(headers)])
            elif r > header_row and headers:
                vals = [cells.get(i, "") for i in sorted(headers)]
                if any(vals):
                    w.writerow(vals)
                    n += 1
            row.clear()
    return n


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for key, name in BOOKS.items():
        with zipfile.ZipFile(RD / "consolidated" / name) as book:
            sst = shared_strings(book)
            for sheet, target in targets(book).items():
                if sheet in SKIP or (only and sheet not in only):
                    continue
                hdr = 1 if sheet.startswith("00_") or sheet == "99_Consolidation_Log" else 3
                n = dump(book, target, sst, OUT / f"{key}__{sheet}.csv", hdr)
                print(key, sheet, n, flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
