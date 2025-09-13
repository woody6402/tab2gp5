#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_extractor.py — Extrahiere ASCII-Gitarren-TAB aus vektorbasierter PDF-Seite.

Voraussetzungen:
  pip install pymupdf numpy

Beispiele:
  python tab_extractor.py --pdf summertime.pdf --page 1 > page1_tab.txt
  python tab_extractor.py --pdf summertime.pdf --page 2 --cols 220 > page2_tab.txt
"""

import sys
import re
import argparse
from typing import List, Tuple, Dict
import numpy as np

try:
    import fitz  # PyMuPDF
except Exception:
    print("Fehler: PyMuPDF (fitz) ist nicht installiert. Bitte: pip install pymupdf", file=sys.stderr)
    sys.exit(2)


def cluster_verticals_by_x(segments: List[Tuple[float, float, float]], tol: float = 0.9) -> Dict[float, List[Tuple[float, float]]]:
    """Gruppiere vertikale Liniensegmente nach X (um fragmentierte Taktstriche zusammenzuführen)."""
    if not segments:
        return {}
    xs = sorted([x for (x, y0, y1) in segments])
    clusters: List[List[float]] = []
    for x in xs:
        if not clusters or abs(x - clusters[-1][-1]) > tol:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    cluster_x = [float(np.mean(c)) for c in clusters]
    x_to_segs: Dict[float, List[Tuple[float, float]]] = {cx: [] for cx in cluster_x}
    for (x, y0, y1) in segments:
        cx = min(cluster_x, key=lambda c: abs(c - x))
        if abs(cx - x) <= tol:
            x_to_segs[cx].append((y0, y1))
    return x_to_segs


def pick_string_ys(band_digits: List[Dict[str, float]], horizontals: List[Tuple[float, float, float]]) -> List[float]:
    """
    Bestimme die sechs Saiten-Y-Positionen pro System.
    1) Nutze horizontale Linien aus dem PDF in der vertikalen Nähe der Ziffern.
    2) Fallback: 6 Quantile der Ziffer-Ys (robust, falls Linien nicht verfügbar).
    """
    y_min = min(d['y'] for d in band_digits) - 25
    y_max = max(d['y'] for d in band_digits) + 25
    cand = [y for (y, x0, x1) in horizontals if y_min <= y <= y_max]
    cand = sorted(cand)
    uniq: List[float] = []
    for y in cand:
        if not uniq or abs(y - uniq[-1]) > 1.0:
            uniq.append(y)
    if len(uniq) >= 6:
        best_group = uniq[:6]
        best_score = 1e9
        for i in range(0, len(uniq) - 5):
            group = uniq[i:i + 6]
            diffs = [group[j + 1] - group[j] for j in range(5)]
            score = max(diffs) - min(diffs)
            if score < best_score:
                best_score = score
                best_group = group
        return best_group
    # Fallback
    y_vals = np.array([d['y'] for d in band_digits], dtype=float)
    centers = np.quantile(y_vals, np.linspace(0, 1, 8)[1:-1])
    return sorted(centers.tolist())


def extract_tab_from_pdf_page(
    pdf_path: str,
    page_index_0_based: int,
    cols: int = 240,
    gap_threshold: float = 40.0,
    vert_tol: float = 0.9,
    coverage_ratio: float = 0.6,
    min_band_digits: int = 6
) -> str:
    """
    Extrahiere ASCII-TAB aus einer vektor-basierten PDF-Seite.
    Gibt den ASCII-Text zurück (mehrere Systeme durch Leerzeile getrennt).
    """
    doc = fitz.open(pdf_path)
    if page_index_0_based < 0 or page_index_0_based >= len(doc):
        raise IndexError(f"Seitenindex außerhalb des Bereichs (0..{len(doc)-1}).")

    page = doc[page_index_0_based]

    # 1) Ziffern (TAB-Zahlen) aus der Textschicht holen
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word_no)
    digits = []
    for (x0, y0, x1, y1, txt, block, line, wno) in words:
        if re.fullmatch(r"\d+", txt):
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            digits.append({'x': cx, 'y': cy, 'txt': txt})
    if not digits:
        return ""

    # 2) In horizontale Bänder (Systeme) aufteilen
    digits_sorted = sorted(digits, key=lambda d: d['y'])
    ys = [d['y'] for d in digits_sorted]
    gaps = np.diff(ys)
    cut_idxs = [i for i, g in enumerate(gaps) if g > gap_threshold]
    bands: List[List[Dict[str, float]]] = []
    start = 0
    for idx in cut_idxs + [len(digits_sorted) - 1]:
        band = digits_sorted[start:idx + 1]
        if len(band) >= min_band_digits:
            bands.append(band)
        start = idx + 1

    # 3) Vektor-Linien extrahieren (horizontal/vertikal)
    draws = page.get_drawings()
    vertical_segments: List[Tuple[float, float, float]] = []
    horizontal_segments: List[Tuple[float, float, float]] = []

    def add_line(x0: float, y0: float, x1: float, y1: float):
        if abs(x1 - x0) < 0.6 and abs(y1 - y0) > 8:  # vertikal
            ylo, yhi = (y0, y1) if y0 < y1 else (y1, y0)
            vertical_segments.append((x0, ylo, yhi))
        elif abs(y1 - y0) < 0.6 and abs(x1 - x0) > 8:  # horizontal
            xlo, xhi = (x0, x1) if x0 < x1 else (x1, x0)
            horizontal_segments.append((y0, xlo, xhi))

    for d in draws:
        for item in d["items"]:
            kind = item[0]
            if kind == "l":
                _, p0, p1 = item
                add_line(p0[0], p0[1], p1[0], p1[1])
            elif kind == "re":
                _, (x, y, w, h), _ = item
                add_line(x, y, x + w, y)
                add_line(x, y + h, x + w, y + h)
                add_line(x, y, x, y + h)
                add_line(x + w, y, x + w, y + h)

    x_to_segs = cluster_verticals_by_x(vertical_segments, tol=vert_tol)

    # 4) ASCII je Band bauen
    def make_ascii_for_band(band_digits: List[Dict[str, float]], cols_local: int = cols) -> str:
        s_ys = pick_string_ys(band_digits, horizontal_segments)
        s_top, s_bot = min(s_ys), max(s_ys)

        # Taktstriche: nur wenn sie den Großteil der Systemhöhe abdecken
        bar_xs: List[float] = []
        for cx, segs in x_to_segs.items():
            coverage = 0.0
            for (y0, y1) in segs:
                lo = max(y0, s_top - 2)
                hi = min(y1, s_bot + 2)
                if hi > lo:
                    coverage += (hi - lo)
            if coverage >= (s_bot - s_top) * coverage_ratio:
                bar_xs.append(cx)
        bar_xs = sorted(bar_xs)

        xs_digits = [d['x'] for d in band_digits]
        min_x = min(xs_digits + bar_xs) - 4 if bar_xs else min(xs_digits) - 4
        max_x = max(xs_digits + bar_xs) + 4 if bar_xs else max(xs_digits) + 4
        scale = (cols_local - 1) / (max_x - min_x) if max_x > min_x else 1.0

        grid = [["-"] * cols_local for _ in range(6)]

        # Barlines zuerst setzen (damit Zahlen sichtbar bleiben)
        for bx in bar_xs:
            c = int((bx - min_x) * scale)
            if 0 <= c < cols_local:
                for s in range(6):
                    grid[s][c] = "|"

        # Ziffern auf nächste Saite mappen
        for d in band_digits:
            s = int(np.argmin([abs(d['y'] - y) for y in s_ys]))
            c = int((d['x'] - min_x) * scale)
            for k, ch in enumerate(d['txt']):
                cc = c + k
                if 0 <= cc < cols_local:
                    grid[s][cc] = ch

        names = ["e", "B", "G", "D", "A", "E"]
        lines = [f"{names[i]}|{''.join(grid[i])}|" for i in range(6)]
        return "\n".join(lines)

    ascii_parts = [make_ascii_for_band(b) for b in bands]
    return "\n\n".join(ascii_parts)


def main():
    ap = argparse.ArgumentParser(description="Extrahiere ASCII-Gitarren-TAB aus einer vektor-basierten PDF-Seite.")
    ap.add_argument("--pdf", required=True, help="Pfad zur PDF-Datei")
    ap.add_argument("--page", type=int, default=1, help="Seitennummer (1-basiert). Standard: 1")
    ap.add_argument("--cols", type=int, default=240, help="Breite der ASCII-Zeile in Spalten. Standard: 240")
    ap.add_argument("--gap-threshold", type=float, default=40.0, help="Schwelle (Y-Lücke) zum Trennen der Systeme. Standard: 40")
    ap.add_argument("--vert-tol", type=float, default=0.9, help="Toleranz (in X) beim Zusammenfassen vertikaler Segmente. Standard: 0.9")
    ap.add_argument("--coverage", type=float, default=0.6, help="Mindestanteil der Systemhöhe, den ein Taktstrich überdecken muss. Standard: 0.6")
    ap.add_argument("--min-band-digits", type=int, default=6, help="Mindestanzahl an Ziffern, damit ein Band als System gilt. Standard: 6")
    args = ap.parse_args()

    try:
        text = extract_tab_from_pdf_page(
            pdf_path=args.pdf,
            page_index_0_based=args.page - 1,  # 1-basiert -> 0-basiert
            cols=args.cols,
            gap_threshold=args.gap_threshold,
            vert_tol=args.vert_tol,
            coverage_ratio=args.coverage,
            min_band_digits=args.min_band_digits
        )
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    except IndexError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}", file=sys.stderr)
        sys.exit(1)

    # Ausgabe auf STDOUT
    if text:
        sys.stdout.write(text.strip() + "\n")
    else:
        # Leere Ausgabe, wenn keine Ziffern gefunden wurden (z.B. gescannte PDFs)
        sys.stdout.write("")


if __name__ == "__main__":
    main()
