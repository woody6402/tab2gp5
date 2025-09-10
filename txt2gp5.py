# txttab2gp5.py
# usage: python txttab2gp5.py input.txt output.gp5
#
# Liest ASCII-TAB-Dateien mit Kopf-/Kommentarzeilen.
# Erkennung:
#   - Tab-Blöcke = 6 aufeinanderfolgende "Saiten"-Zeilen (E|, B|, G|, D|, A|, E|/D|)
#   - Vertikale Akkordspalten: Beginn einer Zahl (mehrstellige Bünde → EIN Onset)
#   - Taktstriche: '|' innerhalb der Tab-Zeilen → bestimmen Taktgrenzen
#
# Dynamische Taktlänge:
#   - Pro Takt werden die Spaltenabstände PROPORTIONAL in Notenlängen umgesetzt.
#   - Adaptive Auflösung: wir versuchen 16tel (Basis 16 Einheiten pro Takt),
#     wenn nötig steigen wir auf 32tel (Basis 32) oder 64tel (Basis 64) hoch,
#     damit JEDE erkannte Note (Onset) im Takt mindestens 1 Einheit erhält.
#   - Zerlegung in reguläre/dotierte Notenwerte (inkl. 32tel/64tel falls nötig).
#
# Hinweise:
#   - Drop-D (unterste Zeile "D|") wird erkannt und die 6. Saite entsprechend tiefer gestimmt.
#   - Hammer-ons etc. (z.B. "7h8") werden toleriert; Onsets sind Spalten, an denen eine Ziffer startet.
#
# Voraussetzung:
#   pip install pyguitarpro
#
import sys
from typing import List, Tuple, Optional
import guitarpro as gp


# ----------------------------- Einlesen / Normalisieren -----------------------------

def is_tab_line2(line: str) -> bool:
    """
    Grobe Heuristik: Zeile beginnt (ab whitespace) mit [E,B,G,D,A,e,b,g,d,a] gefolgt
    (direkt) von '|' oder ' |' oder '-|', und irgendwo danach folgen '-' oder Ziffern
    bzw. TAB-Symbole (h, p, s, /, \, ~, x).
    """
    s = line.lstrip()
    if not s:
        return False
    pipe = s.find('|')
    if pipe == -1 or pipe > 3:  # Label sollte ganz am Anfang stehen
        return False
    head = s[:pipe].strip().lower()
    if not head or head[0] not in ('e', 'b', 'g', 'd', 'a'):
        return False
    tail = s[pipe+1:]
    return any(ch.isdigit() or ch in '-|hps/\\~x' for ch in tail)

def is_tab_line(line: str) -> bool:
    """
    Erkenne sowohl beschriftete TAB-Zeilen (E|, B|, …) als auch
    unbeschriftete Zeilen, die direkt mit '|' beginnen bzw. zwischen
    zwei '|' überwiegend '-', Ziffern oder TAB-Symbole enthalten.
    """
    s = line.rstrip('\n')
    if not s.strip():
        return False

    t = s.lstrip()

    # 1) Beschriftete Variante: "E|", "B|", "G|", "D|", "A|" (ggf. mit Leerzeichen)
    pipe = t.find('|')
    if 0 <= pipe <= 3:
        head = t[:pipe].strip().lower()
        if head and head[0] in ('e', 'b', 'g', 'd', 'a'):
            tail = t[pipe+1:]
            if any(ch.isdigit() or ch in '-|hps/\\~x*().' for ch in tail):
                return True

    # 2) Unbeschriftete Variante: mindestens zwei '|' in der Zeile,
    #    und zwischen dem ersten und letzten '|' sieht es "tab-artig" aus.
    first = t.find('|')
    last  = t.rfind('|')
    if first != -1 and last != -1 and last > first:
        mid = t[first+1:last]
        # Gilt als TAB, wenn zwischen first..last entweder Ziffern vorkommen
        # ODER ausreichend viele '-' vorhanden sind (lange Haltestriche),
        # zusätzlich sind übliche TAB-Symbole erlaubt.
        if (any(ch.isdigit() for ch in mid) or mid.count('-') >= max(5, len(mid)//3)):
            return True

    return False


def normalize_tab_block(raw6: List[str]) -> Tuple[List[str], int]:
    """
    Nimmt 6 Tab-Zeilen (oben = hohe E) und schneidet pro Zeile
    den Bereich zwischen dem ersten und letzten '|' heraus.
    Innere Taktstriche '|' bleiben erhalten. Falls nur ein '|' existiert,
    wird lediglich das führende Label ('E|') entfernt.
    Ergebnis: 6 Strings gleicher Länge.
    """
    if len(raw6) != 6:
        raise ValueError("Erwarte genau 6 Zeilen pro Block (E,B,G,D,A,E/D).")
    cut = []
    for ln in raw6:
        s = ln.rstrip('\n')
        first = s.find('|')
        last  = s.rfind('|')
        if first == -1:
            cut.append(s.strip())
            continue
        if last > first:
            cut.append(s[first+1:last])  # innen behalten
        else:
            cut.append(s[first+1:].strip())
    w = max(len(row) for row in cut) if cut else 0
    cut = [row.ljust(w, '-') for row in cut]
    return cut, w


def detect_row_to_string(raw6: List[str]) -> List[int]:
    """
    Liefert row->string_no (1..6) je Block:
      [1,2,3,4,5,6]  für e-B-G-D-A-E (hoch→tief)
      [6,5,4,3,2,1]  für E-A-D-G-B-e (tief→hoch)
    """
    def head2(ln: str) -> str:
        s = ln.lstrip()
        return s[:2] if len(s) >= 2 else s

    heads = [head2(ln).lower() for ln in raw6]
    high_first = len(heads) >= 2 and heads[0].startswith(('e|','e ')) and heads[1].startswith(('b|','b '))
    low_first  = len(heads) >= 2 and heads[0].startswith(('e|','e ')) and heads[1].startswith(('a|','a '))
    if high_first:
        return [1,2,3,4,5,6]
    if low_first:
        return [6,5,4,3,2,1]
    if heads and heads[-1].startswith(('e|','e ')):
        return [6,5,4,3,2,1]
    return [1,2,3,4,5,6]


def detect_tuning_from_labels(raw6: List[str]) -> List[int]:
    """
    Bestimme einfache Stimmung aus Labels (Standard oder Drop-D).
    Rückgabe: MIDI-Pitches der Saiten 1..6 (oben→unten).
    """
    tuning = [64, 59, 55, 50, 45, 40]  # Standard EADGBE (e4 b3 g3 d3 a2 e2)
    if raw6 and raw6[-1].lstrip().lower().startswith('d|'):
        tuning[5] = 38  # D2 (Drop-D)
    return tuning


def read_ascii_tab(path: str):
    """
    Sucht in der Datei nach 6 aufeinanderfolgenden Tab-Zeilen und erzeugt daraus Blöcke.
    Kopf-/Kommentarzeilen werden ignoriert. Stimmung wird aus dem ersten erkannten Block
    (genauer: dessen Labels) abgeleitet.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip('\n') for ln in f]

    blocks, widths, rowmaps = [], [], []
    first_tuning: Optional[List[int]] = None

    i = 0
    while i < len(lines):
        if is_tab_line(lines[i]):
            if i + 5 < len(lines) and all(is_tab_line(lines[i + k]) for k in range(6)):
                raw6 = lines[i:i+6]
                if first_tuning is None:
                    first_tuning = detect_tuning_from_labels(raw6)
                rowmap = detect_row_to_string(raw6)
                norm6, w = normalize_tab_block(raw6)
                blocks.append(norm6)
                widths.append(w)
                rowmaps.append(rowmap)
                i += 6
            else:
                i += 1
        else:
            i += 1

    if not blocks:
        raise ValueError("Keine gültigen TAB-Blöcke (6 Zeilen) gefunden.")
    if first_tuning is None:
        first_tuning = [64, 59, 55, 50, 45, 40]
    return blocks, widths, rowmaps, first_tuning


# ---------------------- Spalten finden (Akkorde & Taktstriche) ----------------------

def find_chord_and_bar_columns(block: List[str]) -> Tuple[List[int], List[int]]:
    """
    Findet Spaltenindizes für:
      - chord_cols: Start einer Zahl (mehrstellige Bünde → Startspalte),
      - bar_cols:   '|' in beliebiger Zeile (Taktstriche).
    """
    if not block:
        return [], []
    width = len(block[0])
    chord_cols = set()
    bar_cols = set()
    for row in block:
        for c in range(width):
            ch = row[c]
            if ch == '|':
                bar_cols.add(c)
            elif ch.isdigit() and (c == 0 or not row[c-1].isdigit()):
                chord_cols.add(c)
    return sorted(chord_cols), sorted(bar_cols)


def extract_chord_at(block: List[str], rowmap: List[int], col: int) -> List[Tuple[int,int]]:
    """
    Extrahiert den Akkord an Spalte col als [(string_no, fret), ...].
    Erkennt mehrstellige Bünde (10,11,12,...).
    """
    notes = []
    for row_idx in range(6):
        ln = block[row_idx]
        if col >= len(ln):
            continue
        if ln[col].isdigit() and (col == 0 or not ln[col-1].isdigit()):
            j = col
            num = ""
            while j < len(ln) and ln[j].isdigit():
                num += ln[j]
                j += 1
            fret = int(num)
            string_no = int(rowmap[row_idx])  # 1..6
            notes.append((string_no, fret))
    return notes


# ---------------------- Quantisierung (proportional, dynamische Takte) ----------------------

def duration_table_for_base(base_units: int):
    """
    Liefert eine Tabelle [(chunk_units, (denominator, dotted)), ...] für greedy Zerlegung
    bei gegebener Basis (Einheiten pro GANZE Note). Unterstützt 16, 32, 64.
    """
    if base_units == 16:
        # 16tel-Basis
        return [
            (16, (1, False)),   # Ganze
            (12, (2, True)),    # punktierte Halbe
            (8,  (2, False)),   # Halbe
            (6,  (4, True)),    # punktierte Viertel
            (4,  (4, False)),   # Viertel
            (3,  (8, True)),    # punktierte Achtel
            (2,  (8, False)),   # Achtel
            (1,  (16, False)),  # Sechzehntel
        ]
    if base_units == 32:
        # 32tel-Basis (enthält 32tel)
        return [
            (32, (1, False)),   # Ganze
            (24, (2, True)),    # punktierte Halbe
            (16, (2, False)),   # Halbe
            (12, (4, True)),    # punktierte Viertel
            (8,  (4, False)),   # Viertel
            (6,  (8, True)),    # punktierte Achtel
            (4,  (8, False)),   # Achtel
            (3,  (16, True)),   # punktierte Sechzehntel
            (2,  (16, False)),  # Sechzehntel
            (1,  (32, False)),  # Zweiunddreißigstel
        ]
    if base_units == 64:
        # 64tel-Basis (bis 64tel)
        return [
            (64, (1, False)),   # Ganze
            (48, (2, True)),    # punktierte Halbe
            (32, (2, False)),   # Halbe
            (24, (4, True)),    # punktierte Viertel
            (16, (4, False)),   # Viertel
            (12, (8, True)),    # punktierte Achtel
            (8,  (8, False)),   # Achtel
            (6,  (16, True)),   # punktierte Sechzehntel
            (4,  (16, False)),  # Sechzehntel
            (3,  (32, True)),   # punktierte Zweiunddreißigstel
            (2,  (32, False)),  # Zweiunddreißigstel
            (1,  (64, False)),  # Vierundsechzigstel
        ]
    raise ValueError(f"Nicht unterstützte Basis: {base_units}")


def split_into_durations_units(n_units: int, base_units: int) -> List[Tuple[int, bool]]:
    """Greedy-Zerlegung von n_units (in Basis-Einheiten) in (denominator, dotted)."""
    res: List[Tuple[int, bool]] = []
    remaining = n_units
    table = duration_table_for_base(base_units)
    while remaining > 0:
        for chunk, dd in table:
            if chunk <= remaining:
                res.append(dd)
                remaining -= chunk
                break
    return res


def quantize_proportional(spans: List[int], base_units: int, prefer_mask: List[bool]) -> List[int]:
    """
    Weist den Spaltenabständen (spans) proportionale Dauer-Units zu (Summe = base_units).
    - prefer_mask[i] = True bevorzugt die Zuteilung (für Onsets), Pausensegmente dürfen 0 bekommen.
    - spans[i] == 0 → bekommt 0 Units.
    """
    n = len(spans)
    if n == 0 or base_units <= 0:
        return []
    total = sum(spans)
    if total <= 0:
        out = [0]*n
        idx = next((i for i, pref in enumerate(prefer_mask) if pref), 0)
        out[idx] = base_units
        return out

    proportions = [s * float(base_units) / total for s in spans]
    units = [0 if spans[i] == 0 else int(p) for i, p in enumerate(proportions)]  # floor
    remainder = base_units - sum(units)

    if remainder != 0:
        residuals = [(i, (proportions[i] - int(proportions[i]))) for i in range(n)]
        if remainder > 0:
            # Bevorzugte Events zuerst, dann größere Bruchteile
            residuals.sort(key=lambda t: ((0 if prefer_mask[t[0]] else 1), -t[1]))
            k = 0
            while remainder > 0 and n > 0:
                idx = residuals[k % n][0]
                units[idx] += 1
                remainder -= 1
                k += 1
        else:
            # Abziehen: nicht-bevorzugte zuerst, kleinste Bruchteile
            residuals.sort(key=lambda t: ((1 if prefer_mask[t[0]] else 0), t[1]))
            k = 0
            while remainder < 0 and n > 0:
                idx = residuals[k % n][0]
                if units[idx] > 0:
                    units[idx] -= 1
                    remainder += 1
                k += 1
    return units


# --------------------------------- Song-Aufbau ---------------------------------

DEFAULT_BASES = (16, 32, 64)  # adaptive Versuche

def build_song(blocks, widths, rowmaps, title="ASCII TAB", tuning_midi: Optional[List[int]] = None):
    """
    Dynamische Takte:
      1) Pro Block: Erkenne Akkordspalten (Onsets) und Taktstriche '|'.
      2) Taktgrenzen sind die '|' (0..width..); wenn der Schlussstrich fehlt, endet der Takt am Blockende.
      3) Zwischen Taktstart/Taktende: Markerfolge: [Start] + Onset-Spalten + [Ende].
      4) Proportionale Umrechnung in Notenlängen mit ADAPTIVER Auflösung (16/32/64 Einheiten).
      5) Jede Einheit wird in reguläre/dotierte Notenwerte zerlegt und geschrieben.
    """
    # --- Taktgrenzen vorbereiten ---
    measures_per_block: List[List[Tuple[int,int]]] = []
    for block, width in zip(blocks, widths):
        _, bar_cols = find_chord_and_bar_columns(block)
        bounds = [0] + [c for c in bar_cols if 0 < c < width] + [width]
        clean_bounds = [bounds[0]]
        for b in bounds[1:]:
            if b > clean_bounds[-1]:
                clean_bounds.append(b)
        pairs = [(clean_bounds[i], clean_bounds[i+1]) for i in range(len(clean_bounds)-1)]
        measures_per_block.append(pairs)

    total_measures = sum(len(pairs) for pairs in measures_per_block)

    # --- Song/Track ---
    song = gp.models.Song(title=title, tempo=120)
    song.measureHeaders.clear()
    song.tracks.clear()

    ts = gp.models.TimeSignature(4, gp.models.Duration(4))  # feste 4/4-Taktart
    for i in range(total_measures):
        song.addMeasureHeader(gp.models.MeasureHeader(number=i+1, timeSignature=ts))

    if tuning_midi is None:
        tuning_midi = [64, 59, 55, 50, 45, 40]  # Standard

    strings = [
        gp.models.GuitarString(1, tuning_midi[0]),
        gp.models.GuitarString(2, tuning_midi[1]),
        gp.models.GuitarString(3, tuning_midi[2]),
        gp.models.GuitarString(4, tuning_midi[3]),
        gp.models.GuitarString(5, tuning_midi[4]),
        gp.models.GuitarString(6, tuning_midi[5]),
    ]

    track = gp.models.Track(
        song,
        name="Gitarre",
        strings=strings,
        channel=gp.models.MidiChannel(instrument=25)  # Acoustic Nylon
    )
    song.tracks.append(track)

    # --- Pro Block → pro Takt ---
    current_measure_idx = 0
    for block, width, rowmap, pairs in zip(blocks, widths, rowmaps, measures_per_block):
        chord_cols, _ = find_chord_and_bar_columns(block)
        chord_set = set(chord_cols)

        for (m_start, m_end) in pairs:
            measure = track.measures[current_measure_idx]
            voice = measure.voices[0]

            if m_end <= m_start:
                beat = gp.models.Beat(voice, duration=gp.models.Duration(1))  # Ganze Pause
                beat.status = gp.models.BeatStatus.rest
                voice.beats.append(beat)
                current_measure_idx += 1
                continue

            onset_cols = sorted(c for c in chord_set if m_start <= c < m_end)
            markers = [m_start] + onset_cols + [m_end]

            spans: List[int] = []
            events: List[Optional[List[Tuple[int,int]]]] = []
            prefer: List[bool] = []
            for i in range(len(markers)-1):
                c0, c1 = markers[i], markers[i+1]
                spans.append(max(0, c1 - c0))
                if c0 in chord_set:
                    chord = extract_chord_at(block, rowmap, c0)
                    events.append(chord if chord else None)
                    prefer.append(True)
                else:
                    events.append(None)
                    prefer.append(False)

            # --- Adaptive Auflösung wählen: 16 → 32 → 64 ---
            chosen_base = None
            chosen_units = None
            for base in DEFAULT_BASES:
                units = quantize_proportional(spans, base, prefer)
                ok = True
                for u, is_onset, span in zip(units, prefer, spans):
                    if is_onset and span > 0 and u <= 0:
                        ok = False
                        break
                if ok:
                    chosen_base = base
                    chosen_units = units
                    break
            if chosen_base is None:
                # Fallback: nimm höchste Basis
                chosen_base = DEFAULT_BASES[-1]
                chosen_units = quantize_proportional(spans, chosen_base, prefer)

            # --- Beats schreiben ---
            for n_units, chord in zip(chosen_units, events):
                if n_units <= 0:
                    continue
                for den, dotted in split_into_durations_units(n_units, chosen_base):
                    beat = gp.models.Beat(voice, duration=gp.models.Duration(den, dotted))
                    if chord:
                        beat.status = gp.models.BeatStatus.normal
                        for s, f in chord:
                            note = gp.models.Note(
                                beat,
                                value=int(f),
                                string=int(s),
                                type=gp.models.NoteType.normal
                            )
                            beat.notes.append(note)
                    else:
                        beat.status = gp.models.BeatStatus.rest
                    voice.beats.append(beat)

            current_measure_idx += 1

    return song


# -------------------------------------- CLI --------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python txttab2gp5.py input.txt output.gp5")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]

    blocks, widths, rowmaps, tuning = read_ascii_tab(inp)
    song = build_song(blocks, widths, rowmaps, title=inp, tuning_midi=tuning)
    gp.write(song, out, version=(5, 1, 0))  # GP5
    print("geschrieben:", out)

