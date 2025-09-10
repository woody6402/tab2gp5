
# txttab2gp5_fixed.py
# usage:
#   python txttab2gp5_fixed.py input.txt output.gp5
# options:
#   --title "Song Title" --author "Author Name" --tab-spacing 2 --tempo 120 --bases 16,32,64 --dry-run
#
# Hält sich eng an die gewohnte Vorlage (Header → Measure → Track) und
# behebt das Problem "alles in EINEM Takt" durch explizite MeasureHeader/Measure-Erzeugung
# und korrekte TimeSignature-Initialisierung.
#
# Anforderungen aus deinem Pflichtenheft bleiben erhalten:
# - Quantisierung adaptiv (16→32→64)
# - Erste Note im Takt: kleine Vorlaufspalte ist KEINE Pause (via --tab-spacing)
# - Erkennung von mehr TAB-Mustern
# - Fehlertoleranz bei unplausiblen Saiten/Bünden
#
import sys
import re
import argparse
from typing import List, Tuple, Optional

try:
    import guitarpro as gp
except Exception as e:
    gp = None
    _gp_import_error = e

FRET_MAX_DEFAULT = 24

# Wird per CLI überschrieben
LEADING_SILENCE_TOL = 2
DEFAULT_BASES = (16, 32, 64)


# ----------------------------- Erkennung / Normalisieren -----------------------------

_TAB_LABEL_RE = re.compile(r'^\s*[eEbBgGdDaA]\s{0,3}\|(?=[^|]*[0-9hps/\\~x*().\-|])')
_PIPE_FINDER  = re.compile(r'\|')

def is_tab_line(line: str) -> bool:
    s = line.rstrip('\n')
    if not s.strip():
        return False
    t = s.lstrip()

    # Ruler/Count-Zeilen (z.B. "10 | . | . |")
    if re.match(r'^\s*\d+\s*\|[ .|]*$', s):
        return False

    if _TAB_LABEL_RE.search(t):
        return True

    pipes = [m.start() for m in _PIPE_FINDER.finditer(t)]
    if len(pipes) >= 2:
        first, last = pipes[0], pipes[-1]
        if last > first:
            mid = t[first+1:last]
            if re.search(r'\d', mid):
                return True
            if mid.count('-') >= max(5, len(mid)//3):
                return True
    return False


def normalize_tab_block(raw6: List[str]) -> Tuple[List[str], int]:
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
            cut.append(s[first+1:last])
        else:
            cut.append(s[first+1:].strip())
    w = max(len(row) for row in cut) if cut else 0
    cut = [row.ljust(w, '-') for row in cut]
    return cut, w


def detect_row_to_string(raw6: List[str]) -> List[int]:
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
    tuning = [64, 59, 55, 50, 45, 40]  # e4 b3 g3 d3 a2 e2
    if raw6 and raw6[-1].lstrip().lower().startswith('d|'):
        tuning[5] = 38  # Drop-D
    return tuning


def read_ascii_tab(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if is_tab_line(ln)]

    blocks, widths, rowmaps = [], [], []
    first_tuning: Optional[List[int]] = None
    i = 0
    while i + 5 < len(lines):
        raw6 = lines[i:i+6]
        if all(is_tab_line(ln) for ln in raw6):
            if first_tuning is None:
                first_tuning = detect_tuning_from_labels(raw6)
            rowmap = detect_row_to_string(raw6)
            norm6, w = normalize_tab_block(raw6)
            blocks.append(norm6); widths.append(w); rowmaps.append(rowmap)
            i += 6
        else:
            i += 1

    if not blocks:
        raise ValueError("Keine gültigen TAB-Blöcke (6 Zeilen) gefunden.")
    if first_tuning is None:
        first_tuning = [64, 59, 55, 50, 45, 40]
    return blocks, widths, rowmaps, first_tuning


# ----------------------------- Spalten & Akkorde -----------------------------

def is_bar_column(block, col, threshold=4):
    cnt = 0
    for ln in block:
        if col < len(ln) and ln[col] == '|':
            cnt += 1
    return cnt >= threshold


def find_chord_and_bar_columns(block: List[str]) -> Tuple[List[int], List[int]]:
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
    notes = []
    for row_idx in range(6):
        ln = block[row_idx]
        if col >= len(ln):
            continue
        if ln[col].isdigit() and (col == 0 or not ln[col-1].isdigit()):
            j = col
            num = ""
            while j < len(ln) and ln[j].isdigit():
                num += ln[j]; j += 1
            fret = int(num)
            string_no = int(rowmap[row_idx])
            notes.append((string_no, fret))
        elif ln[col] in ('x','X'):
            string_no = int(rowmap[row_idx])
            notes.append((string_no, 0))
    return notes


# ----------------------------- Quantisierung -----------------------------

def duration_table_for_base(base_units: int):
    if base_units == 16:
        return [
            (16, (1, False)), (12, (2, True)), (8, (2, False)), (6, (4, True)),
            (4, (4, False)), (3, (8, True)), (2, (8, False)), (1, (16, False))
        ]
    if base_units == 32:
        return [
            (32,(1,False)),(24,(2,True)),(16,(2,False)),(12,(4,True)),(8,(4,False)),
            (6,(8,True)),(4,(8,False)),(3,(16,True)),(2,(16,False)),(1,(32,False))
        ]
    if base_units == 64:
        return [
            (64,(1,False)),(48,(2,True)),(32,(2,False)),(24,(4,True)),(16,(4,False)),
            (12,(8,True)),(8,(8,False)),(6,(16,True)),(4,(16,False)),(3,(32,True)),
            (2,(32,False)),(1,(64,False))
        ]
    raise ValueError(f"Nicht unterstützte Basis: {base_units}")


def split_into_durations_units(n_units: int, base_units: int) -> List[Tuple[int, bool]]:
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
    units = [0 if spans[i] == 0 else int(p) for i, p in enumerate(proportions)]
    remainder = base_units - sum(units)

    if remainder != 0:
        residuals = [(i, (proportions[i] - int(proportions[i]))) for i in range(n)]
        if remainder > 0:
            residuals.sort(key=lambda t: ((0 if prefer_mask[t[0]] else 1), -t[1]))
            k = 0
            while remainder > 0 and n > 0:
                idx = residuals[k % n][0]
                units[idx] += 1; remainder -= 1; k += 1
        else:
            residuals.sort(key=lambda t: ((1 if prefer_mask[t[0]] else 0), t[1]))
            k = 0
            while remainder < 0 and n > 0:
                idx = residuals[k % n][0]
                if units[idx] > 0:
                    units[idx] -= 1; remainder += 1
                k += 1
    return units


def _make_duration_table_q(quarter_units: int):
    q = float(max(1, quarter_units))
    def U(den: int, dotted: bool = False) -> int:
        units = q * (4.0 / den)
        if dotted:
            units *= 1.5
        return max(1, int(round(units)))
    table = [
        (U(1),   (1, False)),
        (U(2,1), (2, True)),
        (U(2),   (2, False)),
        (U(4,1), (4, True)),
        (U(4),   (4, False)),
        (U(8,1), (8, True)),
        (U(8),   (8, False)),
        (U(16,1),(16, True)),
        (U(16),  (16, False)),
        (U(32),  (32, False)),
    ]
    seen, out = set(), []
    for chunk, dd in table:
        key = (chunk, dd[0], dd[1])
        if key in seen or chunk <= 0:
            continue
        seen.add(key); out.append((chunk, dd))
    out.sort(key=lambda t: -t[0])
    return out

def split_into_durations_units_q(n_units: int, quarter_units: int):
    table = _make_duration_table_q(quarter_units)
    remaining = n_units; out = []
    while remaining > 0:
        for chunk, dd in table:
            if chunk <= remaining:
                out.append(dd); remaining -= chunk; break
        else:
            chunk, dd = table[-1]; out.append(dd); remaining -= chunk
    return out


# ----------------------------- Song-Aufbau -----------------------------

def build_song(blocks, widths, rowmaps, title="ASCII TAB", tuning_midi: Optional[List[int]] = None,
               tempo=120, bases: Tuple[int,...]=DEFAULT_BASES, fret_max=FRET_MAX_DEFAULT,
               fixed_meter: Optional[Tuple[int,int]] = None, quarter_units: int = 4):
    global DEFAULT_BASES
    DEFAULT_BASES = bases

    # ---- Grenzen je Maß ermitteln ----
    measures_per_block: List[List[Tuple[int,int]]] = []
    for block, width in zip(blocks, widths):
        chord_cols, bar_cols = find_chord_and_bar_columns(block)
        bounds = [0] + [c for c in bar_cols if 0 < c < width] + [width]
        clean_bounds = [bounds[0]]
        for b in bounds[1:]:
            if b > clean_bounds[-1]:
                clean_bounds.append(b)
        pairs = [(clean_bounds[i], clean_bounds[i+1]) for i in range(len(clean_bounds)-1)]
        filtered = []
        for a, b in pairs:
            has_onset = any(a <= c < b for c in chord_cols)
            only_bars = all(is_bar_column(block, c) for c in range(a, b))
            if has_onset or not only_bars:
                filtered.append((a, b))
        measures_per_block.append(filtered)

    total_measures = sum(len(pairs) for pairs in measures_per_block)

    # ---- Song/Headers/Track exakt nach Vorlage ----
    if gp is None:
        raise RuntimeError(f"pyguitarpro fehlt: {_gp_import_error!r}")

    song = gp.models.Song(title=title, tempo=tempo)

    # TimeSignature: **wichtig** nur (numerator:int, denominator:int)
    
    #ts = gp.models.TimeSignature(4, gp.models.Duration(4))
    base_numer, base_denom = (fixed_meter if fixed_meter else (4, 4))
    ts = gp.models.TimeSignature(base_numer, gp.models.Duration(base_denom))
    
    # 1) Alle MeasureHeader explizit anlegen und in song.measureHeaders speichern
    song.measureHeaders.clear()
    headers = []
    for i in range(total_measures):
        mh = gp.models.MeasureHeader(number=i+1, timeSignature=ts)
        headers.append(mh)
    song.measureHeaders.extend(headers)

    # 2) Track mit Strings/Stimmung anlegen
    if tuning_midi is None:
        tuning_midi = [64, 59, 55, 50, 45, 40]
    strings = [gp.models.GuitarString(i+1, tuning_midi[i]) for i in range(6)]
    track = gp.models.Track(song, name="Gitarre", strings=strings, channel=gp.models.MidiChannel(instrument=25))
    song.tracks.clear()
    song.tracks.append(track)

    # 3) **Zu JEDEM Header eine Measure im Track erzeugen** (sonst landet alles im ersten Takt)
    track.measures.clear()
    for mh in headers:
        m = gp.models.Measure(track, header=mh)
        track.measures.append(m)

    # ---- Füllen ----
    current_measure_idx = 0
    for block, width, rowmap, pairs in zip(blocks, widths, rowmaps, measures_per_block):
        chord_cols, _ = find_chord_and_bar_columns(block)
        chord_set = set(chord_cols)

        for (m_start, m_end) in pairs:
            measure = track.measures[current_measure_idx]
            voice = measure.voices[0]

            if m_end <= m_start:
                beat = gp.models.Beat(voice, duration=gp.models.Duration(1))
                beat.status = gp.models.BeatStatus.rest
                voice.beats.append(beat)
                current_measure_idx += 1
                continue

            onset_cols = sorted(c for c in chord_set if m_start <= c < m_end)
            markers = [m_start] + onset_cols + [m_end]

            # kurze Vorlaufstrecke NICHT als Pause werten
            c0 = m_start
            skipped = 0
            while c0 < m_end and skipped < LEADING_SILENCE_TOL:
                if is_bar_column(block, c0):
                    c0 += 1; continue
                if extract_chord_at(block, rowmap, c0):
                    break
                c0 += 1; skipped += 1
            if skipped:
                markers[0] = c0
                if len(markers) >= 2 and markers[0] >= markers[1]:
                    markers.pop(0)
                if len(markers) == 1 or markers[0] >= markers[-1]:
                    current_measure_idx += 1
                    continue

            spans: List[int] = []
            events: List[Optional[List[Tuple[int,int]]]] = []
            prefer: List[bool] = []
            for i in range(len(markers)-1):
                a, b = markers[i], markers[i+1]
                spans.append(max(0, b - a))
                if a in chord_set:
                    chord = extract_chord_at(block, rowmap, a)
                    events.append(chord if chord else None)
                    prefer.append(True)
                else:
                    events.append(None); prefer.append(False)

            if fixed_meter:
                numer, denom = fixed_meter
                headers[current_measure_idx].timeSignature = gp.models.TimeSignature(
                    numer, gp.models.Duration(denom)
                )
                measure_units = int(round(max(1, quarter_units) * (numer * 4.0 / denom)))
                chosen_units = quantize_proportional(spans, measure_units, prefer)
                for i, (u, is_onset, sp) in enumerate(zip(chosen_units, prefer, spans)):
                    if is_onset and sp > 0 and u <= 0:
                        chosen_units[i] = 1
            else:
                # Adaptive Basis
                chosen_base = None; chosen_units = None
                for base in DEFAULT_BASES:
                    units = quantize_proportional(spans, base, prefer)
                    ok = True
                    for u, is_onset, span in zip(units, prefer, spans):
                        if is_onset and span > 0 and u <= 0:
                            ok = False; break
                    if ok:
                        chosen_base = base; chosen_units = units; break
                if chosen_base is None:
                    chosen_base = DEFAULT_BASES[-1]
                    chosen_units = quantize_proportional(spans, chosen_base, prefer)

            for n_units, chord in zip(chosen_units, events):
                if n_units <= 0:
                    continue
                for den, dotted in (split_into_durations_units_q(n_units, quarter_units)
                                    if fixed_meter else
                                    split_into_durations_units(n_units, chosen_base)):
                    beat = gp.models.Beat(voice, duration=gp.models.Duration(den, dotted))

                    if chord:
                        beat.status = gp.models.BeatStatus.normal
                        for s, f in chord:
                            s_int = int(s); f_int = int(f)
                            if not (1 <= s_int <= 6 and 0 <= f_int <= fret_max):
                                continue
                            note = gp.models.Note(beat, value=f_int, string=s_int, type=gp.models.NoteType.normal)
                            beat.notes.append(note)
                    else:
                        beat.status = gp.models.BeatStatus.rest
                    voice.beats.append(beat)

            current_measure_idx += 1

    return song


# ----------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser(description="ASCII TAB → GP5")
    p.add_argument("input", help="Pfad zur TAB-Textdatei")
    p.add_argument("output", nargs="?", help="Ausgabedatei .gp5 (Standard: gleich wie input, aber .gp5)")
    p.add_argument("--title", "-t", default=None, help="Songtitel")
    p.add_argument("--author", "-a", default=None, help="Autor/Artist")
    p.add_argument("--tempo", type=int, default=120, help="Tempo (BPM), Standard 120")
    p.add_argument("--tab-spacing", type=int, default=2, help="Vorlaufspalten am Taktbeginn, die NICHT als Pause zählen")
    p.add_argument("--bases", default="16,32,64", help="Quantisierungs-Basen, z.B. 16,32 oder 16,32,64")
    p.add_argument("--fret-max", type=int, default=FRET_MAX_DEFAULT, help="max. Bund (Fehlertoleranz)")
    p.add_argument("--dry-run", action="store_true", help="Kein GP5 schreiben; nur Parsing/Quantisierung ausgeben")
    p.add_argument("--meter", default="", help="Feste Taktart N/D, z.B. 4/4, 3/4, 6/8")

    args = p.parse_args()

    global LEADING_SILENCE_TOL
    LEADING_SILENCE_TOL = max(0, int(args.tab_spacing))

    bases = tuple(int(x) for x in args.bases.split(",") if x.strip())

    blocks, widths, rowmaps, tuning = read_ascii_tab(args.input)
    title = args.title or args.input
    fixed_meter = None
    if args.meter:
        try:
            n_s, d_s = args.meter.split("/", 1)
            fixed_meter = (int(n_s), int(d_s))
        except Exception:
            fixed_meter = None

    if args.dry_run or gp is None:
        # Gesamtmaß-Zähler für Dry-Run
        _total_measures = 0
        for block, width in zip(blocks, widths):
            chord_cols, bar_cols = find_chord_and_bar_columns(block)
            bounds = [0] + [c for c in bar_cols if 0 < c < width] + [width]
            clean = [bounds[0]]
            for b in bounds[1:]:
                if b > clean[-1]: clean.append(b)
            pairs = [(clean[i], clean[i+1]) for i in range(len(clean)-1)]
            _total_measures += len(pairs)
        print(f"MEASURES (gesamt): {_total_measures}")
        
        # Zeige Taktgrenzen pro Block+Takt zur Kontrolle
        for bi, (block, width, rowmap) in enumerate(zip(blocks, widths, rowmaps)):
            chord_cols, bar_cols = find_chord_and_bar_columns(block)
            bounds = [0] + [c for c in bar_cols if 0 < c < width] + [width]
            clean = [bounds[0]]
            for b in bounds[1:]:
                if b > clean[-1]: clean.append(b)
            pairs = [(clean[i], clean[i+1]) for i in range(len(clean)-1)]
            print(f"Block {bi}: width={width}, bars={bar_cols}")
            for (a,b) in pairs:
                onsets = [c for c in chord_cols if a <= c < b]
                print(f"  Measure {a}-{b}: onsets@{onsets}")
        if gp is None:
            print("\n(Achtung: pyguitarpro ist hier nicht installiert; GP5-Ausgabe wurde übersprungen.)")
        return 0

    song = build_song(blocks, widths, rowmaps, title=title, tuning_midi=tuning,
                      tempo=args.tempo, bases=bases, fret_max=args.fret_max,
                      fixed_meter=fixed_meter, quarter_units=4)

    if args.author and hasattr(song, "artist"):
        song.artist = args.author
    elif args.author and hasattr(song, "artistName"):
        song.artistName = args.author

    out = args.output or (args.input.rsplit(".",1)[0] + ".gp5")
    gp.write(song, out, version=(5, 1, 0))
    print("geschrieben:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
