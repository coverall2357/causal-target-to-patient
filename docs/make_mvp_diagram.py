"""Draw the trial-to-target map as a funnel: all RA trials, the labelling set,
and the contrast tests it yields.

Reads the headline counts from data/summary.json (written by
scripts/export_map.py), so the diagram always matches the published map. Each
bar is drawn to scale; the shaded band shows which part of a bar becomes the
next one.

Uses only the standard library. With --png, also writes a PNG at twice the
resolution, which needs cairosvg (pip install cairosvg).

Usage:
    python docs/make_mvp_diagram.py                  # writes docs/mvp_map.svg
    python docs/make_mvp_diagram.py --png
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

W = 1400
X0, BW = 60, 1280          # bar left edge and full width
BAR_H = 46
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
INK, MUTED, LINE = "#1F2933", "#52606D", "#7B8794"
BAND = "#EAF2F8"
CHAR_W = 6.9               # approximate width of one character at 13px

# Segment colours: blue for what carries forward, greys for what doesn't.
FORWARD = "#0072B2"
GREYS = ("#9AA5B1", "#BCC5CE", "#D5DBE1", "#E4E7EB", "#EEF1F4", "#F5F7FA")
# Evidence tiers, strongest first: one hue, dark to light, then grey for none.
TIER_COLOURS = {"primary analysis": "#0B4F80", "primary results": "#0072B2",
                "publication": "#5BA4D6", "stop reason": "#A9CFEA", "none": "#E4E7EB"}

SET_NAMES = {"labelling": "Labelling: phase 1/2–3, finished",
             "context": "Context: phase 4", "ongoing": "Ongoing", "excluded": "Excluded"}
CLASS_NAMES = {"target_contrast": "Target contrast", "head_to_head": "Head-to-head",
               "uncontrolled": "Uncontrolled", "excluded_therapy": "Conventional therapy only",
               "same_target": "Same target", "unmapped": "Unmapped",
               "single_intervention": "Single intervention", "no_test": "No test"}
TIER_NAMES = {"primary analysis": "Primary analysis posted",
              "primary results": "Primary results posted", "publication": "Publication",
              "stop reason": "Stop reason only", "none": "No evidence yet"}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self):
        self.parts = []
        self.under = []   # drawn first, beneath bars and labels

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(self, x, y, s, size=13, weight=400, fill=INK, anchor="start"):
        self.add(f'<text x="{round(x, 2)}" y="{round(y, 2)}" font-family="{FONT}" '
                 f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
                 f'text-anchor="{anchor}">{esc(s)}</text>')


def segments(counts: dict, order, names, colours) -> list[dict]:
    """Bar segments in order, with x positions scaled to the bar width."""
    total = sum(counts.get(k, 0) for k in order)
    out, x = [], X0
    for key, colour in zip(order, colours):
        n = counts.get(key, 0)
        if not n:
            continue
        w = BW * n / total
        out.append({"key": key, "n": n, "x": x, "w": w, "colour": colour,
                    "label": names[key]})
        x += w
    return out


def bar(s: Svg, y: float, segs: list[dict]) -> float:
    """Draw a stacked bar with labels; returns the y below its labels."""
    gap = 2
    for seg in segs:
        s.add(f'<rect x="{round(seg["x"], 2)}" y="{y}" width="{round(max(seg["w"] - gap, 1), 2)}" '
              f'height="{BAR_H}" rx="3" fill="{seg["colour"]}"/>')
    # Labels inside when they fit. The rest go on one row below, laid out right
    # to left so they never overlap; a label that can't sit under its segment
    # gets an elbow leader.
    below_y = y + BAR_H
    outside = []
    for seg in segs:
        dark = seg["colour"] in (FORWARD, "#0B4F80")
        ink = "#FFFFFF" if dark else INK
        name, count = seg["label"], f'{seg["n"]:,}'
        if seg["w"] >= CHAR_W * max(len(name), len(count)) + 22:
            s.text(seg["x"] + 10, y + 20, name, size=13, weight=600, fill=ink)
            s.text(seg["x"] + 10, y + 38, count, size=13, fill=ink)
        else:
            outside.append(seg)
    ty = y + BAR_H + 26
    limit = X0 + BW
    for seg in reversed(outside):
        name, count = seg["label"], f'{seg["n"]:,}'
        width = CHAR_W * (len(name) + 1 + len(count))
        centre = seg["x"] + (seg["w"] - 2) / 2
        right = min(centre + width / 2, limit)
        left = right - width
        if left <= centre <= right:
            s.add(f'<line x1="{round(centre, 2)}" y1="{y + BAR_H + 2}" x2="{round(centre, 2)}" '
                  f'y2="{ty - 15}" stroke="{LINE}" stroke-width="1"/>')
        else:
            elbow = y + BAR_H + 9
            s.add(f'<path d="M{round(centre, 2)},{y + BAR_H + 2} V{elbow} '
                  f'H{round(right - 12, 2)} V{ty - 15}" '
                  f'fill="none" stroke="{LINE}" stroke-width="1"/>')
        s.add(f'<text x="{round(left, 2)}" y="{ty}" font-family="{FONT}" font-size="13" '
              f'fill="{INK}"><tspan font-weight="600">{esc(name)}</tspan>'
              f'<tspan fill="{MUTED}"> {count}</tspan></text>')
        limit = left - 16
        below_y = ty + 6
    return below_y


def band(s: Svg, top: float, bottom: float, x_from: float, x_to: float, label: str) -> None:
    """Shaded band from part of one bar to the whole of the next."""
    s.under.append(f'<path d="M{round(x_from, 2)},{top} L{round(x_to, 2)},{top} '
                   f'L{X0 + BW},{bottom} L{X0},{bottom} Z" fill="{BAND}"/>')
    s.text(X0 + 16, bottom - 12, label, size=13, fill=MUTED)


def heading(s: Svg, y: float, left: str, right: str = "") -> None:
    s.text(X0, y, left, size=17, weight=700)
    if right:
        s.text(X0 + BW, y, right, size=14, weight=600, fill=FORWARD, anchor="end")


def draw(summary: dict) -> str:
    s = Svg()
    sets = segments(summary["by_trial_set"], ["labelling", "context", "ongoing", "excluded"],
                    SET_NAMES, (FORWARD,) + GREYS)
    classes = summary["labelling_by_contrast_class"]
    tests_first = ["target_contrast", "head_to_head"]
    rest = sorted((k for k in classes if k not in tests_first), key=lambda k: -classes[k])
    contrast = segments(classes, tests_first + rest, CLASS_NAMES,
                        (FORWARD, "#56B4E9") + GREYS)
    tiers = segments(summary["labelling_tests_by_evidence_tier"], list(TIER_COLOURS),
                     TIER_NAMES, list(TIER_COLOURS.values()))
    stamp = summary["source_stamp"]
    as_of = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"

    y = 130
    heading(s, y, f'{summary["trials"]:,} RA interventional trials, by trial set')
    y1 = y + 16
    end1 = bar(s, y1, sets)

    y2 = end1 + 100
    lab = sets[0]
    band(s, y1 + BAR_H + 4, y2 - 34, lab["x"], lab["x"] + lab["w"] - 2,
         "Only finished trials with a possible outcome go forward")
    heading(s, y2 - 10, f'{summary["by_trial_set"]["labelling"]:,} labelling trials, '
            "by what they compare at the target level")
    end2 = bar(s, y2 + 6, contrast)

    y3 = end2 + 100
    n_tests = summary["labelling_tests"]
    band(s, y2 + 6 + BAR_H + 4, y3 - 34, contrast[0]["x"], contrast[1]["x"] + contrast[1]["w"] - 2,
         "Only comparisons between different targets count as tests")
    heading(s, y3 - 10, f"{n_tests:,} contrast tests, by strongest outcome evidence",
            f'{summary["targets_tested"]:,} targets ({summary["genes_tested"]:,} genes) tested')
    end3 = bar(s, y3 + 6, tiers)

    h = end3 + 80
    s.text(X0, h - 40, "Sources: ClinicalTrials.gov via AACT (CTTI); ChEMBL (EMBL-EBI); "
           "curated decisions with cited sources in curated/trials/.", size=13, fill=MUTED)
    s.text(X0, h - 20, f"Pipeline outputs dated {as_of}. Data: data/ (CC BY-SA 4.0). "
           "Generated by docs/make_mvp_diagram.py from data/summary.json.", size=13, fill=MUTED)

    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{round(h)}" '
            f'viewBox="0 0 {W} {round(h)}" role="img" aria-label="Funnel from '
            f'{summary["trials"]:,} RA trials to {n_tests:,} contrast tests">',
            f'<rect width="{W}" height="{round(h)}" fill="#FFFFFF"/>']
    s.parts[:0] = head + s.under
    s.text(X0, 52, "What has been tried in RA, against what?", size=28, weight=700)
    s.text(X0, 82, "The RA trial-to-target map: every bar is to scale; shaded bands show "
           "what carries forward", size=15, fill=MUTED)
    s.add("</svg>")
    return "\n".join(s.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "data" / "summary.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "mvp_map.svg")
    parser.add_argument("--png", action="store_true",
                        help="also write a PNG at twice the resolution (needs cairosvg)")
    args = parser.parse_args()

    svg = draw(json.loads(args.summary.read_text()))
    args.out.write_text(svg + "\n")
    print(f"Written {args.out}")
    if args.png:
        import cairosvg

        png = args.out.with_suffix(".png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=2)
        print(f"Written {png}")


if __name__ == "__main__":
    main()
