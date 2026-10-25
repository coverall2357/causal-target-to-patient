"""Draw the conceptual schematic of the project as an SVG.

Three columns (sources; models and labels; tests and benchmark) across three
bands, one per Koller link: mechanism -> target, target -> therapeutic,
therapeutic -> patient. Each box carries a status badge; to update the
diagram as work progresses, edit STATUSES and AS_OF below and rerun.

Uses only the standard library. With --png, also writes a PNG at twice the
resolution, which needs cairosvg (pip install cairosvg).

Usage:
    python docs/make_schematic.py                 # writes docs/schematic.svg
    python docs/make_schematic.py --out path.svg --png
"""

import argparse
from pathlib import Path

# --- Edit these as the work progresses -------------------------------------------

AS_OF = "2026-10-25"

# One of: BUILT, PARTLY BUILT, PROTOTYPE, PLANNED.
STATUSES = {
    "corpus_mechanism": "PLANNED",
    "mechanism_model": "PLANNED",
    "genetics": "PLANNED",
    "trial_universe": "BUILT",
    "corpus_trials": "PARTLY BUILT",
    "outcome_labels": "PROTOTYPE",
    "outcome_table": "PLANNED",
    "corpus_patient": "PLANNED",
    "patient_model": "PLANNED",
    "mapping": "PLANNED",
    "biopsy_cohorts": "PLANNED",
    "stratified_trials": "PLANNED",
}

# --- Style --------------------------------------------------------------------------

W, H = 1600, 1060
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"

# Okabe-Ito colours for Koller's three links; neutrals for structure.
LINK = {1: "#0072B2", 2: "#009E73", 3: "#CC79A7"}
TINT = {1: "#EAF2F8", 2: "#E8F5F0", 3: "#F8EEF4"}
INK, MUTED, LINE = "#1F2933", "#52606D", "#7B8794"
BADGE = {  # status -> (border dash, badge fill, badge text colour)
    "BUILT": (None, "#1F2933", "#FFFFFF"),
    "PARTLY BUILT": (None, "#E69F00", "#1F2933"),
    "PROTOTYPE": (None, "#E69F00", "#1F2933"),
    "PLANNED": ("7 5", "#FFFFFF", "#52606D"),
}

# Column x-positions and widths.
AX, AW = 50, 320          # sources
BX, BW = 392, 64          # LLM extraction band
CX, CW = 478, 404         # models and labels
DX, DW = 934, 626         # tests and benchmark
DL, DR = (DX, 304), (DX + 322, 304)   # two sub-columns in the tests column


class Svg:
    """Collects SVG elements."""

    def __init__(self):
        self.parts = []

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(self, x, y, s, size=14.5, weight=400, fill=INK, anchor="start",
             italic=False, spacing=0):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        x, y = round(x, 2), round(y, 2)  # avoid float noise such as 329.59999
        style = ' font-style="italic"' if italic else ""
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{style}{ls}>'
                 f'{s}</text>')

    def badge(self, x, y, status):
        _, fill, colour = BADGE[status]
        width = round(9 + 7.4 * len(status), 2)
        x = round(x, 2)
        stroke = ' stroke="#9AA5B1" stroke-width="1"' if status == "PLANNED" else ""
        self.add(f'<rect x="{x}" y="{y}" width="{width}" height="20" rx="10" '
                 f'fill="{fill}"{stroke}/>')
        self.text(x + width / 2, y + 14.5, status, size=10.5, weight=700, fill=colour,
                  anchor="middle", spacing=0.6)
        return width

    def box(self, key, x, y, w, h, title, lines, link, tests=None, badge="top"):
        status = STATUSES[key]
        dash = BADGE[status][0]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" '
                 f'stroke="{LINK[link]}" stroke-width="2"{d}/>')
        badge_width = 9 + 7.4 * len(status)
        self.badge(x + w - badge_width - 10, y + 10 if badge == "top" else y + h - 30, status)
        titles = title if isinstance(title, list) else [title]
        for i, t in enumerate(titles):
            self.text(x + 16, y + 30 + i * 23, t, size=18.5, weight=700)
        top = y + 30 + 23 * len(titles) + 4
        for i, line in enumerate(lines):
            self.text(x + 16, top + i * 20.5, line)
        if tests:
            self.text(x + 16, y + h - 14, tests, size=13.5, weight=600, fill=LINK[link],
                      italic=True)

    def arrow(self, d, label=None, lx=0, ly=0, anchor="start", dashed=False):
        da = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="2.2"{da} '
                 f'marker-end="url(#head)"/>')
        if label:
            self.text(lx, ly, label, size=12.5, fill=MUTED, anchor=anchor, italic=True)

    def render(self) -> str:
        return "\n".join(self.parts)


def draw() -> str:
    s = Svg()
    s.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
          f'viewBox="0 0 {W} {H}">')
    s.add('<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
          f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
          f'fill="{LINE}"/></marker></defs>')
    s.add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # Title
    s.text(40, 52, "From Target to Patient: conceptual schematic", size=28, weight=700)
    s.text(40, 82, "causal-target-to-patient · rheumatoid arthritis · models built from the "
           f"literature with AI, tested against data · status as of {AS_OF}", size=15,
           fill=MUTED)

    # Column headers
    for x, w, t in ((AX, AW, "SOURCES"), (CX, CW, "MODELS AND LABELS"),
                    (DX, DW, "TESTS AND BENCHMARK")):
        s.text(x + w / 2, 124, t, size=13, weight=700, fill=MUTED, anchor="middle",
               spacing=1.5)

    # Koller link bands
    rows = {1: (140, 236), 2: (392, 262), 3: (670, 268)}
    labels = {1: "KOLLER LINK 1 · MECHANISM → TARGET",
              2: "KOLLER LINK 2 · TARGET → THERAPEUTIC",
              3: "KOLLER LINK 3 · THERAPEUTIC → PATIENT"}
    for k, (y, h) in rows.items():
        s.add(f'<rect x="24" y="{y}" width="{W - 48}" height="{h}" rx="14" fill="{TINT[k]}"/>')
        s.add(f'<rect x="24" y="{y}" width="7" height="{h}" rx="3" fill="{LINK[k]}"/>')
        s.text(W - 44, y + 22, labels[k], size=12.5, weight=700, fill=LINK[k], spacing=1.2,
               anchor="end")

    # LLM extraction band, spanning all three links
    s.add(f'<rect x="{BX}" y="160" width="{BW}" height="768" rx="12" fill="#FFFFFF" '
          f'stroke="#9AA5B1" stroke-width="1.5"/>')
    s.add(f'<text transform="translate({BX + BW / 2 + 5},544) rotate(-90)" '
          f'font-family="{FONT}" font-size="14" font-weight="700" fill="{MUTED}" '
          'text-anchor="middle" letter-spacing="1">'
          'LLM EXTRACTION · every claim quoted to its source · checked against hand labels'
          '</text>')

    # Link 1: mechanism -> target
    s.box("corpus_mechanism", AX, 172, AW, 186, ["Corpus · layer 2", "Mechanism literature"],
          ["Open Targets text-mined evidence,", "reviews and key primary papers,",
           "capped per mechanism"], 1)
    s.box("mechanism_model", CX, 172, CW, 160, "Mechanism model",
          ["Evidence-tagged graph:", "mechanism → target → RA,",
           "each edge graded by evidence tier"], 1)
    s.box("genetics", DX, 172, DW, 138, "Genetics",
          ["Mendelian randomization, colocalization,",
           "phenome-wide scans (on-target safety)"],
          1, tests="Tests the mechanism model's causal target → RA edges")

    # Link 2: target -> therapeutic
    s.box("trial_universe", AX, 424, AW, 106, "Trial universe",
          ["AACT + ChEMBL: 1,223 RA trials →", "417 controlled tests, mapped to targets"], 2)
    s.box("corpus_trials", AX, 548, AW, 96, "Corpus · layer 1",
          ["Trial-linked publications:", "AACT links + Europe PMC by NCT"], 2)
    s.box("outcome_labels", CX, 424, CW, 220, "Outcome labels",
          ["LLM reads each trial's evidence packet;", "quotes checked against the packet;",
           "scored against blind hand labels", "Efficacy · effect size · enrolled population",
           "Safety axis (later)"], 2)
    s.box("outcome_table", DX, 424, DW, 220, "Mechanism outcome table · benchmark",
          ["Trial results rolled up by mechanism and population.",
           "Do the models predict which mechanisms succeeded,",
           "failed, or worked only in some patients?"],
          2, tests="Scores the mechanism model and the mapping")

    # Link 3: therapeutic -> patient
    s.box("corpus_patient", AX, 702, AW, 214,
          ["Corpus · layer 3", "Patient & pathotype", "literature"],
          ["Seed set (AMP, PEAC, R4RA, STRAP,", "pathotype reviews) + citation", "expansion"],
          3)
    s.box("patient_model", CX, 702, 190, 214, "Patient model",
          ["Subgroups, pathotypes,", "effect modifiers, graded:", "interaction test >",
           "post hoc subgroup >", "observational"], 3, badge="bottom")
    s.box("mapping", CX + 214, 702, 190, 214, "Mapping",
          ["Which mechanisms", "matter in which", "patients, so who", "should respond to",
           "what"], 3)
    s.box("biopsy_cohorts", DL[0], 702, DL[1], 194, "Biopsy cohorts",
          ["AMP/SDY998, PEAC, R4RA", "Foundation model vs. PCA:", "do embeddings recover the",
           "literature's pathotypes?"], 3, tests="Tests the patient model")
    s.box("stratified_trials", DR[0], 702, DR[1], 194, ["Stratified and", "subgroup trials"],
          ["R4RA, STRAP; subgroup results", "in the labelled trials"],
          3, tests="Tests the mapping")

    # Sources -> extraction -> models
    s.arrow(f"M{AX + AW},262 H{CX - 3}")                        # layer 2 -> mechanism model
    s.arrow(f"M{AX + AW},590 H{CX - 3}")                        # layer 1 -> outcome labels
    s.arrow(f"M{AX + AW},628 H{BX + BW + 10} V770 H{CX - 3}")   # layer 1 -> patient model
    s.text(AX + 8, 664, "subgroup claims ↘", size=12.5, fill=MUTED, italic=True)
    s.arrow(f"M{AX + AW},820 H{CX - 3}")                        # layer 3 -> patient model

    # The trial universe anchors the corpus
    s.arrow(f"M{AX + 160},530 V546")                            # -> layer 1
    s.text(AX + 172, 541, "anchors", size=12.5, fill=MUTED, italic=True)
    s.arrow(f"M{AX + 120},424 V360", dashed=True)               # -> layer 2 scope
    s.text(AX + 132, 392, "tested mechanisms set the scope", size=12.5, fill=MUTED,
           italic=True)

    # Models -> mapping
    s.arrow(f"M{CX + 190},808 H{CX + 211}")                     # patient model -> mapping
    s.arrow(f"M{CX + CW},280 H{CX + CW + 26} V760 H{CX + CW + 3}")  # mechanism model -> mapping
    s.text(CX + CW + 32, 690, "mechanisms", size=12.5, fill=MUTED, italic=True)

    # Labels and predictions -> benchmark
    s.arrow(f"M{CX + CW},534 H{DX - 3}")                        # labels -> outcome table
    s.arrow(f"M{CX + CW},321 H{DX + 120} V421", "predicts which mechanisms succeed",
            DX + 132, 368)
    s.arrow(f"M{CX + 214 + 95},916 V940 H{DR[0] + DR[1] / 2} V899",  # mapping -> trials
            "predicts who responds to what", DX + 40, 956)
    s.arrow(f"M{DX},230 H{CX + CW + 3}", "tests", DX - 8, 221, anchor="end")  # genetics

    # Footer: causal methods, and the status legend
    s.add(f'<line x1="40" y1="990" x2="{W - 40}" y2="990" stroke="#D9E2EC" '
          'stroke-width="1.5"/>')
    s.text(40, 1022, "Causal methods throughout: effect modification is not causation; "
           "named estimators; sensitivity analysis; evidence graded by design.",
           size=14.5, weight=600)
    lx = 1090
    for status in ("BUILT", "PROTOTYPE", "PLANNED"):
        lx += s.badge(lx, 1008, status) + 14
    s.add(f'<rect x="{lx + 4}" y="1006" width="40" height="24" rx="6" fill="none" '
          f'stroke="{LINE}" stroke-width="2" stroke-dasharray="7 5"/>')
    s.text(lx + 52, 1022.5, "dashed = not started", size=12.5, fill=MUTED)

    s.add("</svg>")
    return s.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "schematic.svg")
    parser.add_argument("--png", action="store_true",
                        help="also write a PNG at 2x resolution (needs cairosvg)")
    args = parser.parse_args()
    unknown = set(STATUSES.values()) - set(BADGE)
    if unknown:
        parser.error(f"unknown status in STATUSES: {sorted(unknown)}")

    svg = draw()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"Written {args.out}")
    if args.png:
        import cairosvg  # optional dependency, only for PNG output

        png = args.out.with_suffix(".png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=2)
        print(f"Written {png}")


if __name__ == "__main__":
    main()
