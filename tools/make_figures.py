"""Draw every figure in the README, in one visual system: an editorial broadcast look on carbon.

    python tools/make_figures.py

No packages needed. Each figure is a single SVG with CSS animation that plays once when the page
opens (the hero's light trail, the divider and the scoring explainer loop). Viewers who ask for
reduced motion get the finished picture. Headlines are set in Instrument Serif and everything else
in Instrument Sans (both SIL Open Font License, assets/fonts/OFL.txt), embedded in each file so they
look the same everywhere.

Every number is typed in below next to its source. The competition data belongs to the
organisers and is not in this repository.
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
FONTS = ROOT / "assets" / "fonts"
W = 1280

# One accent per job. Lime is NextToken (us); purple is the best score of a round; bronze is the
# certificate; greys are everyone else. Lime and purple pass the colour-blindness and 3:1 contrast
# checks on the carbon background.
C = dict(carbon="#0b0d10", carbon2="#111419", panel="#151920", weave="#0f1216", line="#232830",
         line2="#343a44", ink="#f5f7fa", ink2="#c3c9d4", muted="#8b93a1", lime="#c6ff3d",
         purple="#a970ff", bronze="#e3a35b", red="#ff3b3b", rival="#4a515d", chip="#2b3039")
SANS = '"Instrument Sans", system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
SERIF = '"Instrument Serif", Georgia, "Times New Roman", serif'


def font_faces(only=None):
    """The faces we use (or only the one named), embedded as data URIs."""
    faces = []
    for name, family, weight in [("instrument-sans-400", "Instrument Sans", 400),
                                 ("instrument-sans-600", "Instrument Sans", 600),
                                 ("instrument-serif-400", "Instrument Serif", 400)]:
        if only and name != only:
            continue
        data = base64.b64encode((FONTS / f"{name}.woff2").read_bytes()).decode()
        faces.append(f'@font-face {{ font-family: "{family}"; font-weight: {weight}; font-style: normal; '
                     f"src: url(data:font/woff2;base64,{data}) format('woff2'); }}")
    return "\n".join(faces)


CSS = """
  text { font-family: SANSF; font-weight: 400; fill: INK; }
  .b7 { font-weight: 600; } .b8 { font-family: SERIFF; font-weight: 400; }
  .ink2 { fill: INK2; } .muted { fill: MUTED; } .lime { fill: LIME; } .purple { fill: PURPLE; }
  .bronze { fill: BRONZE; } .dark { fill: CARBON; } .caps { letter-spacing: 1.8px; }
  .grow { transform-box: fill-box; transform-origin: 50% 100%; animation: grow .9s cubic-bezier(.2,.8,.2,1) both; }
  .growx { transform-box: fill-box; transform-origin: 0% 50%; animation: growx .9s cubic-bezier(.2,.8,.2,1) both; }
  .fade { animation: fade .6s ease-out both; }
  .slide { animation: slide .7s cubic-bezier(.2,.8,.2,1) both; }
  .rise { animation: rise .7s cubic-bezier(.2,.8,.2,1) both; }
  .pop { transform-box: fill-box; transform-origin: 50% 50%; animation: pop .5s cubic-bezier(.3,1.5,.5,1) both; }
  .draw { stroke-dasharray: 1; animation: draw 1.6s cubic-bezier(.5,0,.3,1) both; }
  .blink { animation: blink 1.4s steps(2, jump-none) infinite; }
  @keyframes grow { from { transform: scaleY(0); } }
  @keyframes growx { from { transform: scaleX(0); } }
  @keyframes fade { from { opacity: 0; } }
  @keyframes slide { from { opacity: 0; transform: translateX(-36px); } }
  @keyframes rise { from { opacity: 0; transform: translateY(14px); } }
  @keyframes pop { from { opacity: 0; transform: scale(.2); } }
  @keyframes draw { from { stroke-dashoffset: 1; } }
  @keyframes blink { 50% { opacity: .15; } }
"""
REDUCED = "\n  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }\n"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size, cls="", anchor="start", delay=None, extra=""):
    """One line of text. cls picks weight and colour; delay (s) staggers its entrance."""
    a = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}"'
    if anchor != "start":
        a += f' text-anchor="{anchor}"'
    if cls:
        a += f' class="{cls}"'
    if delay is not None:
        a += f' style="animation-delay:{delay:.2f}s"'
    return f"<text {a}{extra}>{s if s.startswith('<tspan') else esc(s)}</text>"


def column(x, y_top, y_base, width, fill, delay):
    """A column with a 4px rounded top, square on the baseline."""
    r = min(4.0, (y_base - y_top) / 2, width / 2)
    d = (f"M{x:.1f},{y_base:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
         f"H{x + width - r:.1f} Q{x + width:.1f},{y_top:.1f} {x + width:.1f},{y_top + r:.1f} V{y_base:.1f} Z")
    return f'<path d="{d}" fill="{fill}" class="grow" style="animation-delay:{delay:.2f}s"/>'


def card(h, kicker, title, subtitle, body, label, desc, source="", css="", defs="", width=W, footer=True):
    """A carbon panel with the broadcast header: accent tab, kicker, title, subtitle."""
    W = width
    subs = subtitle if isinstance(subtitle, list) else [subtitle]
    head = [f'<rect x="0" y="34" width="8" height="64" fill="{C["lime"]}" class="growx"/>',
            t(40, 56, kicker.upper(), 16, "b7 muted caps fade"),
            t(38, 100, title, 46, "b8 rise", delay=0.05)]
    head += [t(40, 134 + 27 * i, line, 21, "ink2 fade", delay=0.15) for i, line in enumerate(subs)]
    foot = [t(40, h - 24, source.upper(), 14, "muted caps")] if source else []
    if footer:
        foot.append(t(W - 40, h - 24, "NEXTTOKEN · TECHARENA 2026", 14, "muted caps", anchor="end"))
    style = (font_faces() + CSS.replace("SANSF", SANS).replace("SERIFF", SERIF).replace("INK2", C["ink2"]).replace("INK", C["ink"])
             .replace("MUTED", C["muted"]).replace("LIME", C["lime"]).replace("PURPLE", C["purple"])
             .replace("BRONZE", C["bronze"]).replace("CARBON", C["carbon"]) + css + REDUCED)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" role="img" aria-labelledby="title desc">
<title id="title">{esc(label)}</title>
<desc id="desc">{esc(desc)}</desc>
<style>{style}</style>
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["carbon2"]}"/><stop offset="1" stop-color="{C["carbon"]}"/></linearGradient>
<pattern id="weave" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="{C["carbon"]}"/><rect width="3" height="6" fill="{C["weave"]}"/></pattern>
<radialGradient id="spot" cx="0.18" cy="0" r="0.9"><stop offset="0" stop-color="#ffffff" stop-opacity="0.06"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/></radialGradient>
<clipPath id="frame"><rect width="{W}" height="{h}" rx="16"/></clipPath>
<filter id="glow" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
{defs}
</defs>
<g clip-path="url(#frame)">
<rect width="{W}" height="{h}" fill="url(#weave)"/>
<rect width="{W}" height="{h}" fill="url(#bg)" opacity="0.55"/>
<rect width="{W}" height="{h}" fill="url(#spot)"/>
{chr(10).join(head)}
{body}
{chr(10).join(foot)}
</g>
<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="16" fill="none" stroke="{C["line"]}"/>
</svg>
"""


# ---------------------------------------------------------------------------------------------
# Column charts
# ---------------------------------------------------------------------------------------------
def column_chart(*, h, n_sub, cats, values, ymax, ticks, tick_fmt, val_fmt, notes=None, ref=None,
                 ref_label="", ref_in_legend=False, markers=None, marker_labels=None, marker_name="",
                 divider=None, divider_label=""):
    """One series of lime columns. Optional: a reference line, a purple marker per column
    (a bullet chart: value to the left of the column, marker label to its right) and a divider."""
    x0, x1 = 120, W - 60
    legend_y = 134 + 27 * n_sub + 16
    has_legend = markers is not None or ref_in_legend
    y_top = legend_y + (64 if has_legend else 36)
    y_base = h - (140 if notes else 120)

    def y(v):
        return y_base - (y_base - y_top) * v / ymax

    b = []
    if has_legend:
        lx = 40
        if markers is not None:     # two kinds of mark: name both (a single series needs no legend)
            b.append(f'<rect x="40" y="{legend_y - 13}" width="14" height="14" rx="2" fill="{C["lime"]}"/>')
            b.append(t(62, legend_y, "NextToken", 18, "ink2"))
            lx = 190
            b.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{legend_y - 6}" y2="{legend_y - 6}" stroke="{C["purple"]}" '
                     f'stroke-width="5" stroke-linecap="round"/>')
            b.append(t(lx + 32, legend_y, marker_name, 18, "ink2"))
        if ref_in_legend:
            b.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{legend_y - 6}" y2="{legend_y - 6}" stroke="{C["ink2"]}" stroke-width="2"/>')
            b.append(t(lx + 32, legend_y, ref_label, 18, "ink2"))
    for v in ticks:
        b.append(f'<line x1="{x0}" x2="{x1}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{C["line"]}"/>')
        b.append(t(x0 - 16, y(v) + 6, tick_fmt(v), 17, "muted", anchor="end"))
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{y_base}" y2="{y_base}" stroke="{C["line2"]}" stroke-width="1.5"/>')
    band, bw = (x1 - x0) / len(cats), 32
    if divider is not None:
        dx = x0 + band * divider
        b.append(f'<line x1="{dx:.1f}" x2="{dx:.1f}" y1="{y_top - 30}" y2="{y_base}" stroke="{C["line2"]}"/>')
        b.append(t(dx + 12, y_top - 14, divider_label + "  →", 16, "muted"))
    for i, cat in enumerate(cats):
        cx = x0 + band * (i + 0.5)
        bx, v, delay = cx - bw / 2, values[i], 0.25 + 0.12 * i
        b.append(column(bx, y(v), y_base, bw, C["lime"], delay))
        if markers is not None:
            b.append(t(bx - 20, y(v) + 8, val_fmt(v), 24, "b7 fade", anchor="end", delay=delay + 0.5))
            my, md = y(markers[i]), 1.0 + 0.12 * i
            b.append(f'<line x1="{cx - 20:.1f}" x2="{cx + 20:.1f}" y1="{my:.1f}" y2="{my:.1f}" stroke="{C["purple"]}" '
                     f'stroke-width="5" stroke-linecap="round" class="fade" style="animation-delay:{md:.2f}s"/>')
            b.append(t(cx + 30, my + 6, marker_labels[i], 18, "ink2 fade", delay=md))
        elif ref is not None and 0 <= y(v) - y(ref) < 36:     # just under the reference line: label beside the top
            b.append(t(bx + bw + 10, y(v) + 18, val_fmt(v), 24, "b7 fade", delay=delay + 0.5))
        else:
            b.append(t(cx, y(v) - 12, val_fmt(v), 24, "b7 fade", anchor="middle", delay=delay + 0.5))
        lines = cat if isinstance(cat, (list, tuple)) else [cat]
        b.append(t(cx, y_base + 32, lines[0], 21, "b7 ink2", anchor="middle"))
        for j, line in enumerate(lines[1:]):
            b.append(t(cx, y_base + 55 + 20 * j, line, 16, "muted", anchor="middle"))
        if notes:
            b.append(t(cx, y_base + 55 + 20 * (len(lines) - 1), notes[i], 16, "muted", anchor="middle"))
    if ref is not None:
        b.append(f'<line x1="{x0}" x2="{x1}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" stroke="{C["ink2"]}" stroke-width="2" '
                 f'class="fade" style="animation-delay:1.1s"/>')
        if not ref_in_legend:
            b.append(t(x1, y(ref) - 10, ref_label, 18, "ink2 fade", anchor="end", delay=1.2))
    return "\n".join(b)


def fig_rounds():
    # Source: the organisers' leaderboard, as recorded in the team's round log. Round 3's top score
    # was ours; round 4's top was 74.15. Final total = (R1 + R2 + 2*R3 + 2*R4) / 6 = 53.19.
    ours, top = [17.40, 46.97, 53.71, 73.67], [38.05, 52.16, 53.71, 74.15]
    h, sub = 580, ["Credit × 100, averaged over every request. Rounds 3 and 4 count double."]
    body = column_chart(h=h, n_sub=len(sub), values=ours, markers=top, marker_name="Top score of the round, any team",
                        marker_labels=["Top 38.05", "Top 52.16", "Top: ours", "Top 74.15"],
                        cats=[("Round 1", "Sat"), ("Round 2", "Sat"), ("Round 3", "Sun · counts ×2"),
                              ("Round 4", "Sun · counts ×2")],
                        ymax=80, ticks=[0, 20, 40, 60, 80], tick_fmt=lambda v: f"{v:g}", val_fmt=lambda v: f"{v:.2f}",
                        divider=2, divider_label="New simulated world")
    return card(h, "Phase 2 · the final", "Four rounds, two days", sub, body,
                "Round scores: NextToken against the top score of each round",
                "Round 1: NextToken 17.40, top 38.05. Round 2: 46.97, top 52.16. Round 3, new simulated world, "
                "double weight: 53.71, the top score of all 15 teams. Round 4, double weight: 73.67, top 74.15.",
                source="Source: organisers' leaderboard")


def fig_recency():
    # Source: round 3 history (20,000 users x 100 random 3-item slates). Items the user had clicked
    # before, shown again: click rate by hours since that click. Aggregates computed for this repo.
    cats = ["< 3 h", "3–6 h", "6–12 h", "12–24 h", "1–2 days", "2+ days"]
    vals, n = [24.2, 16.2, 7.7, 6.9, 5.0, 3.7], [256, 291, 493, 939, 1748, 1157]
    h, sub = 580, ["Chance a past click is clicked again when it is shown, by time since that click."]
    body = column_chart(h=h, n_sub=len(sub), cats=cats, values=vals, ymax=30, ticks=[0, 10, 20, 30],
                        tick_fmt=lambda v: f"{v:g}%", val_fmt=lambda v: f"{v:.1f}%",
                        notes=[f"n = {k:,}" for k in n], ref=1.8, ref_in_legend=True,
                        ref_label="An item the user never clicked: 1.8%")
    return card(h, "Round 3 · what users do", "People re-click what they just clicked", sub, body,
                "Round 3: re-click rate by time since the click",
                "Under 3 hours 24.2%, 3 to 6 hours 16.2%, 6 to 12 hours 7.7%, 12 to 24 hours 6.9%, 1 to 2 days 5.0%, "
                "over 2 days 3.7%. An item the user never clicked: 1.8%.",
                source="Source: round 3 history, random 3-item slates")


def fig_own_clicks():
    # Source: round 3 pooled slates (all teams' picks, released before round 4): 110,201 slates with
    # a click, 2.9 items each. Wins / fair share (1 / items in the slate), by how many times the
    # user had clicked the item before. Aggregates computed for this repo.
    cats = ["Never", "Once", "Twice", "3 times", "4+ times"]
    vals, n = [0.44, 0.55, 1.04, 1.39, 1.93], [30981, 144633, 52285, 27977, 63689]
    h, sub = 580, ["Wins ÷ fair share in round 3's pooled slates, by how often the user had clicked the item before."]
    body = column_chart(h=h, n_sub=len(sub), cats=cats, values=vals, ymax=2.0, ticks=[0, 0.5, 1.0, 1.5, 2.0],
                        tick_fmt=lambda v: f"{v:g}×", val_fmt=lambda v: f"{v:.2f}×",
                        notes=[f"{k / 1000:.0f}K showings" for k in n], ref=1.0, ref_label="Fair share")
    return card(h, "Round 4 · which past click", "The more they clicked it, the more it wins", sub, body,
                "Round 4: wins against fair share by the user's past clicks on the item",
                "Never clicked 0.44 times fair share, clicked once 0.55, twice 1.04, 3 times 1.39, 4 or more 1.93.",
                source="Source: round 3 pooled slates, 110,201 clicks")


def fig_fresh():
    # Source: the team's round 2 analysis: the two 'strong' general items we played in round 2, for
    # users new to the platform who had not clicked them; wins / fair share by times already shown.
    cats = ["First showing", ("Shown once", "and ignored"), ("Shown 3–4", "times"), ("Shown 9+", "times")]
    h, sub = 560, ["The two items we backed in round 2, for new users: wins ÷ fair share by times already shown."]
    body = column_chart(h=h, n_sub=len(sub), cats=cats, values=[0.93, 0.40, 0.28, 0.05], ymax=1.2,
                        ticks=[0, 0.4, 0.8, 1.2], tick_fmt=lambda v: f"{v:g}×", val_fmt=lambda v: f"{v:.2f}×",
                        ref=1.0, ref_label="Fair share")
    return card(h, "Round 2 · the lesson", "Fresh only looks strong until it is ignored", sub, body,
                "Round 2: a fresh item's pull collapses once it is shown and ignored",
                "First showing 0.93 times fair share; after one ignored showing 0.40; after 3 to 4 showings 0.28; "
                "after 9 or more 0.05.", source="Source: round 2 results")


# ---------------------------------------------------------------------------------------------
# Phase 1: public AUC over 42 uploads (the whole week, then the last two days zoomed in)
# ---------------------------------------------------------------------------------------------
UPLOADS = [  # (day, upload, public AUC), from the Phase 1 log; the public leader's best was 0.834673
    ("30 Oct", "v1", 0.7813), ("30 Oct", "v2", 0.788295), ("30 Oct", "v3", 0.797322), ("30 Oct", "v4a", 0.818623),
    ("31 Oct", "v4i", 0.823468), ("31 Oct", "v5", 0.826373), ("31 Oct", "v6b", 0.826776), ("31 Oct", "v7g", 0.827132),
    ("31 Oct", "v7u", 0.827224), ("1 Nov", "g30", 0.828377), ("1 Nov", "gx60", 0.831325), ("1 Nov", "gx80", 0.831311),
    ("1 Nov", "mx70", 0.833218), ("2 Nov", "mx70s", 0.833356), ("2 Nov", "u60", 0.833785), ("2 Nov", "u70", 0.833944),
    ("2 Nov", "u80", 0.833955), ("2 Nov", "u18", 0.834110), ("2 Nov", "w36", 0.834163), ("2 Nov", "v15", 0.834130),
    ("2 Nov", "q12", 0.834208), ("2 Nov", "q18x", 0.834362), ("3 Nov", "p1", 0.834289), ("3 Nov", "r1", 0.834293),
    ("3 Nov", "r2", 0.834384), ("3 Nov", "r3", 0.834351), ("3 Nov", "r5", 0.834395), ("3 Nov", "r6", 0.834373),
    ("3 Nov", "r7", 0.834295), ("3 Nov", "r8", 0.834438), ("3 Nov", "r9", 0.834414), ("3 Nov", "r13", 0.834398),
    ("4 Nov", "r14", 0.834446), ("4 Nov", "r21", 0.834451), ("4 Nov", "r19", 0.834483), ("4 Nov", "r20", 0.834447),
    ("4 Nov", "r17", 0.834492), ("4 Nov", "r22", 0.834517), ("4 Nov", "r24", 0.834534), ("4 Nov", "r25", 0.834560),
    ("4 Nov", "r27", 0.834583), ("4 Nov", "r29", 0.834618),
]
LEADER = 0.834673


def panel(*, first, x0, x1, y0, y1, lo, hi, ticks, tick_fmt, notes, title, start, leader_label=False):
    """Best score so far as a lime step trace; every upload as a dot (filled when it set a new best)."""
    pts, n = UPLOADS[first:], len(UPLOADS) - first

    def X(i):
        return x0 + (x1 - x0) * (i + 0.5) / n

    def Y(v):
        return y1 - (y1 - y0) * (v - lo) / (hi - lo)

    out = [t(x0, y0 - 26, title.upper(), 17, "b7 muted caps")]
    for v in ticks:
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{C["line"]}"/>')
        out.append(t(x0 - 14, Y(v) + 6, tick_fmt(v), 16, "muted", anchor="end"))
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{C["line2"]}" stroke-width="1.5"/>')
    days = [p[0] for p in pts]
    starts = [i for i in range(n) if i == 0 or days[i] != days[i - 1]]
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        if i:
            xs = x0 + (x1 - x0) * i / n
            out.append(f'<line x1="{xs:.1f}" x2="{xs:.1f}" y1="{y0}" y2="{y1}" stroke="{C["line"]}"/>')
        out.append(t(x0 + (x1 - x0) * (i + end) / 2 / n, y1 + 28, days[i], 15, "muted", anchor="middle"))
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(LEADER):.1f}" y2="{Y(LEADER):.1f}" stroke="{C["purple"]}" stroke-width="2.5"/>')
    if leader_label:
        out.append(t(x1, Y(LEADER) - 10, "Public leader 0.834673", 17, "ink2", anchor="end"))
    best_before = max((p[2] for p in UPLOADS[:first]), default=-1.0)
    best, d = best_before, f"M{x0:.1f}"
    for i, (_, _, v) in enumerate(pts):
        new = max(best, v)
        if i == 0:
            d += f",{Y(new):.1f} H{X(0):.1f}"
        elif new != best:
            d += f" H{X(i):.1f} V{Y(new):.1f}"            # step up where an upload set a new best
        best = new
    d += f" H{x1:.1f}"
    out.append(f'<path d="{d}" pathLength="1" fill="none" stroke="{C["lime"]}" stroke-width="3" stroke-linejoin="round" '
               f'class="draw" style="animation-delay:{start:.2f}s"/>')
    best = best_before
    for i, (_, name, v) in enumerate(pts):
        is_best, best = v > best, max(best, v)
        delay = start + 1.6 * (i + 0.5) / n
        if is_best:
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5.5" fill="{C["lime"]}" stroke="{C["carbon"]}" '
                       f'stroke-width="2.5" class="pop" style="animation-delay:{delay:.2f}s"/>')
        else:
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{C["carbon"]}" stroke="{C["muted"]}" '
                       f'stroke-width="2" class="pop" style="animation-delay:{delay:.2f}s"/>')
        if name in notes and notes[name][1] is None:      # label level with the dot, no leader
            label, _, anchor = notes[name]
            out.append(t(X(i) - 14, Y(v) + 6, label, 18, "b7 fade", anchor="end", delay=delay + .3))
        elif name in notes:          # elbow callout: straight down from the dot, then the label beside it
            label, ty, anchor = notes[name]
            side = 1 if anchor == "start" else -1
            out.append(f'<path d="M{X(i):.1f},{Y(v) + 9:.1f} V{Y(ty):.1f} h{6 * side}" fill="none" stroke="{C["line2"]}" '
                       f'stroke-width="1.5" class="fade" style="animation-delay:{delay + .3:.2f}s"/>')
            out.append(t(X(i) + 11 * side, Y(ty) + 6, label, 17, "ink2 fade", anchor=anchor, delay=delay + .3))
    return out


def fig_phase1():
    h, sub = 600, ["Filled dots set a new best. Purple: the public leader, 0.834673. Ours finished at 0.834618."]
    body = panel(first=0, x0=104, x1=660, y0=230, y1=500, lo=0.776, hi=0.8385, ticks=[0.78, 0.80, 0.82, 0.84],
                 tick_fmt=lambda v: f"{v:.2f}", title="The whole week", start=0.2,
                 notes={"v4a": ("11-way choice model", 0.806, "start"), "g30": ("GPU Gibbs sampler", 0.812, "start"),
                        "gx60": ("What users are shown", 0.818, "start"),
                        "mx70": ("Cleaned feedback loop", 0.824, "start")})
    body += panel(first=22, x0=800, x1=W - 60, y0=230, y1=500, lo=0.83424, hi=0.83472,
                  ticks=[0.8343, 0.8344, 0.8345, 0.8346, 0.8347], tick_fmt=lambda v: f"{v:.4f}",
                  title="The last two days, zoomed in", start=1.2, leader_label=True,
                  notes={"r8": ("More sampler runs", 0.83431, "start"), "r19": ("Newest 3 clicks", 0.83439, "start"),
                         "r29": ("Final 0.834618", None, "end")})
    return card(h, "Phase 1 · online · 30 Oct – 4 Nov", "42 uploads to 0.834618", sub, "\n".join(body),
                "Phase 1: public leaderboard AUC over 42 uploads",
                "AUC rose from 0.7813 to 0.8332 in three days (11-way choice model, GPU Gibbs sampler, modelling what "
                "users are shown, cleaning the feedback loop), then by small steps to 0.834618 on 4 November, 0.000055 "
                "below the public leader's 0.834673.", source="Source: public leaderboard, every upload")


# ---------------------------------------------------------------------------------------------
# Hero: the classification, a timing tower and the position after each round
# ---------------------------------------------------------------------------------------------
def draw_arena(b, css, *, cx, cy, radius, gap, ux, head_y, team_x):
    """How the arena works, as a network: 15 teams each pick one item, the picks pool into one slate,
    and the user picks from it. NextToken's path is lime, and a light keeps running along it."""
    import math
    picks = "AABABCBABCACBBC"            # an illustrative spread of 15 picks over three items
    item_y = {"A": cy - gap, "B": cy, "C": cy + gap}
    ours = 6                              # NextToken's node, on item B
    share = {"A": 25, "B": 55, "C": 20}
    b.append(t(team_x, head_y, "15 TEAMS", 14, "b7 muted caps fade", delay=0.3))
    b.append(t(cx, head_y, "ONE POOLED SLATE", 14, "b7 muted caps fade", anchor="middle", delay=0.3))
    b.append(t(ux, head_y, "THE USER", 14, "b7 muted caps fade", anchor="middle", delay=0.3))
    nodes = []
    for i in range(15):
        a = math.radians(-62 + 124 * i / 14)
        nodes.append((cx - radius * math.cos(a), cy + radius * math.sin(a)))
    for i, (x, y) in enumerate(nodes):
        iy = item_y[picks[i]]
        stroke, width = (C["lime"], 3) if i == ours else (C["line2"], 1.5)
        b.append(f'<path d="M{x:.1f},{y:.1f} C{x + 0.4 * radius:.1f},{y:.1f} {cx - 0.5 * radius:.1f},{iy} {cx - 26},{iy}" '
                 f'pathLength="1" fill="none" stroke="{stroke}" stroke-width="{width}" class="draw" '
                 f'style="animation-delay:{0.5 + 0.06 * i:.2f}s;animation-duration:1.1s"/>')
    for i, (x, y) in enumerate(nodes):
        mine = i == ours
        b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{9 if mine else 6}" fill="{C["lime"] if mine else C["rival"]}" '
                 f'stroke="{C["carbon"]}" stroke-width="3" class="pop" style="animation-delay:{0.35 + 0.05 * i:.2f}s"/>')
    nx, ny = nodes[ours]
    b.append(t(nx - 18, ny + 6, "NextToken", 19, "b7 fade", anchor="end", delay=0.9))
    reach = ux - cx
    for item, iy in item_y.items():
        mine = item == "B"
        b.append(f'<path d="M{cx + 26},{iy} C{cx + 0.4 * reach:.1f},{iy} {ux - 0.4 * reach:.1f},{cy} {ux - 34},{cy}" '
                 f'pathLength="1" fill="none" stroke="{C["lime"] if mine else C["line2"]}" '
                 f'stroke-width="{2 + share[item] / 8:.1f}" stroke-linecap="round" class="draw" '
                 f'style="animation-delay:1.7s;animation-duration:.9s"/>')
        b.append(f'<g class="pop" style="animation-delay:{1.3 + 0.1 * "ABC".index(item):.2f}s">'
                 f'<circle cx="{cx}" cy="{iy}" r="26" fill="{C["lime"] if mine else C["chip"]}" stroke="{C["carbon"]}" '
                 f'stroke-width="3"/>' + t(cx, iy + 9, item, 28, "b8 dark" if mine else "b8", anchor="middle") + "</g>")
        # each share sits on the open side of its link: C's link climbs to the user, so its label goes below
        label_y = iy + 32 if item == "C" else iy - 14
        b.append(t(cx + 46, label_y, f"{share[item]}%", 19, "b7 fade" if mine else "muted fade", delay=2.3))
    b.append(f'<g class="pop" style="animation-delay:2.4s"><circle cx="{ux}" cy="{cy}" r="34" fill="{C["panel"]}" '
             f'stroke="{C["ink2"]}" stroke-width="2"/><circle cx="{ux}" cy="{cy - 9}" r="9" fill="{C["ink2"]}"/>'
             f'<path d="M{ux - 16},{cy + 18} Q{ux},{cy - 4} {ux + 16},{cy + 18}" fill="{C["ink2"]}"/></g>')
    b.append(t(ux, cy + 64, "picks B 55% of", 16, "muted fade", anchor="middle", delay=2.6))
    b.append(t(ux, cy + 84, "the time it clicks", 16, "muted fade", anchor="middle", delay=2.6))
    trail = (f"M{nx:.1f},{ny:.1f} C{nx + 0.4 * radius:.1f},{ny:.1f} {cx - 0.5 * radius:.1f},{cy} {cx - 26},{cy} "
             f"L{cx + 26},{cy} C{cx + 0.4 * reach:.1f},{cy} {ux - 0.4 * reach:.1f},{cy} {ux - 34},{cy}")
    b.append(f'<path d="{trail}" pathLength="1" fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round" '
             f'class="comet"/>')
    css.append(".comet { stroke-dasharray: .07 .93; opacity: 0; animation: comet 3.4s linear 3s infinite; }")
    css.append("@keyframes comet { 0% { stroke-dashoffset: 1; opacity: 0; } 10%, 85% { opacity: .95; } "
               "100% { stroke-dashoffset: 0; opacity: 0; } }")


def hero_text(b, *, kicker_right, title_size, title_y, lines, stats_x, stats_y, value_size):
    """The left (or top) half of the title card: name, certificate, a sentence, three numbers."""
    b.append(t(30, title_y, '<tspan>Next</tspan><tspan class="lime">Token</tspan>', title_size, "b8 slide", delay=0.1,
               extra=' letter-spacing="-2"'))
    cy = title_y + 50
    b.append(f'<g class="rise" style="animation-delay:.45s"><circle cx="58" cy="{cy}" r="17" fill="{C["bronze"]}"/>'
             + t(58, cy + 9, "3", 26, "b8 dark", anchor="middle")
             + t(88, cy + 13, "Third Place certificate", 42, "b8 bronze")
             + t(90, cy + 43, "TALENT DEVELOPMENT TRACK  ·  FINAL OF 15 TEAMS", 14, "b7 muted caps") + "</g>")
    for i, line in enumerate(lines):
        b.append(t(40, cy + 100 + 30 * i, line, 23, "ink2 rise", delay=0.6 + 0.05 * i))
    stats = [("53.71", "ROUND 3 SCORE", "highest of all 15 teams"), ("0.834618", "PHASE 1 AUC", "top 15, qualified"),
             ("18 s", "ROUND 4 RUN", "train and answer 400,000")]
    for k, (value, label, note) in enumerate(stats):
        x, dl = stats_x[k], 0.85 + 0.12 * k
        b.append(f'<g class="rise" style="animation-delay:{dl:.2f}s">'
                 f'<line x1="{x}" x2="{x + 40}" y1="{stats_y - 68}" y2="{stats_y - 68}" stroke="{C["lime"]}" stroke-width="3"/>'
                 + t(x - 3, stats_y, value, value_size, "b8") + t(x, stats_y + 32, label, 14, "b7 muted caps")
                 + t(x, stats_y + 56, note, 18, "ink2") + "</g>")


HERO_DESC = ("NextToken: Third Place certificate at the Huawei Ireland TechArena 2026 final. Round 3 score 53.71, the "
             "highest of all 15 teams. Phase 1 AUC 0.834618, top 15. Round 4 entry trained and answered 400,000 "
             "requests in 18 seconds. Diagram: 15 teams each pick one item, the picks pool into one slate, and the "
             "simulated user picks NextToken's item B 55% of the time.")


def fig_hero():
    """The wide title card, for desktop."""
    h, b, css = 640, [], []
    b.append(t(40, 54, "HUAWEI IRELAND TECHARENA 2026  ·  ON-SITE FINAL  ·  DUBLIN, 14–15 NOV", 15, "b7 muted caps fade"))
    b.append(t(W - 40, 54, "RECOMMENDER SYSTEMS", 15, "b7 muted caps fade", anchor="end"))
    b.append(f'<line x1="40" x2="{W - 40}" y1="74" y2="74" stroke="{C["line"]}"/>')
    hero_text(b, kicker_right=True, title_size=150, title_y=214,
              lines=["A recommender that learned what simulated users come back to,",
                     "built over two phases and four live rounds against 14 other teams."],
              stats_x=[40, 250, 500], stats_y=516, value_size=62)
    draw_arena(b, css, cx=1012, cy=352, radius=222, gap=90, ux=1196, head_y=122, team_x=800)
    return card_raw(h, "\n".join(b), "\n  ".join(css), "NextToken at the Huawei Ireland TechArena 2026 final", HERO_DESC)


def fig_hero_portrait():
    """The tall title card, served to phones."""
    w, h, b, css = 720, 1100, [], []
    b.append(t(40, 54, "HUAWEI IRELAND TECHARENA 2026  ·  FINAL", 15, "b7 muted caps fade"))
    b.append(t(w - 40, 54, "DUBLIN", 15, "b7 muted caps fade", anchor="end"))
    b.append(f'<line x1="40" x2="{w - 40}" y1="74" y2="74" stroke="{C["line"]}"/>')
    hero_text(b, kicker_right=False, title_size=140, title_y=206,
              lines=["A recommender that learned what simulated", "users come back to, built over two phases",
                     "and four live rounds against 14 other teams."],
              stats_x=[40, 252, 470], stats_y=572, value_size=52)
    b.append(f'<line x1="40" x2="{w - 40}" y1="672" y2="672" stroke="{C["line"]}"/>')
    draw_arena(b, css, cx=372, cy=896, radius=176, gap=74, ux=612, head_y=712, team_x=40)
    return card_raw(h, "\n".join(b), "\n  ".join(css), "NextToken at the Huawei Ireland TechArena 2026 final",
                    HERO_DESC, width=w)


def card_raw(h, body, css, label, desc, width=W, footer=True):
    """The card frame without the standard header (used by the hero)."""
    svg = card(h, "", "", [], body, label, desc, css=css, width=width, footer=footer)
    # drop the empty standard header (tab, kicker, title) the hero does not use
    return svg.replace(f'<rect x="0" y="34" width="8" height="64" fill="{C["lime"]}" class="growx"/>', "", 1)


# ---------------------------------------------------------------------------------------------
# Scoring explainer (loops): teams pick, duplicates merge, the simulator scores the slate
# ---------------------------------------------------------------------------------------------
def fig_scoring():
    h, period = 700, 11.0
    css, b = [], []

    def anim(name, a, z, kind="fade"):
        """Hidden until a% of the loop, in by z%, held, then out just before the loop restarts."""
        frm = {"fade": "opacity:0", "grow": "opacity:1;transform:scaleX(0)", "draw": "stroke-dashoffset:1",
               "pop": "opacity:0;transform:scale(.2)", "slide": "opacity:0;transform:translateX(-30px)"}[kind]
        to = {"fade": "opacity:1", "grow": "opacity:1;transform:scaleX(1)", "draw": "stroke-dashoffset:0;opacity:1",
              "pop": "opacity:1;transform:scale(1)", "slide": "opacity:1;transform:translateX(0)"}[kind]
        css.append(f"@keyframes {name} {{ 0%,{a}% {{ {frm} }} {z}%,90% {{ {to} }} 96%,100% {{ {to};opacity:0 }} }}")
        return f"animation:{name} {period}s ease-in-out infinite"

    for k, (x, lab) in enumerate([(40, "Teams pick"), (560, "Duplicates merge"), (880, "The simulator scores the slate")]):
        b.append(f'<rect x="{x}" y="208" width="30" height="30" rx="3" fill="{C["lime"]}"/>')
        b.append(t(x + 15, 230, str(k + 1), 19, "b7 dark", anchor="middle"))
        b.append(t(x + 42, 231, lab, 21, "b7 ink2"))
    teams = [("NextToken", "B", True), ("Team 2", "A", False), ("Team 3", "B", False), ("Team 4", "C", False),
             ("Team 5", "A", False)]
    credit = {"A": 25, "B": 55, "C": 20}
    slate_y = {"A": 330, "B": 420, "C": 510}
    for i, (team, item, ours) in enumerate(teams):
        y = 290 + 62 * i
        b.append(f'<g style="{anim(f"team{i}", 0, 3 + 2 * i, "slide")}">'
                 f'<rect x="40" y="{y - 24}" width="330" height="48" rx="4" fill="{C["panel"]}"'
                 f'{" stroke=" + chr(34) + C["lime"] + chr(34) if ours else ""}/>'
                 f'<rect x="40" y="{y - 24}" width="5" height="48" fill="{C["lime"] if ours else C["rival"]}"/>'
                 + t(62, y + 8, team, 21, "b7" if ours else "ink2") + "</g>")
        b.append(f'<g style="{anim(f"cred{i}", 54 + 2 * i, 58 + 2 * i)}">'
                 + t(352, y + 9, f"+{credit[item]}", 30, "b8" + (" lime" if ours else ""), anchor="end") + "</g>")
        b.append(f'<g style="{anim(f"pick{i}", 10 + 2 * i, 14 + 2 * i, "pop")};transform-box:fill-box;transform-origin:50% 50%">'
                 f'<circle cx="418" cy="{y}" r="24" fill="{C["lime"] if ours else C["chip"]}"/>'
                 + t(418, y + 9, item, 26, "b8 dark" if ours else "b8", anchor="middle") + "</g>")
        sy = slate_y[item]
        b.append(f'<path d="M446,{y} C510,{y} 530,{sy} 590,{sy}" pathLength="1" fill="none" '
                 f'stroke="{C["lime"] if ours else C["line2"]}" stroke-width="{3 if ours else 2}" stroke-dasharray="1" '
                 f'style="{anim(f"link{i}", 22 + 2 * i, 34 + 2 * i, "draw")}"/>')
    b.append(t(40, 636, "+ 10 MORE TEAMS", 17, "b7 muted caps"))
    b.append(f'<g style="{anim("slate", 30, 38)}"><rect x="576" y="280" width="270" height="290" rx="8" '
             f'fill="{C["panel"]}" stroke="{C["line2"]}"/>' + t(590, 600, "ONE SLATE · EACH ITEM ONCE", 16, "b7 muted caps")
             + "</g>")
    for item, y in slate_y.items():
        ours = item == "B"
        b.append(f'<g style="{anim(f"slot{item}", 34, 40, "pop")};transform-box:fill-box;transform-origin:50% 50%">'
                 f'<circle cx="618" cy="{y}" r="24" fill="{C["lime"] if ours else C["chip"]}"/>'
                 + t(618, y + 9, item, 26, "b8 dark" if ours else "b8", anchor="middle") + "</g>")
        b.append(f'<g style="{anim(f"lab{item}", 36, 42)}">' + t(656, y + 8, f"Item {item}", 22, "ink2") + "</g>")
        bw = 3.4 * credit[item]
        b.append(f'<rect x="880" y="{y - 16}" width="{bw:.0f}" height="32" rx="3" fill="{C["lime"] if ours else C["rival"]}" '
                 f'style="{anim(f"bar{item}", 42, 50, "grow")};transform-box:fill-box;transform-origin:0% 50%"/>')
        b.append(f'<g style="{anim(f"pct{item}", 48, 52)}">' + t(894 + bw, y + 11, f"{credit[item]}%", 32, "b8") + "</g>")
    b.append(f'<g style="{anim("note", 60, 66)}">' + t(880, 610, "Teams on B get 55 each.", 32, "b8")
             + t(880, 640, "Matching another team costs nothing.", 21, "ink2") + "</g>")
    sub = ["For each request (one user at one moment), every team recommends one item.",
           "Credit = the chance the simulated user picks your item, if they click anything."]
    return card(h, "The rules", "How a round of the final is scored", sub, "\n".join(b),
                "How a round of the TechArena final is scored",
                "Each team recommends one item for the request. Identical picks merge into one slate, here items A, B "
                "and C. The simulated user would pick A 25%, B 55% and C 20% of the time if they click. Every team "
                "that picked B, including NextToken, gets 55; teams on A get 25; the team on C gets 20.",
                css="\n  " + "\n  ".join(css))


# ---------------------------------------------------------------------------------------------
# Divider: a lap trace with sector marks and a lime marker running along it
# ---------------------------------------------------------------------------------------------
def divider():
    h = 44
    marks = "".join(f'<line x1="{x}" x2="{x}" y1="16" y2="28" stroke="{C["line2"]}" stroke-width="2"/>'
                    f'<text x="{x + 8}" y="14" font-size="12" class="b7 muted caps">S{k + 1}</text>'
                    for k, x in enumerate([40, 440, 840]))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" role="presentation">
<style>{font_faces("instrument-sans-600")}
  text {{ font-family: {SANS}; fill: {C["muted"]}; }} .b7 {{ font-weight: 600; }} .caps {{ letter-spacing: 2px; }}
  .car {{ animation: lap 5.5s cubic-bezier(.45,.05,.55,.95) infinite; }}
  @keyframes lap {{ from {{ transform: translateX(0); }} to {{ transform: translateX({W - 120}px); }} }}
  @media (prefers-reduced-motion: reduce) {{ .car {{ animation: none; }} }}
</style>
<defs><linearGradient id="tail" x1="0" x2="1"><stop offset="0" stop-color="{C["lime"]}" stop-opacity="0"/><stop offset="1" stop-color="{C["lime"]}" stop-opacity=".9"/></linearGradient></defs>
<line x1="40" x2="{W - 40}" y1="22" y2="22" stroke="{C["line2"]}" stroke-width="2"/>
{marks}
<g transform="translate({W - 64} 13)"><rect width="24" height="18" fill="{C["ink2"]}"/><rect width="6" height="6" fill="{C["carbon"]}"/><rect x="12" width="6" height="6" fill="{C["carbon"]}"/><rect x="6" y="6" width="6" height="6" fill="{C["carbon"]}"/><rect x="18" y="6" width="6" height="6" fill="{C["carbon"]}"/><rect y="12" width="6" height="6" fill="{C["carbon"]}"/><rect x="12" y="12" width="6" height="6" fill="{C["carbon"]}"/></g>
<g class="car"><rect x="-60" y="20" width="110" height="4" fill="url(#tail)"/><circle cx="52" cy="22" r="6" fill="{C["lime"]}"/></g>
</svg>
"""


# ---------------------------------------------------------------------------------------------
# Folder pages: a header card per folder, and the cards on the front page that link into them
# ---------------------------------------------------------------------------------------------
def folder_header(kicker, title, subtitle, stats):
    """The top of a folder's README: where you are, what is inside, three numbers."""
    h, b = 360, []
    b.append(f'<rect x="0" y="40" width="8" height="74" fill="{C["lime"]}" class="growx"/>')
    b.append(t(40, 62, kicker.upper(), 15, "b7 muted caps fade"))
    b.append(t(36, 122, title, 60, "b8 slide", delay=0.1, extra=' letter-spacing="-0.5"'))
    b.append(t(40, 162, subtitle, 22, "ink2 rise", delay=0.3))
    for k, (value, label, note) in enumerate(stats):
        x, dl = 40 + 400 * k, 0.5 + 0.12 * k
        b.append(f'<g class="rise" style="animation-delay:{dl:.2f}s">'
                 f'<line x1="{x}" x2="{x + 40}" y1="206" y2="206" stroke="{C["lime"]}" stroke-width="3"/>'
                 + t(x - 2, 266, value, 54, "b8") + t(x, 296, label, 14, "b7 muted caps") + t(x, 320, note, 18, "ink2")
                 + "</g>")
    return card_raw(h, "\n".join(b), "", title, f"{kicker}. {title}. {subtitle} "
                    + " ".join(f"{v} {l.lower()}, {n}." for v, l, n in stats), footer=False)


def explore_card(kicker, title, text, path):
    """A wide card on the front page that invites a click into a folder. The arrow keeps nudging."""
    h, b = 170, []
    b.append(f'<rect x="0" y="0" width="8" height="{h}" fill="{C["lime"]}"/>')
    b.append(t(44, 50, kicker.upper(), 14, "b7 muted caps"))
    b.append(t(42, 100, title, 42, "b8"))
    b.append(t(44, 136, text, 20, "ink2"))
    b.append(t(W - 140, 92, path, 16, "b7 muted", anchor="end"))
    b.append(f'<g class="nudge"><circle cx="{W - 84}" cy="86" r="30" fill="{C["lime"]}"/>'
             f'<path d="M{W - 98},86 H{W - 70} M{W - 81},74 L{W - 69},86 L{W - 81},98" fill="none" stroke="{C["carbon"]}" '
             f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></g>')
    css = (".nudge { animation: nudge 2.4s ease-in-out infinite; }\n"
           "  @keyframes nudge { 0%, 60%, 100% { transform: translateX(0); } 75% { transform: translateX(8px); } }")
    return card_raw(h, "\n".join(b), css, title, f"{kicker}: {title}. {text} Opens {path}.", footer=False)


HEADERS = {
    "header-phase1": ("Phase 1 · online · 30 Oct – 4 Nov 2026", "A Bayesian choice model, sampled on a GPU",
                      "Predict which of 10 shown items a user clicks. Scored by ROC AUC over 529,390 rows.",
                      [("0.834618", "PUBLIC AUC", "0.000055 behind the leader"), ("491", "SAMPLER RUNS", "in 11 model families"),
                       ("8", "TASTE DIMENSIONS", "per user and per item")]),
    "header-phase2": ("Phase 2 · on-site final · Dublin, 14–15 Nov 2026", "Four live rounds, one pooled slate",
                      "Recommend one item per request; every team's pick lands in the same slate.",
                      [("53.71", "ROUND 3 SCORE", "highest of all 15 teams"), ("400,000", "REQUESTS PER ROUND", "rounds 3 and 4"),
                       ("240 s", "PER SCORING RUN", "imports included")]),
    "header-round3": ("Phase 2 · round 3 · Sun 10:00–13:00", "Each user's most credible past click",
                      "No training and no weights: every pick is computed from the data when predict() runs.",
                      [("53.71", "ROUND 3 SCORE", "highest of all 15 teams"), ("0", "MODEL WEIGHTS", "rules measured on the history"),
                       ("13 s", "FOR 400,000 REQUESTS", "on a laptop; the limit is 240 s")]),
    "header-round4": ("Phase 2 · round 4 · Sun 14:00–16:00", "A ranker trained inside the scoring run",
                      "LightGBM learns from 94,735 head-to-heads, then ranks each user's own past clicks.",
                      [("73.67", "ROUND 4 SCORE", "rounds 3 and 4 count double"), ("160", "TREES, 7 LEAVES", "95 s budget, rule fallback"),
                       ("18 s", "ON THE ORGANISERS' MACHINE", "train and answer 400,000")]),
}
EXPLORE = {
    "explore-phase1": ("Phase 1 · the model", "Inside the Bayesian sampler",
                       "The 11-way choice model, exposure, the feedback loop, and how 491 runs became one file.", "phase1/"),
    "explore-phase2": ("Phase 2 · the final", "The final, round by round",
                       "The rules, the four rounds, and both entries with the evidence behind them.", "phase2/"),
    "explore-round3": ("Round 3 · top score of the round", "The latest-click entry",
                       "Each user's most credible past click, in one readable file. No training.", "phase2/round3/"),
    "explore-round4": ("Round 4 · trained in the run", "The ranker entry",
                       "LightGBM over each user's past clicks, trained inside the 240-second limit.", "phase2/round4/"),
}

FIGURES = {"hero": fig_hero, "hero-portrait": fig_hero_portrait, "scoring": fig_scoring, "rounds": fig_rounds, "phase1": fig_phase1,
           "recency": fig_recency, "own-clicks": fig_own_clicks, "fresh-items": fig_fresh}
FIGURES.update({name: (lambda a=args: folder_header(*a)) for name, args in HEADERS.items()})
FIGURES.update({name: (lambda a=args: explore_card(*a)) for name, args in EXPLORE.items()})

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        (OUT / f"{name}.svg").write_text(make(), encoding="utf-8")
        print(f"wrote assets/figures/{name}.svg")
    (OUT / "divider.svg").write_text(divider(), encoding="utf-8")
    print("wrote assets/figures/divider.svg")
