"""Visual system for the ATS Optimizer — "The Instrument".

A dark control-panel identity: the IBM Plex superfamily (an engineering
typeface, chosen over the usual serif/acid-green defaults), tabular mono
numerals so every score reads like a meter, and one signature element — a
semicircular gauge whose red/amber/green bands and target tick carry the whole
product thesis: a truthful measurement with an honest ceiling.

Everything renders through st.markdown(unsafe_allow_html=True); no components,
no iframes, so it inherits the page background cleanly.
"""

import html
import math

import streamlit as st

# --- Palette (kept in one place; CSS below reads these literals) -----------
INK = "#13161B"
PANEL = "#1A1E25"
LINE = "#2C333D"
TEXT = "#E7EAEE"
MUTED = "#98A2B0"
ACCENT = "#E7B24C"       # restrained brand signal
BAD = "#E0685A"
MID = "#E7B24C"
GOOD = "#59C08A"
VERIFIED = "#54C7BE"     # truthful micro-accent

TARGET = 80              # the honest target; also the amber→green boundary


def band(score: float) -> str:
    """Semantic colour for a 0–100 score. The boundaries are the product's:
    below 50 the keyword floor is unreachable honestly; 80 is the target."""
    return BAD if score < 50 else (MID if score < TARGET else GOOD)


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {{
  --ink:{INK}; --panel:{PANEL}; --line:{LINE}; --text:{TEXT};
  --muted:{MUTED}; --accent:{ACCENT}; --verified:{VERIFIED};
}}

/* Base type — IBM Plex Sans for prose, mono reserved for data. The explicit
   element list is needed to out-specify Streamlit's own font rules; the .inst-*
   mono classes each carry their own family so they still win by inheritance. */
html, body, .stApp,
.stApp p, .stApp li, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
.stApp {{ background: var(--ink); }}

/* Tighten the default block width a touch; instruments aren't sprawling. */
.block-container {{ max-width: 1080px; padding-top: 3.4rem; }}

/* Masthead ---------------------------------------------------------------- */
.inst-mast {{ margin: 0 0 1.6rem; }}
.inst-eyebrow {{
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem; font-weight: 500;
  letter-spacing: .24em; line-height: 1.6; text-transform: uppercase;
  color: var(--accent); display: flex; align-items: center; gap: .6rem;
}}
.inst-eyebrow::after {{ content:""; height:1px; flex:1; background:var(--line); }}
.inst-title {{
  font-size: 2.5rem; font-weight: 700; line-height: 1.02; letter-spacing: -.02em;
  margin: .5rem 0 .35rem; color: var(--text);
}}
.inst-thesis {{ color: var(--muted); font-size: 1rem; max-width: 46ch; }}
.inst-thesis b {{ color: var(--text); font-weight: 600; }}

/* Section header ---------------------------------------------------------- */
.inst-h {{
  display:flex; align-items:baseline; gap:.7rem; margin: 1.9rem 0 .9rem;
}}
.inst-h .n {{
  font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--accent);
  letter-spacing:.18em;
}}
.inst-h .t {{ font-size:1.15rem; font-weight:600; color:var(--text); letter-spacing:-.01em; }}
.inst-h .r {{ flex:1; height:1px; background:var(--line); align-self:center; }}

/* Cards ------------------------------------------------------------------- */
.inst-card {{
  background: var(--panel); border:1px solid var(--line); border-radius:12px;
  padding: 1.15rem 1.3rem;
}}

/* Gauge ------------------------------------------------------------------- */
.inst-gauge {{ display:flex; gap:1.6rem; align-items:center; flex-wrap:wrap; }}
.inst-readout {{ font-family:'IBM Plex Mono',monospace; }}
.inst-readout .big {{ font-size:3.4rem; font-weight:600; line-height:1; }}
.inst-readout .sub {{ color:var(--muted); font-size:.8rem; letter-spacing:.04em; margin-top:.3rem; }}
.inst-delta {{ font-family:'IBM Plex Mono',monospace; font-size:1rem; font-weight:500; }}

/* Meters ------------------------------------------------------------------ */
.inst-meter {{ margin: .55rem 0 1rem; }}
.inst-meter .row {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; }}
.inst-meter .lab {{ font-size:.92rem; color:var(--text); font-weight:500; }}
.inst-meter .lab .w {{ color:var(--muted); font-family:'IBM Plex Mono',monospace; font-size:.74rem; margin-left:.4rem; }}
.inst-meter .val {{ font-family:'IBM Plex Mono',monospace; font-size:.82rem; color:var(--muted); white-space:nowrap; }}
.inst-meter .val b {{ color:var(--text); }}
.inst-track {{ position:relative; height:8px; background:#11141a; border:1px solid var(--line);
  border-radius:6px; margin:.42rem 0 .3rem; overflow:hidden; }}
.inst-fill {{ position:absolute; inset:0 auto 0 0; border-radius:6px; }}
.inst-meter .det {{ font-size:.76rem; color:var(--muted); }}

/* Chips ------------------------------------------------------------------- */
.inst-chips {{ display:flex; flex-wrap:wrap; gap:.4rem; }}
.inst-chip {{
  font-family:'IBM Plex Mono',monospace; font-size:.78rem; padding:.2rem .55rem;
  border-radius:5px; border:1px solid var(--line); color:var(--text); white-space:nowrap;
}}
.inst-chip.mh   {{ border-color:{ACCENT}; color:#F4D68C; }}
.inst-chip.exact{{ border-color:{VERIFIED}; color:{VERIFIED}; }}
.inst-chip.sem  {{ border-color:{VERIFIED}; color:{VERIFIED}; border-style:dashed; }}
.inst-chip.miss {{ color:var(--muted); opacity:.75; }}
.inst-chip .m {{ opacity:.7; }}
.inst-legend {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--muted); margin-top:.6rem; }}

/* Gap ledger -------------------------------------------------------------- */
.inst-gap {{ display:flex; gap:.7rem; padding:.55rem 0; border-top:1px solid var(--line); }}
.inst-gap:first-child {{ border-top:none; }}
.inst-gap .dot {{ width:9px; height:9px; border-radius:50%; margin-top:.42rem; flex:none; }}
.inst-gap .body .it {{ font-weight:600; color:var(--text); }}
.inst-gap .body .un {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:{BAD};
  letter-spacing:.06em; margin-left:.5rem; }}
.inst-gap .body .ra {{ color:var(--muted); font-size:.86rem; margin-top:.15rem; }}

/* Banner ------------------------------------------------------------------ */
.inst-banner {{ border-radius:10px; padding:.85rem 1.1rem; margin:.4rem 0 1.2rem;
  border:1px solid var(--line); border-left-width:3px; background:var(--panel); }}
.inst-banner.ok {{ border-left-color:{GOOD}; }}
.inst-banner.ceil {{ border-left-color:{MID}; }}
.inst-banner .hd {{ font-weight:600; color:var(--text); }}
.inst-banner .bd {{ color:var(--muted); font-size:.88rem; margin-top:.25rem; }}

/* Nudge Streamlit's primary button toward the instrument look. */
.stButton>button[kind="primary"], .stDownloadButton>button {{
  font-family:'IBM Plex Mono',monospace; font-weight:600; letter-spacing:.04em;
  border-radius:8px;
}}
</style>
"""


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


def _esc(s) -> str:
    return html.escape(str(s))


# --- Gauge ------------------------------------------------------------------

_CX, _CY, _R = 130.0, 138.0, 112.0


def _pt(score: float, radius: float = _R) -> tuple[float, float]:
    """A point on the 180° dial: score 0 → left, 50 → top, 100 → right."""
    theta = math.radians(180 * (1 - max(0.0, min(100.0, score)) / 100))
    return _CX + radius * math.cos(theta), _CY - radius * math.sin(theta)


def _arc(s1: float, s2: float, color: str, width: float) -> str:
    x1, y1 = _pt(s1)
    x2, y2 = _pt(s2)
    return (
        f'<path d="M {x1:.1f} {y1:.1f} A {_R} {_R} 0 0 1 {x2:.1f} {y2:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="butt"/>'
    )


def gauge_svg(score: float, before: float | None = None) -> str:
    nx, ny = _pt(score, _R * 0.80)                       # needle tip
    tix, tiy = _pt(TARGET, _R - 16)
    tox, toy = _pt(TARGET, _R + 8)                       # target tick
    parts = [
        f'<svg viewBox="0 0 260 168" width="260" height="168" '
        f'role="img" aria-label="ATS score {score:.0f} of 100">',
        _arc(0, 50, BAD, 12),
        _arc(50, TARGET, MID, 12),
        _arc(TARGET, 100, GOOD, 12),
        f'<line x1="{tix:.1f}" y1="{tiy:.1f}" x2="{tox:.1f}" y2="{toy:.1f}" '
        f'stroke="{GOOD}" stroke-width="2.5"/>',
        f'<text x="{tox:.1f}" y="{toy - 6:.1f}" fill="{MUTED}" font-size="9" '
        f'font-family="IBM Plex Mono, monospace" text-anchor="middle">{TARGET}</text>',
    ]
    if before is not None:
        bx, by = _pt(before)
        parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="3.2" fill="{MUTED}"/>')
    parts += [
        f'<line x1="{_CX}" y1="{_CY}" x2="{nx:.1f}" y2="{ny:.1f}" '
        f'stroke="{TEXT}" stroke-width="3" stroke-linecap="round"/>',
        f'<circle cx="{_CX}" cy="{_CY}" r="6" fill="{TEXT}"/>',
        "</svg>",
    ]
    return "".join(parts)


def render_gauge(score: float, before: float | None = None):
    col = band(score)
    delta_html = ""
    if before is not None:
        d = score - before
        dc = GOOD if d >= 0 else BAD
        delta_html = (
            f'<div class="inst-delta" style="color:{dc}">{d:+.1f} '
            f'<span style="color:{MUTED};font-weight:400">from {before:.1f}</span></div>'
        )
    st.markdown(
        f'<div class="inst-card inst-gauge">{gauge_svg(score, before)}'
        f'<div class="inst-readout">'
        f'<div class="big" style="color:{col}">{score:.1f}'
        f'<span style="color:{MUTED};font-size:1.1rem;font-weight:400"> / 100</span></div>'
        f'<div class="sub">ATS MATCH SCORE</div>{delta_html}</div></div>',
        unsafe_allow_html=True,
    )


# --- Section header, meters, chips, gaps, banner ---------------------------

def header(num: str, title: str):
    st.markdown(
        f'<div class="inst-h"><span class="n">{_esc(num)}</span>'
        f'<span class="t">{_esc(title)}</span><span class="r"></span></div>',
        unsafe_allow_html=True,
    )


def masthead(title: str, thesis_html: str):
    st.markdown(
        f'<div class="inst-mast"><div class="inst-eyebrow">Truthful résumé tailoring</div>'
        f'<div class="inst-title">{_esc(title)}</div>'
        f'<div class="inst-thesis">{thesis_html}</div></div>',
        unsafe_allow_html=True,
    )


def meters(dims, before_by_name=None):
    """dims: DimensionScore list. Fill width = raw; readout = weighted/max."""
    before_by_name = before_by_name or {}
    rows = []
    for d in dims:
        raw = max(0.0, min(1.0, d.raw))
        maxpts = d.weight * 100
        b = before_by_name.get(d.name)
        was = f' <span style="opacity:.7">· was {b.weighted:.1f}</span>' if b else ""
        rows.append(
            f'<div class="inst-meter"><div class="row">'
            f'<span class="lab">{_esc(d.name.replace("_", " ").title())}'
            f'<span class="w">{maxpts:.0f}% WEIGHT</span></span>'
            f'<span class="val"><b>{d.weighted:.1f}</b> / {maxpts:.0f}{was}</span></div>'
            f'<div class="inst-track"><div class="inst-fill" '
            f'style="width:{raw * 100:.1f}%;background:{ACCENT};opacity:.85"></div></div>'
            + (f'<div class="det">{_esc(d.detail)}</div>' if d.detail else "")
            + "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _chip(text: str, cls: str, marker: str = "") -> str:
    m = f'<span class="m">{marker}</span>' if marker else ""
    return f'<span class="inst-chip {cls}">{_esc(text)}{m}</span>'


def chips(keywords, missing: bool):
    if not keywords:
        st.markdown('<div class="inst-legend">None.</div>', unsafe_allow_html=True)
        return
    out = []
    for k in keywords:
        if missing:
            out.append(_chip(k.keyword, "miss", "★" if k.is_must_have else ""))
        elif k.match_type == "semantic":
            out.append(_chip(k.keyword, "sem", "≈" + ("★" if k.is_must_have else "")))
        else:
            cls = "mh" if k.is_must_have else "exact"
            out.append(_chip(k.keyword, cls, "★" if k.is_must_have else ""))
    st.markdown(f'<div class="inst-chips">{"".join(out)}</div>', unsafe_allow_html=True)


def legend(text: str):
    st.markdown(f'<div class="inst-legend">{_esc(text)}</div>', unsafe_allow_html=True)


_GAP_DOT = {"critical": BAD, "important": MID, "minor": GOOD}


def gap_row(severity: str, item: str, rationale: str, unsupported: bool):
    color = _GAP_DOT.get(severity, MUTED)
    un = '<span class="un">UNSUPPORTED</span>' if unsupported else ""
    st.markdown(
        f'<div class="inst-gap"><span class="dot" style="background:{color}"></span>'
        f'<div class="body"><div class="it">{_esc(item)}{un}</div>'
        f'<div class="ra">{_esc(rationale)}</div></div></div>',
        unsafe_allow_html=True,
    )


def banner(ok: bool, head: str, body: str):
    st.markdown(
        f'<div class="inst-banner {"ok" if ok else "ceil"}">'
        f'<div class="hd">{_esc(head)}</div><div class="bd">{_esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # Gauge geometry self-check: dial endpoints and midpoint land where the
    # 180° mapping says they must.
    x0, y0 = _pt(0)
    x100, y100 = _pt(100)
    x50, y50 = _pt(50)
    assert abs(x0 - (_CX - _R)) < 0.5 and abs(y0 - _CY) < 0.5, (x0, y0)
    assert abs(x100 - (_CX + _R)) < 0.5 and abs(y100 - _CY) < 0.5, (x100, y100)
    assert abs(x50 - _CX) < 0.5 and abs(y50 - (_CY - _R)) < 0.5, (x50, y50)
    assert band(49) == BAD and band(60) == MID and band(80) == GOOD
    assert "<svg" in gauge_svg(63.5, 61.6) and "<path" in gauge_svg(63.5)
    print("ui self-check OK")
