"""Beyond the GPA — UC Berkeley admissions residual dashboard."""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types as genai_types

# --------------------------------------------------------------------------
# Constants grounded in dashboard_data.csv (verified by direct inspection)
# --------------------------------------------------------------------------

BERKELEY_CAMPUS = "Berkeley"
YEAR_MIN, YEAR_MAX = 2022, 2025
SCHOOL_KEY = ["high_school", "city"]
DEFAULT_MIN_APPLICANTS = 20

# expected_admit_rate / admit_rate_residual have zero coverage for fall_term
# 2022 in the source file — every residual-based view therefore reflects
# 2023-2025 even when the year filter includes 2022.
RESIDUAL_COVERAGE_NOTE = (
    "Modeled expected admission rates are available for 2023-2025 only "
    "(not 2022) in the source data. Every view built on the residual "
    "reflects that 3-year window even when 2022 is included in the filter."
)

CHAR_OPTIONS = {
    "Applicant GPA": "applicant_gpa",
    "a-g Completion Rate": "ag_completion_rate",
    "Free/Reduced Lunch %": "frpm_pct",
    "Math Proficiency (% Met)": "caaspp_math_pct_met",
    "ELA Proficiency (% Met)": "caaspp_ela_pct_met",
    "Graduation Rate": "grad_rate",
    "School Size (cohort students)": "cohort_students",
}

COLORS = {
    "bg": "#F1EEEA",
    "card": "#FFFFFF",
    "card_border": "#E2DED7",
    "text": "#1E1E24",
    "muted": "#6D6A67",
    "accent": "#FF6A00",
    "positive": "#15803D",
    "negative": "#DC2626",
    "grid": "#E7E2D8",
}

st.set_page_config(
    page_title="Beyond the GPA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    html, body, [class*="css"] {{
        color: {COLORS['text']};
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    .eyebrow {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: {COLORS['accent']};
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .hero-title {{
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.15;
        color: {COLORS['text']};
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
    }}
    .hero-subtitle {{
        font-size: 1.02rem;
        color: {COLORS['muted']};
        max-width: 780px;
        line-height: 1.55;
        margin-bottom: 1rem;
    }}
    .badge {{
        display: inline-block;
        background: rgba(255, 106, 0, 0.10);
        border: 1px solid rgba(255, 106, 0, 0.35);
        color: {COLORS['accent']};
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 2rem;
    }}

    .section-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {COLORS['muted']};
        margin-bottom: 0.25rem;
    }}
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS['text']};
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }}
    .section-caption {{
        font-size: 0.9rem;
        color: {COLORS['muted']};
        margin-bottom: 1.1rem;
        line-height: 1.5;
        max-width: 820px;
    }}

    .card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(30, 30, 36, 0.06);
    }}

    .kpi-label {{
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {COLORS['muted']};
        margin-bottom: 0.5rem;
    }}
    .kpi-value {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {COLORS['text']};
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.82rem;
        color: {COLORS['muted']};
        margin-top: 0.35rem;
    }}
    .kpi-value.positive {{ color: {COLORS['positive']}; }}
    .kpi-value.negative {{ color: {COLORS['negative']}; }}

    .insight-box {{
        background: rgba(255, 106, 0, 0.06);
        border: 1px solid rgba(255, 106, 0, 0.25);
        border-left: 3px solid {COLORS['accent']};
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.95rem;
        color: {COLORS['text']};
        line-height: 1.55;
        margin: 1rem 0 0.5rem 0;
    }}

    .finding-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 1.4rem 1.4rem 1.2rem 1.4rem;
        height: 100%;
    }}
    .finding-num {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {COLORS['accent']};
        opacity: 0.55;
        margin-bottom: 0.4rem;
    }}
    .finding-title {{
        font-size: 1.02rem;
        font-weight: 700;
        color: {COLORS['text']};
        margin-bottom: 0.5rem;
    }}
    .finding-body {{
        font-size: 0.89rem;
        color: {COLORS['muted']};
        line-height: 1.55;
    }}

    .caveat {{
        font-size: 0.78rem;
        color: {COLORS['muted']};
        font-style: italic;
        margin-top: 0.4rem;
    }}

    hr.divider {{
        border: none;
        border-top: 1px solid {COLORS['card_border']};
        margin: 2.4rem 0 1.6rem 0;
    }}

    div[data-testid="stMetric"] {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] label {{
        color: {COLORS['muted']} !important;
        opacity: 1 !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
    }}

    /* Ensure select / input / slider widgets stay legible on the dark ground
       regardless of the host environment's OS light/dark preference. */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        background-color: {COLORS['card']} !important;
        border-color: {COLORS['card_border']} !important;
        color: {COLORS['text']} !important;
    }}
    div[data-testid="stTextInput"] input::placeholder {{
        color: {COLORS['muted']} !important;
        opacity: 1;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: {COLORS['card']} !important;
    }}
    div[data-testid="stSlider"] [role="slider"] {{
        background-color: {COLORS['accent']} !important;
        border-color: {COLORS['accent']} !important;
    }}
    div[data-baseweb="slider"] div[style*="background-color"] {{
        background-color: {COLORS['accent']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["card"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="-apple-system, Segoe UI, Roboto, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=COLORS["bg"], font_size=12, font_color=COLORS["text"]),
)


# --------------------------------------------------------------------------
# Data loading & preparation
# --------------------------------------------------------------------------

@st.cache_data
def load_raw(path: str = "dashboard_data.csv") -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def prepare_berkeley(df: pd.DataFrame) -> pd.DataFrame:
    b = df[
        (df["campus"] == BERKELEY_CAMPUS)
        & (df["fall_term"] >= YEAR_MIN)
        & (df["fall_term"] <= YEAR_MAX)
    ].copy()

    dup_names = set(
        b.groupby("high_school")["city"].nunique().loc[lambda s: s > 1].index
    )
    b["school_label"] = np.where(
        b["high_school"].isin(dup_names),
        b["high_school"] + " (" + b["city"] + ")",
        b["high_school"],
    )
    return b


def filter_rows(base: pd.DataFrame, year_range, county: str, search: str) -> pd.DataFrame:
    lo, hi = year_range
    r = base[(base["fall_term"] >= lo) & (base["fall_term"] <= hi)]
    if county != "All":
        r = r[r["county"] == county]
    if search:
        r = r[r["school_label"].str.contains(search, case=False, na=False)]
    return r.copy()


def build_school_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per school: applicant-weighted residual metrics + school characteristics.

    Weighted residual = sum(residual_year * applicants_year) / sum(applicants_year),
    computed only over rows where admit_rate_residual is defined (both actual
    and modeled-expected rates present). total_applicants (used for the min-
    applicant filter) is summed over ALL rows in the selected window, so a
    school can appear in the applicant-count filter even in a year without a
    modeled expectation.
    """
    key = SCHOOL_KEY + ["school_label"]

    totals = rows.groupby(key, as_index=False)["applicants"].sum().rename(
        columns={"applicants": "total_applicants"}
    )

    obs_rows = rows.dropna(subset=["admits", "applicants"])
    obs_sums = obs_rows.groupby(key, as_index=False)[["admits", "applicants"]].sum()
    obs_sums["observed_admit_rate"] = obs_sums["admits"] / obs_sums["applicants"]
    obs_sums = obs_sums[key + ["observed_admit_rate"]]

    resid_rows = rows.dropna(
        subset=["admit_rate_residual", "admit_rate", "expected_admit_rate", "applicants"]
    ).copy()
    resid_rows["w_actual"] = resid_rows["admit_rate"] * resid_rows["applicants"]
    resid_rows["w_expected"] = resid_rows["expected_admit_rate"] * resid_rows["applicants"]
    resid_rows["w_residual"] = resid_rows["admit_rate_residual"] * resid_rows["applicants"]

    resid_agg = resid_rows.groupby(key).agg(
        sum_w=("applicants", "sum"),
        sum_w_actual=("w_actual", "sum"),
        sum_w_expected=("w_expected", "sum"),
        sum_w_residual=("w_residual", "sum"),
        n_years_residual=("fall_term", "nunique"),
        pos_years=("admit_rate_residual", lambda s: int((s > 0).sum())),
        neg_years=("admit_rate_residual", lambda s: int((s < 0).sum())),
    ).reset_index()
    resid_agg["actual_admit_rate"] = resid_agg["sum_w_actual"] / resid_agg["sum_w"]
    resid_agg["expected_admit_rate"] = resid_agg["sum_w_expected"] / resid_agg["sum_w"]
    resid_agg["residual"] = resid_agg["sum_w_residual"] / resid_agg["sum_w"]
    resid_agg = resid_agg[
        key
        + [
            "actual_admit_rate",
            "expected_admit_rate",
            "residual",
            "n_years_residual",
            "pos_years",
            "neg_years",
        ]
    ]

    chars = rows.groupby(key, as_index=False).agg(
        county=("county", "first"),
        charter=("charter", "first"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        applicant_gpa=("applicant_gpa", "mean"),
        admit_gpa=("admit_gpa", "mean"),
        ag_completion_rate=("ag_completion_rate", "mean"),
        grad_rate=("grad_rate", "mean"),
        frpm_pct=("frpm_pct", "mean"),
        caaspp_math_pct_met=("caaspp_mathematics_pct_met", "mean"),
        caaspp_ela_pct_met=("caaspp_ela_pct_met", "mean"),
        college_going_rate=("college_going_rate", "mean"),
        cohort_students=("cohort_students", "mean"),
    )

    out = (
        totals.merge(chars, on=key, how="left")
        .merge(obs_sums, on=key, how="left")
        .merge(resid_agg, on=key, how="left")
    )
    out["n_years_residual"] = out["n_years_residual"].fillna(0).astype(int)
    out["pos_share"] = out["pos_years"] / out["n_years_residual"].replace(0, np.nan)
    out["neg_share"] = out["neg_years"] / out["n_years_residual"].replace(0, np.nan)
    out["is_consistent_over"] = (out["n_years_residual"] >= 3) & (out["pos_share"] >= 0.75)
    out["is_consistent_under"] = (out["n_years_residual"] >= 3) & (out["neg_share"] >= 0.75)
    return out


def apply_min_applicants(summary: pd.DataFrame, min_applicants: int) -> pd.DataFrame:
    return summary[summary["total_applicants"] >= min_applicants].copy()


def pearson_r(x: pd.Series, y: pd.Series):
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 5:
        return np.nan, n
    xv, yv = x[m].astype(float), y[m].astype(float)
    if xv.std() == 0 or yv.std() == 0:
        return np.nan, n
    return float(np.corrcoef(xv, yv)[0, 1]), n


def fmt_pp(x, decimals=1, signed=True):
    if pd.isna(x):
        return "—"
    sign = "+" if (signed and x >= 0) else ""
    return f"{sign}{x * 100:.{decimals}f} pp"


def fmt_pct(x, decimals=1):
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x, decimals=0):
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


# --------------------------------------------------------------------------
# Gemini narrative (optional — grounded strictly in numbers already computed
# elsewhere on the page; never used as a source of facts on its own).
# --------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.6-flash"

AI_SYSTEM_INSTRUCTION = (
    "You are a careful data-journalism assistant writing a short narrative blurb for "
    "'Beyond the GPA', a dashboard analyzing UC Berkeley admissions residuals (observed "
    "minus modeled-expected admit rate) for Bay Area public high schools. Rules: "
    "(1) Use ONLY the numbers in the user's JSON payload — never invent, estimate, round "
    "differently, or bring in outside knowledge about any specific school. "
    "(2) Express residuals in percentage points (pp), never plain percent. "
    "(3) Never claim causation; use language like 'associated with', 'outperformed the "
    "modeled expectation', or 'observed pattern'. "
    "(4) This is aggregated school-level data — never make claims about individual "
    "students or anyone's odds of admission. "
    "(5) Write 3-5 plain-language sentences. No bullet points, no markdown, no headers. "
    "(6) If a field is null, omit it instead of guessing a value."
)


def _get_gemini_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GEMINI_API_KEY", "")


@st.cache_resource
def get_gemini_client():
    api_key = _get_gemini_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def generate_ai_narrative(stats: dict) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "No Gemini API key configured. Add GEMINI_API_KEY to .streamlit/secrets.toml "
            "(local) or the app's Secrets settings (Streamlit Cloud)."
        )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=json.dumps(stats, default=str),
        config=genai_types.GenerateContentConfig(
            system_instruction=AI_SYSTEM_INSTRUCTION,
            temperature=0.4,
            # gemini-3.6-flash spends a variable, sometimes large, share of this
            # budget on internal reasoning tokens before the visible reply —
            # values below ~3000 were observed to truncate mid-sentence.
            max_output_tokens=3072,
        ),
    )
    text = (resp.text or "").strip()
    finish_reason = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
    if not text:
        if str(finish_reason) == "FinishReason.MAX_TOKENS":
            raise RuntimeError("Gemini hit its output-length limit before finishing. Please try again.")
        raise RuntimeError("Gemini returned an empty response.")
    return text


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

raw = load_raw()
base = prepare_berkeley(raw)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown('<div class="eyebrow">BEYOND THE GPA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Which Bay Area schools beat Berkeley admissions expectations?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">An interactive analysis of observed vs. expected UC Berkeley freshman '
    "admission outcomes among California public high schools, 2022–2025.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="badge">School-level aggregated data • 2022–2025</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Filter bar
# --------------------------------------------------------------------------

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns([1.3, 1, 1, 1.3])
    with fc1:
        year_range = st.slider(
            "Year range", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), step=1
        )
    with fc2:
        min_applicants = st.number_input(
            "Min. Berkeley applicants (period total)",
            min_value=0,
            max_value=500,
            value=DEFAULT_MIN_APPLICANTS,
            step=5,
        )
    with fc3:
        counties = ["All"] + sorted(base["county"].dropna().unique().tolist())
        county = st.selectbox("County", counties, index=0)
    with fc4:
        search = st.text_input("School search", placeholder="e.g. Lincoln, Berkeley High…")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(RESIDUAL_COVERAGE_NOTE)

rows = filter_rows(base, year_range, county, search)
summary_all = build_school_summary(rows)
summary = apply_min_applicants(summary_all, min_applicants)
thresh_keys = summary[SCHOOL_KEY]
rows_thresh = rows.merge(thresh_keys, on=SCHOOL_KEY)

if summary.empty:
    st.warning(
        "No schools meet the current filter combination. Try lowering the minimum "
        "applicant threshold, choosing a different county, or clearing the search box."
    )
    st.stop()

# --------------------------------------------------------------------------
# Section 1 — Overview
# --------------------------------------------------------------------------

st.markdown('<div class="section-eyebrow">Section 01</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">How large is the observed-vs-expected admissions gap, '
    "and how much of the qualifying applicant pool does it cover?</div>",
    unsafe_allow_html=True,
)

schools_analyzed = len(summary)
total_applicants = int(summary["total_applicants"].sum())

obs_rows_thresh = rows_thresh.dropna(subset=["admits", "applicants"])
overall_admit_rate = (
    obs_rows_thresh["admits"].sum() / obs_rows_thresh["applicants"].sum()
    if obs_rows_thresh["applicants"].sum() > 0
    else np.nan
)

resid_summary = summary.dropna(subset=["residual"])
if not resid_summary.empty:
    top_row = resid_summary.loc[resid_summary["residual"].idxmax()]
    bottom_row = resid_summary.loc[resid_summary["residual"].idxmin()]
    median_gap = float(resid_summary["residual"].median())
else:
    top_row = bottom_row = None
    median_gap = np.nan

n_consistent_over = int(summary["is_consistent_over"].sum())
n_consistent_under = int(summary["is_consistent_under"].sum())
r_gpa, n_gpa = pearson_r(summary["applicant_gpa"], summary["residual"])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="card"><div class="kpi-label">Schools Analyzed</div>
        <div class="kpi-value">{schools_analyzed}</div>
        <div class="kpi-sub">meeting the applicant threshold</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="card"><div class="kpi-label">Berkeley Applicants</div>
        <div class="kpi-value">{total_applicants:,}</div>
        <div class="kpi-sub">total applications, {year_range[0]}–{year_range[1]}</div></div>""",
        unsafe_allow_html=True,
    )
with k3:
    gap_class = "positive" if pd.notna(median_gap) and median_gap >= 0 else ("negative" if pd.notna(median_gap) else "")
    st.markdown(
        f"""<div class="card"><div class="kpi-label">Median Admissions Gap</div>
        <div class="kpi-value {gap_class}">{fmt_pp(median_gap) if pd.notna(median_gap) else "—"}</div>
        <div class="kpi-sub">median residual across qualifying schools</div></div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="card"><div class="kpi-label"># Consistent Overperformers</div>
        <div class="kpi-value positive">{n_consistent_over}</div>
        <div class="kpi-sub">≥3 yrs data, positive gap in ≥75% of years</div></div>""",
        unsafe_allow_html=True,
    )

if top_row is not None:
    st.markdown(
        f"""<div class="insight-box">
        <strong>{top_row['school_label']}</strong> recorded the largest positive admissions
        residual in the selected period, outperforming its modeled expectation by
        <strong>{fmt_pp(top_row['residual'])}</strong>
        ({fmt_pct(top_row['actual_admit_rate'])} actual vs. {fmt_pct(top_row['expected_admit_rate'])} expected,
        across {int(top_row['n_years_residual'])} year(s) of modeled data).
        </div>""",
        unsafe_allow_html=True,
    )

ai_stats = {
    "filters": {
        "year_range": list(year_range),
        "county": county,
        "min_applicants_threshold": int(min_applicants),
    },
    "schools_analyzed": schools_analyzed,
    "total_berkeley_applicants": total_applicants,
    "overall_admit_rate_pct": round(overall_admit_rate * 100, 1) if pd.notna(overall_admit_rate) else None,
    "median_admissions_gap_pp": round(median_gap * 100, 1) if pd.notna(median_gap) else None,
    "top_overperformer": None if top_row is None else {
        "school": top_row["school_label"],
        "residual_pp": round(float(top_row["residual"]) * 100, 1),
        "actual_admit_rate_pct": round(float(top_row["actual_admit_rate"]) * 100, 1),
        "expected_admit_rate_pct": round(float(top_row["expected_admit_rate"]) * 100, 1),
        "years_with_modeled_data": int(top_row["n_years_residual"]),
    },
    "largest_underperformer": None if bottom_row is None else {
        "school": bottom_row["school_label"],
        "residual_pp": round(float(bottom_row["residual"]) * 100, 1),
        "actual_admit_rate_pct": round(float(bottom_row["actual_admit_rate"]) * 100, 1),
        "expected_admit_rate_pct": round(float(bottom_row["expected_admit_rate"]) * 100, 1),
        "years_with_modeled_data": int(bottom_row["n_years_residual"]),
    },
    "consistent_overperformer_count": n_consistent_over,
    "consistent_underperformer_count": n_consistent_under,
    "applicant_gpa_correlation_r": None if np.isnan(r_gpa) else round(r_gpa, 2),
    "applicant_gpa_correlation_n": n_gpa,
}
ai_stats_key = json.dumps(ai_stats, sort_keys=True, default=str)

ai_head_col, ai_btn_col = st.columns([4, 1.4])
with ai_head_col:
    st.markdown(
        '<div class="section-eyebrow" style="margin-top:0.6rem;">AI Narrative (Gemini)</div>'
        '<div class="section-caption" style="margin-bottom:0.4rem;">Sends only the aggregated '
        "statistics already shown above to Gemini — no raw student records — and asks for a short, "
        "strictly-grounded plain-language summary.</div>",
        unsafe_allow_html=True,
    )
with ai_btn_col:
    generate_clicked = st.button("✨ Generate AI narrative", key="gen_ai_narrative", use_container_width=True)

if generate_clicked:
    with st.spinner("Asking Gemini for a narrative grounded in the current filters…"):
        try:
            st.session_state["ai_narrative_text"] = generate_ai_narrative(ai_stats)
            st.session_state["ai_narrative_key"] = ai_stats_key
        except Exception as exc:
            st.session_state.pop("ai_narrative_text", None)
            st.error(f"Couldn't generate an AI narrative: {exc}")

if "ai_narrative_text" in st.session_state:
    st.markdown(
        f'<div class="insight-box" style="border-left-color:#A78BFA; background: rgba(167,139,250,0.08); '
        f'border-color: rgba(167,139,250,0.25);">✨ {st.session_state["ai_narrative_text"]}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("ai_narrative_key") != ai_stats_key:
        st.caption("Filters changed since this narrative was generated — click the button again to refresh it.")
    st.caption("AI-generated from the aggregated statistics above; verify before citing.")

# --------------------------------------------------------------------------
# Section 2 — Actual vs. Expected (hero)
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Section 02</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Actual vs. Expected Berkeley Admission Rate</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Each point is one high school. Points above the diagonal '
    "outperformed their modeled expectation; points below underperformed it. Point size reflects "
    "total Berkeley applicants (capped for readability).</div>",
    unsafe_allow_html=True,
)

hero_df = summary.dropna(subset=["actual_admit_rate", "expected_admit_rate", "residual"]).copy()

if hero_df.empty:
    st.info("No schools with both an actual and expected admit rate in the current filters.")
else:
    size_cap = hero_df["total_applicants"].quantile(0.95)
    hero_df["size_capped"] = hero_df["total_applicants"].clip(upper=size_cap)

    axis_max = max(hero_df["actual_admit_rate"].max(), hero_df["expected_admit_rate"].max()) * 1.08
    axis_min = 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(color=COLORS["muted"], dash="dash", width=1.5),
            name="y = x (expectation)",
            hoverinfo="skip",
        )
    )

    colors = np.where(hero_df["residual"] >= 0, COLORS["positive"], COLORS["negative"])
    customdata = np.stack(
        [
            hero_df["school_label"],
            hero_df["city"],
            hero_df["county"],
            hero_df["total_applicants"],
            hero_df["actual_admit_rate"] * 100,
            hero_df["expected_admit_rate"] * 100,
            hero_df["residual"] * 100,
            hero_df["applicant_gpa"],
            hero_df["ag_completion_rate"],
            hero_df["frpm_pct"] * 100,
            hero_df["n_years_residual"],
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=hero_df["expected_admit_rate"],
            y=hero_df["actual_admit_rate"],
            mode="markers",
            marker=dict(
                size=hero_df["size_capped"],
                sizemode="area",
                sizeref=2.0 * hero_df["size_capped"].max() / (34.0 ** 2),
                sizemin=4,
                color=colors,
                opacity=0.78,
                line=dict(width=0.5, color=COLORS["bg"]),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}, %{customdata[2]} County<br>"
                "Applicants: %{customdata[3]:,.0f}<br>"
                "Actual admit rate: %{customdata[4]:.1f}%<br>"
                "Expected admit rate: %{customdata[5]:.1f}%<br>"
                "Residual: %{customdata[6]:+.1f} pp<br>"
                "Applicant GPA: %{customdata[7]:.2f}<br>"
                "a-g completion: %{customdata[8]:.1f}%<br>"
                "FRPM: %{customdata[9]:.1f}%<br>"
                "Years w/ modeled data: %{customdata[10]:.0f}"
                "<extra></extra>"
            ),
            name="Schools",
            showlegend=False,
        )
    )

    extremes = pd.concat(
        [
            hero_df.nlargest(2, "residual"),
            hero_df.nsmallest(2, "residual"),
        ]
    ).drop_duplicates(subset=SCHOOL_KEY)
    for _, r in extremes.iterrows():
        fig.add_annotation(
            x=r["expected_admit_rate"],
            y=r["actual_admit_rate"],
            text=r["school_label"].title(),
            showarrow=True,
            arrowhead=0,
            arrowcolor=COLORS["muted"],
            ax=25 if r["residual"] >= 0 else -25,
            ay=-25 if r["residual"] >= 0 else 25,
            font=dict(size=10, color=COLORS["text"]),
            bgcolor=COLORS["bg"],
            bordercolor=COLORS["card_border"],
            borderpad=3,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=560,
        xaxis=dict(
            title="Expected admit rate", tickformat=".0%", range=[axis_min, axis_max],
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            title="Actual admit rate", tickformat=".0%", range=[axis_min, axis_max],
            gridcolor=COLORS["grid"],
        ),
        legend=dict(orientation="h", y=1.06, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div class="caveat">Above the dashed line: outperforming expectation · '
        f"Below the dashed line: underperforming expectation · {len(hero_df)} schools shown.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Section 3 — Overperformance leaderboard
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Section 03 · Gap Leaderboard</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Who Beats Expectations Most?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Ranked by applicant-weighted admissions gap (residual). '
    "Only schools meeting the minimum applicant threshold are eligible.</div>",
    unsafe_allow_html=True,
)

direction = st.radio(
    "Show", ["Largest Positive Gaps", "Largest Negative Gaps"], horizontal=True, key="leaderboard_dir"
)

lb_df = summary.dropna(subset=["residual"]).copy()
if lb_df.empty:
    st.info("No schools with a modeled residual in the current filters.")
else:
    n_show = min(15, len(lb_df))
    if direction == "Largest Positive Gaps":
        lb = lb_df.nlargest(n_show, "residual").sort_values("residual")
        bar_color = COLORS["positive"]
    else:
        lb = lb_df.nsmallest(n_show, "residual").sort_values("residual", ascending=False)
        bar_color = COLORS["negative"]

    fig_lb = go.Figure(
        go.Bar(
            x=lb["residual"] * 100,
            y=lb["school_label"],
            orientation="h",
            marker_color=bar_color,
            customdata=np.stack(
                [
                    lb["actual_admit_rate"] * 100,
                    lb["expected_admit_rate"] * 100,
                    lb["total_applicants"],
                    lb["n_years_residual"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Actual: %{customdata[0]:.1f}%<br>Expected: %{customdata[1]:.1f}%<br>"
                "Gap: %{x:+.1f} pp<br>Applicants: %{customdata[2]:,.0f}<br>"
                "Years of data: %{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    fig_lb.add_vline(x=0, line_width=1, line_color=COLORS["grid"])
    fig_lb.update_layout(
        **PLOTLY_LAYOUT,
        height=max(380, 28 * n_show),
        xaxis=dict(title="Admissions gap (percentage points)", gridcolor=COLORS["grid"]),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_lb, use_container_width=True)

# --------------------------------------------------------------------------
# Section 4 — Consistency
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Section 04 · Consistency</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">One Great Year—or a Pattern?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">A <strong>consistent overperformer</strong> is defined here as a school '
    "with at least 3 years of modeled residual data (2023-2025) and a positive residual in at least 75% "
    "of those years — the mirror definition applies to consistent underperformers. Blank cells indicate "
    "no modeled residual for that school-year (including all of 2022).</div>",
    unsafe_allow_html=True,
)

heat_mode = st.selectbox(
    "Schools to display",
    ["Top overperformers + underperformers", "Consistent overperformers", "Consistent underperformers", "Search results"],
    key="heat_mode",
)

cons_df = summary.dropna(subset=["residual"]).copy()
if heat_mode == "Top overperformers + underperformers":
    heat_schools = pd.concat([cons_df.nlargest(8, "residual"), cons_df.nsmallest(8, "residual")])
elif heat_mode == "Consistent overperformers":
    heat_schools = cons_df[cons_df["is_consistent_over"]].sort_values("residual", ascending=False)
elif heat_mode == "Consistent underperformers":
    heat_schools = cons_df[cons_df["is_consistent_under"]].sort_values("residual")
else:
    heat_schools = cons_df

heat_schools = heat_schools.drop_duplicates(subset=SCHOOL_KEY).head(20)

if heat_schools.empty:
    st.info("No schools match this view. Try a different selection or loosen the filters above.")
else:
    heat_keys = heat_schools[SCHOOL_KEY + ["school_label", "residual"]]
    heat_rows = rows.merge(heat_keys[SCHOOL_KEY], on=SCHOOL_KEY)
    pivot = heat_rows.pivot_table(
        index=SCHOOL_KEY, columns="fall_term", values="admit_rate_residual", aggfunc="mean"
    )
    for yr in range(YEAR_MIN, YEAR_MAX + 1):
        if yr not in pivot.columns:
            pivot[yr] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    order = heat_schools.sort_values("residual", ascending=True)[SCHOOL_KEY + ["school_label"]]
    pivot = pivot.reindex(order.set_index(SCHOOL_KEY).index)
    labels = order["school_label"].tolist()

    z = pivot.values * 100
    max_abs = np.nanmax(np.abs(z)) if np.isfinite(z).any() and not np.all(np.isnan(z)) else 10

    fig_heat = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in pivot.columns],
            y=labels,
            colorscale=[
                [0.0, COLORS["negative"]],
                [0.5, "#F5F1E9"],
                [1.0, COLORS["positive"]],
            ],
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            hovertemplate="%{y} — %{x}<br>Residual: %{z:+.1f} pp<extra></extra>",
            colorbar=dict(title="pp", tickformat="+.0f"),
            xgap=3,
            ygap=3,
        )
    )
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        height=max(360, 26 * len(labels)),
        xaxis=dict(title="", type="category", side="bottom"),
        yaxis=dict(title="", autorange="reversed" if False else True),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --------------------------------------------------------------------------
# Section 5 — School Explorer
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Section 05 · School Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Explore a High School</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Select any Bay Area high school to see its own observed-vs-expected '
    "admissions gap and how it has moved year to year.</div>",
    unsafe_allow_html=True,
)

explorer_options = base.sort_values("school_label")[SCHOOL_KEY + ["school_label"]].drop_duplicates()
if county != "All":
    explorer_options = explorer_options.merge(
        base.loc[base["county"] == county, SCHOOL_KEY].drop_duplicates(), on=SCHOOL_KEY
    )

default_idx = 0
if top_row is not None and top_row["school_label"] in explorer_options["school_label"].values:
    default_idx = int(explorer_options["school_label"].tolist().index(top_row["school_label"]))

selected_label = st.selectbox(
    "🔎 Select a high school", explorer_options["school_label"].tolist(), index=default_idx
)
sel_key = explorer_options.loc[explorer_options["school_label"] == selected_label, SCHOOL_KEY].iloc[0]

sel_summary_row = summary_all[
    (summary_all["high_school"] == sel_key["high_school"]) & (summary_all["city"] == sel_key["city"])
]
sel_rows = base[(base["high_school"] == sel_key["high_school"]) & (base["city"] == sel_key["city"])].sort_values(
    "fall_term"
)

if sel_summary_row.empty:
    st.info("No data for this school in the selected years.")
else:
    s = sel_summary_row.iloc[0]
    st.markdown(f"#### {selected_label}")
    if s["total_applicants"] < min_applicants:
        st.caption(
            f"⚠ This school has {int(s['total_applicants'])} total Berkeley applicants in the selected "
            f"window, below the current minimum threshold of {min_applicants}. Figures below can be noisy."
        )

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.metric(
            "Actual Admit Rate",
            fmt_pct(s["actual_admit_rate"]) if pd.notna(s["actual_admit_rate"]) else fmt_pct(s["observed_admit_rate"]),
        )
    with e2:
        st.metric("Expected Rate", fmt_pct(s["expected_admit_rate"]))
    with e3:
        st.metric("Gap", fmt_pp(s["residual"]) if pd.notna(s["residual"]) else "—")
    with e4:
        st.metric("Applicants", f"{int(s['total_applicants']):,}")
    with e5:
        st.metric("Years Analyzed", str(int(s["n_years_residual"])))

    ts = sel_rows[["fall_term", "admit_rate", "expected_admit_rate"]].copy()
    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=ts["fall_term"], y=ts["admit_rate"] * 100, mode="lines+markers", name="Actual",
            line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=8),
            connectgaps=False,
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=ts["fall_term"], y=ts["expected_admit_rate"] * 100, mode="lines+markers", name="Expected",
            line=dict(color=COLORS["muted"], width=2, dash="dash"), marker=dict(size=7),
            connectgaps=False,
        )
    )
    fig_ts.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        xaxis=dict(title="", tickmode="array", tickvals=list(range(YEAR_MIN, YEAR_MAX + 1)), gridcolor=COLORS["grid"]),
        yaxis=dict(title="Admit rate (%)", gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.caption(f"{selected_label.title()}: actual vs. expected admit rate, {YEAR_MIN}–{YEAR_MAX}")
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption("No modeled expected rate exists for fall 2022, so that point is omitted rather than interpolated.")

# --------------------------------------------------------------------------
# Section 6 — What's associated with the gap?
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Section 06 · What\'s Associated?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What Is Associated With Overperformance?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Correlation describes association, not causation. School-level '
    "characteristics are aggregated over the selected window; the model behind expected_admit_rate is not "
    "distributed with the data, so any relationship shown here is descriptive, not an explanation of the model.</div>",
    unsafe_allow_html=True,
)

var_label = st.selectbox("Compare admissions gap against", list(CHAR_OPTIONS.keys()), key="char_var")
var_col = CHAR_OPTIONS[var_label]

char_df = summary.dropna(subset=["residual", var_col]).copy()
if char_df.empty:
    st.info("Not enough data to plot this relationship under the current filters.")
else:
    r, n = pearson_r(char_df[var_col], char_df["residual"])
    fig_char = go.Figure()
    size_cap = char_df["total_applicants"].quantile(0.95)
    fig_char.add_trace(
        go.Scatter(
            x=char_df[var_col],
            y=char_df["residual"] * 100,
            mode="markers",
            marker=dict(
                size=char_df["total_applicants"].clip(upper=size_cap),
                sizemode="area",
                sizeref=2.0 * size_cap / (30.0 ** 2),
                sizemin=4,
                color=COLORS["accent"],
                opacity=0.7,
                line=dict(width=0.5, color=COLORS["bg"]),
            ),
            text=char_df["school_label"],
            hovertemplate=f"<b>%{{text}}</b><br>{var_label}: %{{x:.2f}}<br>Gap: %{{y:+.1f}} pp<extra></extra>",
            showlegend=False,
        )
    )
    if not np.isnan(r) and n >= 5:
        xv = char_df[var_col].astype(float)
        yv = (char_df["residual"] * 100).astype(float)
        coeffs = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ys = np.polyval(coeffs, xs)
        fig_char.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", line=dict(color=COLORS["muted"], dash="dot"),
                showlegend=False, hoverinfo="skip",
            )
        )
    fig_char.update_layout(
        **PLOTLY_LAYOUT,
        height=460,
        xaxis=dict(title=var_label, gridcolor=COLORS["grid"]),
        yaxis=dict(title="Admissions gap (pp)", gridcolor=COLORS["grid"]),
    )
    st.plotly_chart(fig_char, use_container_width=True)

    if not np.isnan(r):
        st.markdown(
            f'<div class="insight-box">r = {r:.2f} (n = {n} schools). '
            "Association ≠ causation.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Not enough variation in the current selection to compute a correlation.")

# --------------------------------------------------------------------------
# Section 7 — Key answer
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Section 07 · Key Answer</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What We Found</div>', unsafe_allow_html=True)

pos_candidates = summary[summary["pos_years"] > 0].sort_values(
    ["pos_share", "n_years_residual", "residual"], ascending=[False, False, False]
)
most_persistent_pos = pos_candidates.iloc[0] if not pos_candidates.empty else None

neg_candidates = summary[summary["neg_years"] > 0].sort_values(
    ["neg_share", "n_years_residual", "residual"], ascending=[False, False, True]
)
most_persistent_neg = neg_candidates.iloc[0] if not neg_candidates.empty else None

f1, f2, f3 = st.columns(3)
with f1:
    if not resid_summary.empty:
        gap_body = (
            f"Across {len(resid_summary)} qualifying schools, admissions gaps ranged from "
            f"{fmt_pp(bottom_row['residual'])} to {fmt_pp(top_row['residual'])}, with a median of "
            f"{fmt_pp(median_gap)}."
        )
    else:
        gap_body = "No schools with a modeled residual are available under the current filters."
    st.markdown(
        f"""<div class="finding-card">
        <div class="finding-num">01</div>
        <div class="finding-title">Size of the gap</div>
        <div class="finding-body">{gap_body}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with f2:
    if most_persistent_pos is not None:
        pos_body = (
            f"{most_persistent_pos['school_label']} exceeded its expected admit rate in "
            f"{int(most_persistent_pos['pos_years'])} of {int(most_persistent_pos['n_years_residual'])} years "
            f"with modeled data, with an applicant-weighted gap of {fmt_pp(most_persistent_pos['residual'])}."
        )
    else:
        pos_body = "No school in the current filters has a positive gap in any modeled year."
    st.markdown(
        f"""<div class="finding-card">
        <div class="finding-num">02</div>
        <div class="finding-title">Most persistent positive gap</div>
        <div class="finding-body">{pos_body}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with f3:
    if most_persistent_neg is not None:
        neg_body = (
            f"{most_persistent_neg['school_label']} fell below its expected admit rate in "
            f"{int(most_persistent_neg['neg_years'])} of {int(most_persistent_neg['n_years_residual'])} years "
            f"with modeled data, with an applicant-weighted gap of {fmt_pp(most_persistent_neg['residual'])}."
        )
    else:
        neg_body = "No school in the current filters has a negative gap in any modeled year."
    st.markdown(
        f"""<div class="finding-card">
        <div class="finding-num">03</div>
        <div class="finding-title">Most persistent negative gap</div>
        <div class="finding-body">{neg_body}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="caveat" style="margin-top:0.8rem;">"Persistent" here means at least 3 years of modeled '
    "data with the same-direction gap in at least 75% of those years — see Section 04 for the full "
    "definition. This is aggregated school-level data: figures describe applicant pools and outcomes for "
    "a school, not the chances of any individual student.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Section 8 — Methodology
# --------------------------------------------------------------------------

st.markdown('<hr class="divider">', unsafe_allow_html=True)
with st.expander("Methodology & Limitations"):
    st.markdown(
        f"""
**Data unit** — One row in the source file represents one high school × year × UC campus.
This dashboard filters to `campus == "{BERKELEY_CAMPUS}"` and `fall_term` between {YEAR_MIN} and {YEAR_MAX}.

**Observed rate** — `admits / applicants`, summed at the school level before dividing
(never the mean of school-year rates), consistent with the fact that schools send very
different numbers of applicants.

**Expected rate** — Taken directly from the dataset's `expected_admit_rate` field. The
model that produced it was not distributed with the data, so this dashboard treats it as a
given baseline rather than re-deriving or explaining its inputs. **`expected_admit_rate` and
`admit_rate_residual` have no coverage for fall 2022** in this file — residual-based views
therefore reflect 2023-2025 even when 2022 is included in the year filter.

**Residual** — `admit_rate_residual` = observed admit rate − expected admit rate, in
percentage points. Positive means a school admitted more applicants than the model expected;
negative means fewer.

**Multi-year residual** — For rankings spanning multiple years, each school's residual is
the applicant-weighted average across its available years:
`sum(residual_year × applicants_year) / sum(applicants_year)`. A school's "years represented"
count only years where both the observed and expected rate are present.

**Consistency** — A "consistent overperformer" has at least 3 years of modeled residual data
and a positive residual in at least 75% of those years; "consistent underperformer" is the
mirror definition. Fewer than 3 years of modeled data is treated as insufficient evidence of
a pattern.

**Suppressed values** — UC redacts small applicant/admit/enrollee cells for privacy. Blank
does not mean zero, and this dashboard never fills missing counts with 0; suppressed rows are
simply excluded from the specific calculation that needs them.

**Aggregated data** — Every figure describes a school and its applicant pool, not individual
students. Nothing here estimates any individual's chance of admission.

**Causality** — Associations shown in Section 06 and the r-values elsewhere describe
correlation, not causal effect. School characteristics are outcomes of many overlapping
factors and should not be read as levers that directly change admission odds.

**Small samples** — Schools with few Berkeley applicants produce volatile admit rates. The
minimum-applicant filter (default {DEFAULT_MIN_APPLICANTS}, currently set to {min_applicants})
exists to keep noisy, low-n schools out of the rankings and charts by default.

**AI narrative (optional)** — The "Generate AI narrative" button sends only the aggregated
statistics already displayed on the page (school counts, rates, residuals, correlation
values) to Google's Gemini API and asks it to phrase them as a short blurb under an
explicit instruction not to invent numbers or claim causation. It is a restatement of
numbers computed elsewhere in this app, not an independent source of analysis — verify
against the charts and cards above before citing it.
        """
    )

st.markdown(
    f'<div class="caveat" style="margin-top:1.5rem;">Beyond the GPA — UC Berkeley admissions residual '
    f"analysis, Bay Area public high schools, {YEAR_MIN}–{YEAR_MAX}. Aggregated school-level data; "
    "no individual-student predictions or causal claims.</div>",
    unsafe_allow_html=True,
)
