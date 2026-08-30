# Beyond the GPA

**Which Bay Area public high schools consistently outperform their expected UC Berkeley admission rate?**

An interactive Streamlit dashboard analyzing observed vs. modeled UC Berkeley freshman
admission outcomes for California public high schools, 2022–2025.

## Research Question

From Fall 2022–2025, among Bay Area public high schools applying to UC Berkeley, how large
is the gap between observed and expected freshman admit rates, and which schools
consistently have the largest positive or negative gaps?

## Why This Question Matters

Admission rates alone don't say much — a school full of high-GPA applicants "should"
have a higher admit rate than a school with a more varied applicant pool. This dashboard
instead looks at the **residual**: how a school's actual Berkeley admit rate compares to
a modeled *expected* rate already provided in the dataset. Schools that consistently beat
that expectation are doing something worth understanding; schools that consistently fall
short may be worth a closer look too.

## Dataset

`dashboard_data.csv` — one row per high school × year × UC campus, drawn from the UC
Information Center and California Department of Education, for Bay Area public high
schools. The dashboard filters to `campus == "Berkeley"` and `fall_term` 2022–2025.

Key fields used: `applicants`, `admits`, `admit_rate`, `expected_admit_rate`,
`admit_rate_residual`, `applicant_gpa`, `ag_completion_rate`, `frpm_pct`,
`caaspp_mathematics_pct_met`, `caaspp_ela_pct_met`, and `cohort_students`.

**Known data quirks handled explicitly by the dashboard:**

- `expected_admit_rate` / `admit_rate_residual` have **no coverage for fall 2022** in the
  source file — residual-based views reflect 2023–2025 even when 2022 is included in the
  year filter (called out in the UI).
- Blank counts are UC-redacted (suppressed), not zero — never filled with `0`.
- Aggregate rates are computed as `sum(admits) / sum(applicants)`, never the mean of
  per-school rates.
- Same-named schools in different cities (e.g. two "Fremont High School"s) are distinct
  schools — grouped by `high_school` + `city`, not by name alone.

## Methodology

- **Observed rate** = `admits / applicants`, summed before dividing.
- **Residual** = observed admit rate − `expected_admit_rate`, in percentage points.
- **Multi-year residual** for a school = applicant-weighted average across its available
  years: `sum(residual_year × applicants_year) / sum(applicants_year)`.
- **Consistent overperformer** = at least 3 years of modeled residual data with a
  positive residual in ≥75% of those years (mirror definition for underperformers).
- **Minimum applicant filter** (default 20 total Berkeley applicants across the selected
  period) keeps small, noisy applicant pools out of rankings and charts by default.
- Every calculation is derived live from the CSV — nothing is hard-coded.

## Dashboard Features

The dashboard follows the shape of the research question, section by section:

1. **Overview** — global filters (year range, county, minimum applicants, school search)
   and 4 KPI cards: schools analyzed, Berkeley applicants, median admissions gap, and count
   of consistent overperformers. Includes an optional AI narrative (Gemini): on request, it
   restates the aggregated statistics already on the page as a short plain-language blurb,
   under an explicit no-invention / no-causal-claims instruction — see
   [AI Narrative Setup](#ai-narrative-setup-optional).
2. **Expected vs. Actual** — the hero scatter: every qualifying school plotted against the
   `y = x` line, sized by applicant volume.
3. **Gap Leaderboard** — top schools by largest positive or largest negative
   applicant-weighted gap.
4. **Consistency** — a 2022–2025 heatmap distinguishing a persistent pattern from one
   unusual year, with the "consistent" definition stated alongside it.
5. **School Explorer** — pick a school, see its actual/expected/gap/applicants/years-of-data
   at a glance, plus its own year-by-year trend line.
6. **What's Associated?** — residual vs. a selected school characteristic (GPA, a-g
   completion, FRPM, proficiency, graduation rate, size), with Pearson's r and an explicit
   association-not-causation label.
7. **Key Answer** — three findings computed live from the filtered data: the size of the
   gap, the most persistent positive gap, and the most persistent negative gap.

A full methodology/limitations expander sits at the bottom.

## Limitations

- This is aggregated **school-level** data — nothing here predicts an individual
  student's chance of admission.
- Relationships shown are **associations, not causal effects**.
- `expected_admit_rate`'s underlying model was not distributed with the data; it is used
  here as a given baseline, not re-derived.
- Small applicant pools produce volatile rates — use the minimum-applicant filter.
- UC's reported GPA is capped-weighted with a ceiling around 4.40.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads `dashboard_data.csv` from the repository root — no absolute paths, no
external services required. The dashboard is fully functional without any API key; the
Gemini narrative button is the one optional feature that needs one (see below).

## AI Narrative Setup (optional)

The "Generate AI narrative" button (top of the dashboard) calls the Gemini API. Everything
else in the dashboard works without it.

**Local:** create `.streamlit/secrets.toml` (already git-ignored) with:

```toml
GEMINI_API_KEY = "your-key-here"
```

**Streamlit Community Cloud:** in the app's *Settings → Secrets*, add the same
`GEMINI_API_KEY = "your-key-here"` line. Never commit an API key to the repository.

If no key is configured, the button shows a clear error instead of crashing the app.
