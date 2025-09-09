import json
import os
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ---------- Page config ----------
st.set_page_config(page_title="RiskIntel Dashboard", layout="wide")


# ---------- Constants ----------
DATA_PATH = "data/processed/merged_features.csv"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
FEEDBACK_DIR = os.path.join(REPORTS_DIR, "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.jsonl")


# ---------- Helpers ----------
def _to_datetime_safe(series: pd.Series) -> Optional[pd.Series]:
    try:
        converted = pd.to_datetime(series, errors="coerce")
        if converted.notna().any():
            return converted
        return None
    except Exception:
        return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    candidates = [
        "date",
        "published_at",
        "timestamp",
        "created_at",
        "time",
        "datetime",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_cols:
            col = lower_cols[key]
            if _to_datetime_safe(df[col]) is not None:
                return col
    # Try any column that parses nicely to datetime
    for c in df.columns:
        parsed = _to_datetime_safe(df[c])
        if parsed is not None:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_df(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df_local = pd.read_csv(path)
        return df_local
    except Exception:
        return pd.DataFrame()


def _parse_dt_from_filename(name: str) -> Optional[datetime]:
    # Expect patterns like shap_summary_YYYYMMDD_HHMMSS.png
    try:
        parts = name.split("_")
        if len(parts) >= 3:
            stamp = parts[-1].split(".")[0]  # HHMMSS or HHMMSS.ext
            date_part = parts[-2]
            dt = datetime.strptime(f"{date_part}_{stamp}", "%Y%m%d_%H%M%S")
            return dt
    except Exception:
        return None
    return None


def _list_files_safe(dir_path: str) -> List[str]:
    try:
        return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)])
    except Exception:
        return []


def _glob_candidates() -> Dict[str, List[str]]:
    figs = _list_files_safe(FIGURES_DIR)
    reps = _list_files_safe(REPORTS_DIR)
    return {
        "summary_pngs": [f for f in figs if os.path.basename(f).startswith("shap_summary_") and f.endswith(".png")],
        "bar_pngs": [f for f in figs if os.path.basename(f).startswith("shap_feature_importance_") and f.endswith(".png")],
        "force_global_htmls": [f for f in reps if os.path.basename(f).startswith("shap_force_summary_") and f.endswith(".html")],
        "force_local_htmls": [f for f in reps if os.path.basename(f).startswith("shap_force_local_") and f.endswith(".html")],
        "values_jsons": [f for f in reps if os.path.basename(f).startswith("shap_values_") and f.endswith(".json")],
        "manifests": [f for f in reps if os.path.basename(f).startswith("shap_manifest_") and f.endswith(".json")],
    }


def _pick_latest_timestamped(files: List[str]) -> Optional[str]:
    latest_file = None
    latest_dt = None
    for f in files:
        dt = _parse_dt_from_filename(os.path.basename(f))
        if dt is None:
            # fallback to file mtime
            try:
                dt = datetime.fromtimestamp(os.path.getmtime(f))
            except Exception:
                dt = None
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_file = f
    return latest_file


@st.cache_data(show_spinner=False)
def discover_latest_shap_run() -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Return a dict of selected artifact paths and a list of available run keys.

    Run key heuristic: derive from manifest filename or artifact timestamp; used for selector.
    """
    cands = _glob_candidates()
    manifests = cands["manifests"]
    runs: Dict[str, Dict[str, Optional[str]]] = {}

    # Prefer manifests to collect run artifacts with exact paths
    for mf in manifests:
        try:
            with open(mf, "r") as f:
                m = json.load(f)
            # Expect keys saved by training; guard with .get
            run_key = os.path.splitext(os.path.basename(mf))[0].replace("shap_manifest_", "")
            runs[run_key] = {
                "manifest": mf,
                "summary_png": m.get("summary_png") or _pick_latest_timestamped(cands["summary_pngs"]),
                "bar_png": m.get("feature_importance_png") or _pick_latest_timestamped(cands["bar_pngs"]),
                "force_global_html": m.get("force_global_html") or _pick_latest_timestamped(cands["force_global_htmls"]),
                "force_local_html": m.get("force_local_html") or _pick_latest_timestamped(cands["force_local_htmls"]),
                "values_json": m.get("values_json") or _pick_latest_timestamped(cands["values_jsons"]),
            }
        except Exception:
            # Ignore malformed manifest
            continue

    # If no manifests, construct at least one pseudo-run from newest artifacts
    if not runs:
        run_key = "latest"
        runs[run_key] = {
            "manifest": None,
            "summary_png": _pick_latest_timestamped(cands["summary_pngs"]) or (os.path.join(FIGURES_DIR, "shap_summary.png") if os.path.exists(os.path.join(FIGURES_DIR, "shap_summary.png")) else None),
            "bar_png": _pick_latest_timestamped(cands["bar_pngs"]),
            "force_global_html": _pick_latest_timestamped(cands["force_global_htmls"]),
            "force_local_html": _pick_latest_timestamped(cands["force_local_htmls"]),
            "values_json": _pick_latest_timestamped(cands["values_jsons"]),
        }

    # Pick latest run by comparing artifact times (summary_png preferred)
    def run_time(k: str) -> datetime:
        path = runs[k].get("summary_png") or runs[k].get("bar_png") or runs[k].get("values_json")
        if path and os.path.exists(path):
            return datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.min

    available_keys = sorted(list(runs.keys()), key=run_time, reverse=True)
    selected_key = available_keys[0]
    return runs[selected_key], available_keys


@st.cache_data(show_spinner=False)
def load_shap_values(values_json_path: Optional[str]) -> pd.DataFrame:
    if not values_json_path or not os.path.exists(values_json_path):
        return pd.DataFrame()
    try:
        with open(values_json_path, "r") as f:
            data = json.load(f)
        # Expect dict or list; normalize to DataFrame best-effort
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.json_normalize(data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def bundle_shap_zip(artifacts: Dict[str, Optional[str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in ["summary_png", "bar_png", "force_global_html", "force_local_html", "values_json", "manifest"]:
            p = artifacts.get(key)
            if not p or not os.path.exists(p):
                continue
            arcname = os.path.basename(p)
            try:
                zf.write(p, arcname=arcname)
            except Exception:
                continue
    buf.seek(0)
    return buf.read()


def clear_caches():
    load_df.clear()
    discover_latest_shap_run.clear()
    load_shap_values.clear()


# ---------- Data ----------
df = load_df()


# ---------- Sidebar controls ----------
st.sidebar.header("Filters")

if st.sidebar.button("Refresh data", use_container_width=True):
    clear_caches()
    st.experimental_rerun()

countries: List[str] = []
if not df.empty and "country" in df.columns:
    countries = sorted([c for c in df["country"].dropna().unique().tolist() if isinstance(c, str)])
selected_countries = st.sidebar.multiselect(
    "Country",
    options=countries,
    default=countries,
)

date_col = detect_date_column(df) if not df.empty else None
date_range: Optional[Tuple[datetime, datetime]] = None
if date_col and not df.empty:
    parsed_dt = _to_datetime_safe(df[date_col])
    if parsed_dt is not None and parsed_dt.notna().any():
        min_dt = parsed_dt.min()
        max_dt = parsed_dt.max()
        dr = st.sidebar.date_input(
            "Date range",
            value=(min_dt.date(), max_dt.date()),
        )
        if isinstance(dr, tuple) and len(dr) == 2:
            start_dt = datetime.combine(dr[0], datetime.min.time())
            end_dt = datetime.combine(dr[1], datetime.max.time())
            date_range = (start_dt, end_dt)
    else:
        st.sidebar.info("No valid date column detected.")
else:
    st.sidebar.info("No date column available. Skipping date filter.")

artifacts, available_run_keys = discover_latest_shap_run()
selected_run = st.sidebar.selectbox(
    "Model/SHAP run",
    options=available_run_keys,
    index=0,
)
# If user picks a different run than auto-selected, rebuild artifacts dict to match key
if selected_run != available_run_keys[0]:
    # Recompute runs and select chosen key
    cands = _glob_candidates()
    runs: Dict[str, Dict[str, Optional[str]]] = {}
    manifests = cands["manifests"]
    for mf in manifests:
        try:
            with open(mf, "r") as f:
                m = json.load(f)
            run_key = os.path.splitext(os.path.basename(mf))[0].replace("shap_manifest_", "")
            runs[run_key] = {
                "manifest": mf,
                "summary_png": m.get("summary_png") or _pick_latest_timestamped(cands["summary_pngs"]),
                "bar_png": m.get("feature_importance_png") or _pick_latest_timestamped(cands["bar_pngs"]),
                "force_global_html": m.get("force_global_html") or _pick_latest_timestamped(cands["force_global_htmls"]),
                "force_local_html": m.get("force_local_html") or _pick_latest_timestamped(cands["force_local_htmls"]),
                "values_json": m.get("values_json") or _pick_latest_timestamped(cands["values_jsons"]),
            }
        except Exception:
            continue
    if not runs and selected_run == "latest":
        artifacts = {
            "manifest": None,
            "summary_png": _pick_latest_timestamped(cands["summary_pngs"]) or (os.path.join(FIGURES_DIR, "shap_summary.png") if os.path.exists(os.path.join(FIGURES_DIR, "shap_summary.png")) else None),
            "bar_png": _pick_latest_timestamped(cands["bar_pngs"]),
            "force_global_html": _pick_latest_timestamped(cands["force_global_htmls"]),
            "force_local_html": _pick_latest_timestamped(cands["force_local_htmls"]),
            "values_json": _pick_latest_timestamped(cands["values_jsons"]),
        }
    elif selected_run in runs:
        artifacts = runs[selected_run]


# Apply filters
filtered = df.copy()
if not filtered.empty:
    if selected_countries:
        if "country" in filtered.columns:
            filtered = filtered[filtered["country"].isin(selected_countries)]
    if date_col and date_range is not None:
        parsed_dt = _to_datetime_safe(filtered[date_col])
        if parsed_dt is not None:
            mask = (parsed_dt >= date_range[0]) & (parsed_dt <= date_range[1])
            filtered = filtered[mask]


# ---------- Tabs ----------
overview_tab, explain_tab, dataset_tab, feedback_tab = st.tabs([
    "Overview",
    "Model Explainability",
    "Dataset",
    "Feedback",
])


# ---------- Overview Tab ----------
with overview_tab:
    st.subheader("Risk score over time")
    chart_df = filtered.copy()
    line_rendered = False
    if not chart_df.empty and "risk_score" in chart_df.columns:
        # Ensure a datetime column
        dt_col = date_col or (chart_df.index.name if chart_df.index.name else None)
        if dt_col and dt_col in chart_df.columns:
            x_series = _to_datetime_safe(chart_df[dt_col])
            if x_series is not None:
                chart_df = chart_df.assign(_x=x_series)
        elif date_col is None:
            # synthesize index as datetime if possible
            try:
                chart_df = chart_df.reset_index().rename(columns={"index": "idx"})
                chart_df["_x"] = chart_df["idx"]
            except Exception:
                chart_df["_x"] = range(len(chart_df))
        else:
            chart_df["_x"] = range(len(chart_df))

        try:
            import plotly.express as px  # type: ignore

            if "country" in chart_df.columns:
                fig = px.line(
                    chart_df.sort_values("_x"),
                    x="_x",
                    y="risk_score",
                    color="country",
                    title="Risk score trend",
                )
            else:
                fig = px.line(
                    chart_df.sort_values("_x"), x="_x", y="risk_score", title="Risk score trend"
                )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            line_rendered = True
        except Exception:
            line_rendered = False

    if not line_rendered:
        if not chart_df.empty and "risk_score" in chart_df.columns:
            st.line_chart(chart_df.set_index(chart_df.get("_x", pd.Series(range(len(chart_df)))))["risk_score"], use_container_width=True)
        else:
            st.info("No data available to render risk trend.")

    st.markdown("**Latest rows**")
    if not filtered.empty:
        st.dataframe(filtered.tail(50), use_container_width=True)
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("No rows after filtering.")


# ---------- Model Explainability Tab ----------
with explain_tab:
    st.subheader("SHAP Summary and Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        if artifacts.get("summary_png") and os.path.exists(artifacts["summary_png"]):
            st.image(artifacts["summary_png"], caption="SHAP summary (beeswarm)", use_column_width=True)
        else:
            st.info("SHAP summary image not found.")
    with col2:
        if artifacts.get("bar_png") and os.path.exists(artifacts["bar_png"]):
            st.image(artifacts["bar_png"], caption="Feature importance (bar)", use_column_width=True)
        else:
            st.info("Feature importance image not found.")

    st.divider()
    with st.expander("Global force plot", expanded=False):
        if artifacts.get("force_global_html") and os.path.exists(artifacts["force_global_html"]):
            try:
                with open(artifacts["force_global_html"], "r", encoding="utf-8") as f:
                    html = f.read()
                components.html(html, height=500, scrolling=True)
            except Exception:
                st.warning("Unable to render global force plot.")
        else:
            st.info("Global force plot HTML not found.")

    with st.expander("Local force plot (top-impact instance)", expanded=False):
        if artifacts.get("force_local_html") and os.path.exists(artifacts["force_local_html"]):
            try:
                with open(artifacts["force_local_html"], "r", encoding="utf-8") as f:
                    html = f.read()
                components.html(html, height=500, scrolling=True)
            except Exception:
                st.warning("Unable to render local force plot.")
        else:
            st.info("Local force plot HTML not found.")

    shap_df = load_shap_values(artifacts.get("values_json"))
    if not shap_df.empty:
        st.markdown("SHAP values sample")
        st.dataframe(shap_df.head(10), use_container_width=True)

    zip_bytes = bundle_shap_zip(artifacts)
    if zip_bytes:
        st.download_button(
            label="Download SHAP artifacts (ZIP)",
            data=zip_bytes,
            file_name="shap_artifacts.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ---------- Dataset Tab ----------
with dataset_tab:
    st.subheader("Dataset")
    key_cols = ["date", "country", "title", "sentiment_score", "gdp", "unemployment", "risk_score"]
    present = [c for c in key_cols if c in filtered.columns]
    if not filtered.empty:
        show_df = filtered[present] if present else filtered
        st.dataframe(show_df, use_container_width=True)
        with st.expander("Row inspector and local force plot note", expanded=False):
            st.caption("The saved local force plot corresponds to the top-impact instance from training.")
            if artifacts.get("force_local_html") and os.path.exists(artifacts["force_local_html"]):
                try:
                    with open(artifacts["force_local_html"], "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=500, scrolling=True)
                except Exception:
                    st.warning("Unable to render local force plot preview.")
            else:
                st.info("Local force plot HTML not found for preview.")
    else:
        st.warning("No dataset to display.")


# ---------- Feedback Tab ----------
with feedback_tab:
    st.subheader("Feedback")
    st.caption("Help us improve the dashboard")
    with st.form("feedback_form", clear_on_submit=True):
        rating = st.slider("Rating", min_value=1, max_value=5, value=5)
        features_used = st.multiselect(
            "Features used",
            options=["Overview", "Explainability", "Dataset", "Exports"],
        )
        comments = st.text_area("Comments", placeholder="Your feedback...")
        email = st.text_input("Email (optional)")
        submitted = st.form_submit_button("Submit feedback", use_container_width=True)
    if submitted:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rating": int(rating),
            "features_used": features_used,
            "comments": comments,
            "email": email,
            "filters": {
                "countries": selected_countries,
                "date_range": [date_range[0].isoformat(), date_range[1].isoformat()] if date_range else None,
                "run": selected_run,
            },
        }
        try:
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            try:
                st.toast("Feedback submitted. Thank you!", icon="✅")
            except Exception:
                st.success("Feedback submitted. Thank you!")
        except Exception:
            st.error("Failed to save feedback. Please try again later.")