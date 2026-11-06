"""Market data quality dashboard.  Run: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mdq import config, dashboard_data as dd, store, viz

st.set_page_config(page_title="Market data quality", layout="wide")

LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
              font=dict(color=viz.INK), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


@st.cache_resource
def get_con():
    if not config.DB_PATH.exists():
        return None
    return store.connect(config.DB_PATH, read_only=True)


con = get_con()
page = st.sidebar.radio("Page", ["Overview", "Issues", "Ticker", "Sources", "Corporate actions",
                                 "Benchmark"])
st.sidebar.caption("Data: Yahoo Finance and Nasdaq.com, fetched locally for personal research; "
                   "not redistributed.")

if con is None and page != "Benchmark":
    st.warning("No database found. Run `python -m mdq fetch` and `python -m mdq pipeline` first.")
    st.stop()

if page == "Overview":
    st.title("Dataset quality")
    ds = dd.dataset_scores(con)
    sc = dd.security_scores(con)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", f"{ds.score.mean():.3f}")
    c2.metric("Worst day", f"{ds.score.min():.2f}", str(ds.loc[ds.score.idxmin(), 'date'].date()),
              delta_color="off")
    c3.metric("Securities", int(sc.shape[0]))
    c4.metric("Scored issues", int(dd.issue_counts(con).n.sum()))
    fig = go.Figure(go.Scatter(x=ds.date, y=ds.score, mode="lines", line=dict(color=viz.SERIES[0], width=2),
                               name="dataset score", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"))
    fig.update_layout(title="Mean quality score across securities, per day", yaxis_title="score",
                      **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Securities by average score")
    st.dataframe(sc, width="stretch", hide_index=True)

elif page == "Issues":
    st.title("Issues by type and severity")
    inc = st.checkbox("Include info (corroborated, unscored)", value=False)
    cnt = dd.issue_counts(con, include_info=inc)
    order = ["info", "warning", "error", "critical"]
    fig = px.bar(cnt, x="n", y="check_name", color="severity", orientation="h",
                 color_discrete_map=viz.SEVERITY_COLOR, category_orders={"severity": order},
                 log_x=True, labels={"n": "issues (log scale)", "check_name": ""})
    fig.update_layout(title="Issue count by check", **{**LAYOUT, "hovermode": "closest"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cnt, width="stretch", hide_index=True)
    check = st.selectbox("Show issues for check", ["(all)"] + sorted(cnt.check_name.unique()))
    df = dd.issues(con, check_name=None if check == "(all)" else check, include_info=inc, limit=2000)
    st.dataframe(df, width="stretch", hide_index=True)

elif page == "Ticker":
    st.title("Ticker drill-down")
    secs = dd.securities(con).security_id.tolist()
    sec = st.selectbox("Security", secs, index=secs.index("CMCSA") if "CMCSA" in secs else 0)
    adj = st.radio("Series", ["split", "raw"], horizontal=True,
                   help="split = as delivered (split-adjusted); raw = reconstructed as-traded")
    p = dd.prices(con, sec, adj)
    fl = dd.flagged_points(con, sec)
    show_info = st.checkbox("Show info flags", value=False)
    if not show_info:
        fl = fl[fl.severity != "info"]
    fig = go.Figure()
    for src, g in p.groupby("source_id"):
        fig.add_trace(go.Scatter(x=g.date, y=g.close, mode="lines", name=src,
                                 line=dict(color=viz.SOURCE_COLOR.get(src, viz.SERIES[2]), width=1.6)))
    if adj == "split" and len(fl):
        for sev, g in fl.dropna(subset=["close"]).groupby("severity"):
            fig.add_trace(go.Scatter(x=g.date, y=g.close, mode="markers", name=f"flag: {sev}",
                                     marker=dict(size=9, color=viz.SEVERITY_COLOR[sev],
                                                 line=dict(color=viz.SURFACE, width=2)),
                                     text=g.check_name, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=f"{sec} close by source with flagged points", **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    sc = dd.security_scores(con, sec)
    fig2 = go.Figure(go.Scatter(x=sc.date, y=sc.score, mode="lines", line=dict(color=viz.SERIES[0])))
    fig2.update_layout(title=f"{sec} daily quality score", yaxis_range=[0, 102], **LAYOUT)
    st.plotly_chart(fig2, width="stretch")
    st.dataframe(dd.issues(con, security_id=sec, include_info=show_info), width="stretch",
                 hide_index=True)

elif page == "Sources":
    st.title("Source vs source")
    secs = dd.securities(con).security_id.tolist()
    sec = st.selectbox("Security", secs, index=secs.index("CMCSA") if "CMCSA" in secs else 0)
    d = dd.source_diff(con, sec)
    fig = go.Figure(go.Scatter(x=d.date, y=d.diff_bps, mode="lines", name="yahoo − nasdaq",
                               line=dict(color=viz.SERIES[0], width=1.5)))
    tol = config.DEFAULT_PARAMS.recon_tolerance_bps
    for y in (tol, -tol):
        fig.add_hline(y=y, line=dict(color=viz.INK_2, width=1, dash="dot"))
    fig.update_layout(title=f"{sec}: close difference, bps (dotted = ±{tol:g} bps tolerance)",
                      yaxis_title="bps", **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Days compared", len(d))
    c2.metric("Median |diff| (bps)", f"{d.diff_bps.abs().median():.3f}")
    # same rule as the check: tolerance + half a cent relative to price
    allowed = tol + 0.005 / ((d.close_yahoo + d.close_nasdaq) / 2) * 1e4
    c3.metric("Days beyond tolerance", int((d.diff_bps.abs() > allowed).sum()))

elif page == "Corporate actions":
    st.title("Corporate actions")
    st.subheader("Split verification")
    st.caption("adjusted_move ≈ 1 means the vendor back-adjusted correctly; raw_move ≈ 1/ratio "
               "is the as-traded jump.")
    st.dataframe(dd.split_verification(con), width="stretch", hide_index=True)
    kind = st.radio("Events", ["split", "dividend"], horizontal=True)
    st.dataframe(dd.corporate_actions(con, action_type=kind), width="stretch",
                 hide_index=True)

elif page == "Benchmark":
    st.title("Injected-error benchmark (held-out test slice)")
    res = dd.benchmark_results()
    if res is None:
        st.warning("Run `python -m mdq benchmark` first.")
        st.stop()
    t = dd.benchmark_table(res)
    metric = st.radio("Metric", ["f1", "precision", "recall"], horizontal=True)
    m = t.pivot(index="detector", columns="error_type", values=metric)
    fig = px.imshow(m, text_auto=".2f", zmin=0, zmax=1, aspect="auto",
                    color_continuous_scale=[viz.SURFACE] + viz.SEQ_BLUE)
    fig.update_layout(title=f"{metric} by detector × error type", **{**LAYOUT, "hovermode": "closest"})
    st.plotly_chart(fig, width="stretch")
    o = t[t.detector.isin(["zscore", "mad", "iforest"]) &
          t.error_type.isin(["spike_x10", "spike_div100", "small_spike"])]
    fig2 = px.bar(o, x="error_type", y=metric, color="detector", barmode="group",
                  color_discrete_map=viz.DETECTOR_COLOR)
    fig2.update_layout(title="Outlier detectors", yaxis_range=[0, 1.05], **{**LAYOUT, "hovermode": "closest"})
    st.plotly_chart(fig2, width="stretch")
    st.subheader("Tuned parameters")
    st.json(res["tuned_params"])
    st.dataframe(t, width="stretch", hide_index=True)
