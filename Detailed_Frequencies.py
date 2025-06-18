import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from RFM_by_snapshot import build_rfm    

st.set_page_config(layout="wide")

# ---------------------------------------------------------------
# 0 ▪ File sumber
# ---------------------------------------------------------------
TX_FILE_PATH     = "mapped_treatments.xlsx"
MEMBER_FILE_PATH = "no_redundant_data.xlsx"

# ---------------------------------------------------------------
# 0a ▪ Rentang tanggal transaksi
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def tx_date_range(path: str):
    df = pd.read_excel(path, usecols=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df["Date"].min().date(), df["Date"].max().date()

MIN_DATE, MAX_DATE = tx_date_range(TX_FILE_PATH)

# ---------------------------------------------------------------
# 0b ▪ Wrapper load_rfm (per-snapshot cache)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_rfm(snapshot: date) -> pd.DataFrame:
    return build_rfm(
        tx_file=TX_FILE_PATH,
        member_file=MEMBER_FILE_PATH,
        snapshot=snapshot.isoformat()
    )

# ---------------------------------------------------------------
# 1 ▪ Helper turunan
# ---------------------------------------------------------------
def add_fstage(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan FStage = Total_Items_Purchased; top-coded ke 24."""
    df = df.copy()
    df["FStage"] = df["Total_Items_Purchased"].clip(upper=24)
    return df

score_map = {"Low": 1, "Medium": 2, "High": 3}

# ---------------------------------------------------------------
# 2 ▪ Tabs
# ---------------------------------------------------------------
tab1, tab2 = st.tabs(["Distribution", "Member Flow"])

# ==============================================================
# TAB 1 — DISTRIBUTION
# ==============================================================
with tab1:
    st.subheader("Distribution by FStage × R_Score")

    snap_date = st.date_input(
        "Snapshot date",
        value=MAX_DATE,
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        key="snap"
    )

    status_sel = st.radio(
        "Status", ["All", "Member", "Non-Member"],
        horizontal=True, key="status1"
    )

    # ----- Load snapshot & derive cols -------------------------
    with st.spinner("Loading snapshot…"):
        rfm_df = load_rfm(snap_date)
        if status_sel != "All":
            rfm_df = rfm_df[rfm_df["Status"] == status_sel]
        rfm_df = add_fstage(rfm_df)

    # ----- Summary --------------------------------------------
    summary = (
        rfm_df.groupby(["FStage", "R_Score"])
              .size()
              .reset_index(name="Count")
    )
    summary["Percent"] = summary.groupby("FStage")["Count"] \
                                .transform(lambda s: s / s.sum() * 100)

    fig = px.bar(
        summary,
        x="FStage",
        y="Count",
        color="R_Score",
        category_orders={"FStage": list(range(1, 25)),
                         "R_Score": ["Low", "Medium", "High"]},
        color_discrete_map={"Low": "red", "Medium": "orange", "High": "green"},
        text=summary.apply(lambda r: f"{int(r.Count)} ({r.Percent:.1f}%)", axis=1),
        labels={"FStage": "F Stage", "Count": "Members", "R_Score": "R Score"},
        title=f"R_Score Distribution per FStage — {snap_date:%d %b %Y}"
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(barmode="stack", xaxis=dict(tickmode="linear"),
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Total Members: {len(rfm_df)}**")
    st.dataframe(rfm_df, use_container_width=True)

# ==============================================================
# TAB 2 — MEMBER FLOW
# ==============================================================
with tab2:
    st.subheader("Member Flow (Before → After)")

    col1, col2 = st.columns(2)
    with col1:
        before_date = st.date_input("Before date", value=MIN_DATE,
                                    min_value=MIN_DATE, max_value=MAX_DATE,
                                    key="before")
    with col2:
        after_date = st.date_input("After date",  value=MAX_DATE,
                                   min_value=MIN_DATE, max_value=MAX_DATE,
                                   key="after")

    if after_date < before_date:
        st.error("❌ After date harus ≥ Before date")
        st.stop()

    status_sel2 = st.radio(
        "Status", ["All", "Member", "Non-Member"],
        horizontal=True, key="status2"
    )

    # ----- Load 2 snapshots -----------------------------------
    with st.spinner("Loading both snapshots…"):
        df_before = load_rfm(before_date)
        df_after  = load_rfm(after_date)

        if status_sel2 != "All":
            df_before = df_before[df_before["Status"] == status_sel2]
            df_after  = df_after [df_after ["Status"] == status_sel2]

        df_before = add_fstage(df_before)
        df_after  = add_fstage(df_after)

    # ----- Merge & classify flow ------------------------------
    merged = pd.merge(
        df_after, df_before,
        on="NAMA CUSTOMER",
        how="outer",
        suffixes=("_after", "_before"),
        indicator=True
    )

    new_df = merged[merged["_merge"] == "left_only"]
    stay_df = merged[
        (merged["_merge"] == "both") &
        (merged["FStage_after"] == merged["FStage_before"]) &
        (merged["R_Score_after"] == merged["R_Score_before"])
    ]
    down_df = merged[
        (merged["_merge"] == "both") &
        (merged["FStage_after"] == merged["FStage_before"]) &
        (merged["R_Score_before"].map(score_map) > merged["R_Score_after"].map(score_map))
    ]
    up_df = merged[
        (merged["_merge"] == "both") &
        (
            (merged["FStage_after"] > merged["FStage_before"]) |
            (
                (merged["FStage_after"] == merged["FStage_before"]) &
                (merged["R_Score_after"].map(score_map) > merged["R_Score_before"].map(score_map))
            )
        )
    ]

    # ----- Pie summary ----------------------------------------
    stats = pd.DataFrame({
        "Status": ["New", "Stay", "Downgrade", "Upgrade"],
        "Count":  [len(new_df), len(stay_df), len(down_df), len(up_df)]
    })
    fig2 = px.pie(
        stats, names="Status", values="Count",
        hole=0.4, title=f"Member Flow {before_date} → {after_date}"
    )
    fig2.update_traces(textinfo="label+percent+value")
    st.plotly_chart(fig2, use_container_width=True)

    # ----- Detail tables (expanders) --------------------------
    with st.expander(f"New ({len(new_df)})"):
        st.dataframe(
            new_df[["NAMA CUSTOMER", "FStage_after", "R_Score_after"]],
            use_container_width=True
        )
    with st.expander(f"Stay ({len(stay_df)})"):
        st.dataframe(
            stay_df[["NAMA CUSTOMER", "FStage_after", "R_Score_after"]],
            use_container_width=True
        )
    with st.expander(f"Downgrade ({len(down_df)})"):
        st.dataframe(
            down_df[["NAMA CUSTOMER", "FStage_before", "R_Score_before",
                     "FStage_after", "R_Score_after"]],
            use_container_width=True
        )
    with st.expander(f"Upgrade ({len(up_df)})"):
        st.dataframe(
            up_df[["NAMA CUSTOMER", "FStage_before", "R_Score_before",
                   "FStage_after", "R_Score_after"]],
            use_container_width=True
        )
