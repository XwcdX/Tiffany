# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from RFM_by_snapshot import build_rfm    # fungsi yang telah Anda buat

st.set_page_config(layout="wide")

# --------------------------------------------------------------------
# 0.  File dan utilitas dasar
# --------------------------------------------------------------------
TX_FILE_PATH     = "mapped_treatments.xlsx"
MEMBER_FILE_PATH = "no_redundant_data.xlsx"


@st.cache_data(show_spinner=False)
def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


tx_df = load_transactions(TX_FILE_PATH)
min_tx, max_tx = tx_df["Date"].min().date(), tx_df["Date"].max().date()

@st.cache_data(show_spinner=True)
def load_rfm(snapshot: date) -> pd.DataFrame:
    """Cache per-snapshot."""
    return build_rfm(
        tx_file=TX_FILE_PATH,
        member_file=MEMBER_FILE_PATH,
        snapshot=snapshot.isoformat(),
    )

# --------------------------------------------------------------------
# 1. Tab container
# --------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Snapshot Segmentation", "🔀 Segmentation Flow"])

# ====================================================================
# TAB 1 – SNAPSHOT SEGMENTATION
# ====================================================================
with tab1:
    # ───── Filter snapshot ─────
    snapshot_choice = st.date_input(
        "Snapshot date", value=max_tx, min_value=min_tx, max_value=max_tx
    )
    status_choice = st.radio(
        "Status Customer", ["All", "Member", "Non-Member"], index=0
    )

    # ───── Hitung RFM ─────
    rfm_df = load_rfm(snapshot_choice)
    print(rfm_df.columns)
    if status_choice != "All":
        rfm_df = rfm_df[rfm_df["Status"] == status_choice]

    st.markdown(f"### Total customer: **{len(rfm_df)}**")

    # ───── Treemap ─────
    seg_counts = (
        rfm_df["RFM_Classification"]
        .value_counts()
        .reset_index(name="Count")
        .rename(columns={"index": "RFM_Classification"})
    )
    seg_counts["Percent"] = seg_counts["Count"] / seg_counts["Count"].sum() * 100

    fig = px.treemap(
        seg_counts,
        path=["RFM_Classification"],
        values="Count",
        color="Count",
        color_continuous_scale="Blues",
        title=f"Jumlah Customer per Segmen RFM (Snapshot {snapshot_choice})",
        custom_data=["Percent"],
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{value} (%{customdata[0]:.1f}%)",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>",
        textfont_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ───── Dropdown & Table detail ─────
    seg_options = ["All"] + sorted(rfm_df["RFM_Classification"].unique())
    chosen_seg = st.selectbox("Pilih RFM Segment untuk detail:", seg_options)

    rfm_df_detail = (
        rfm_df if chosen_seg == "All"
        else rfm_df[rfm_df["RFM_Classification"] == chosen_seg]
    )
    st.markdown(f"**Total customer di segmen : {len(rfm_df_detail)}**")

    with st.expander("Lihat data RFM detail"):
        st.dataframe(rfm_df_detail, use_container_width=True)

# ====================================================================
# TAB 2 – SEGMENTATION FLOW (Sankey + detail)
# ====================================================================
with tab2:
    import plotly.graph_objects as go
    from streamlit_plotly_events import plotly_events

    st.header("RFM Segmentation Flow")

    # ───── Date range picker (di main page, bukan sidebar) ───────────
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start Date",
            value=min_tx,
            min_value=min_tx,
            max_value=max_tx,
            key="start",
        )
    with col_end:
        end_date = st.date_input(
            "End Date",
            value=max_tx,
            min_value=min_tx,
            max_value=max_tx,
            key="end",
        )

    if start_date > end_date:
        st.error("❌ End Date harus ≥ Start Date")
        st.stop()

    # ───── Hitung RFM sebelum & sesudah (cached) ─────────────────────
    with st.spinner("Menghitung RFM…"):
        rfm_before = load_rfm(start_date)
        rfm_after  = load_rfm(end_date)

    # ───── Bentuk DataFrame flow: OldCategory → NewCategory ──────────
    flow_df = (
        rfm_before[["NAMA CUSTOMER", "RFM_Classification"]]
        .rename(columns={"RFM_Classification": "OldCategory"})
        .merge(
            rfm_after[["NAMA CUSTOMER", "RFM_Classification"]],
            on="NAMA CUSTOMER",
            how="right",
        )
        .rename(columns={"RFM_Classification": "NewCategory"})
    )
    flow_df["OldCategory"].fillna("New Customer", inplace=True)

    sankey_df = (
        flow_df.groupby(["OldCategory", "NewCategory"]).size().reset_index(name="Count")
    )

    # ───── Siapkan node & link Sankey ────────────────────────────────
    old_cats = sorted(sankey_df["OldCategory"].unique())
    new_cats = sorted(sankey_df["NewCategory"].unique())

    node_labels = [f"Old: {c}" for c in old_cats] + [f"New: {c}" for c in new_cats]
    node_colors = ["lightgray"] * len(old_cats) + ["lightblue"] * len(new_cats)

    old_map = {c: i for i, c in enumerate(old_cats)}
    new_map = {c: i + len(old_cats) for i, c in enumerate(new_cats)}

    # link arrays
    link_source, link_target, link_value, link_hover = [], [], [], []

    # total transaksi tiap old-category → untuk persentase link
    old_totals = flow_df.groupby("OldCategory").size().to_dict()

    for _, row in sankey_df.iterrows():
        src = old_map[row["OldCategory"]]
        tgt = new_map[row["NewCategory"]]
        cnt = row["Count"]
        pct = cnt / old_totals.get(row["OldCategory"], 1) * 100

        link_source.append(src)
        link_target.append(tgt)
        link_value.append(cnt)
        link_hover.append(
            f"From {row['OldCategory']} → {row['NewCategory']}<br>"
            f"Count: {cnt} ({pct:.1f}%)"
        )

    # ───── Hover text untuk node (total out / in + breakdown) ────────
    new_totals = flow_df.groupby("NewCategory").size().to_dict()
    node_hovertext = []

    # old nodes
    for cat in old_cats:
        node_hovertext.append(
            f"Old Category: {cat}<br>Total Out: {old_totals.get(cat,0)}"
        )

    # new nodes
    for cat in new_cats:
        new_total = new_totals.get(cat, 0)
        old_val = old_totals.get(cat, 0)
        inc_pct = f"{((new_total - old_val) / old_val * 100):.2f}%" if old_val else "N/A"

        # rincian asal→tujuan per new cat
        flows = sankey_df[sankey_df["NewCategory"] == cat]
        detail_lines = [
            f"from {r['OldCategory']}: {r['Count']} ({r['Count']/new_total*100:.1f}%)"
            for _, r in flows.iterrows()
        ]
        detail_html = "<br>".join(detail_lines)

        node_hovertext.append(
            f"New Category: {cat}<br>Total In: {new_total}<br>"
            f"Increase: {inc_pct}<br>Flows:<br>{detail_html}"
        )

    # ───── Buat diagram Sankey ───────────────────────────────────────
    fig_sankey = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                label=node_labels,
                color=node_colors,
                line=dict(color="black", width=1),
                customdata=node_hovertext,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=link_source,
                target=link_target,
                value=link_value,
                customdata=link_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )
    )
    fig_sankey.update_layout(
        title_text=f"Segmentation Flow ({start_date} → {end_date})",
        # paper_bgcolor="rgba(0,0,0,0)",   # latar luar
        # plot_bgcolor="rgba(0,0,0,0)",    # latar dalam
        font_size=10,
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True
    )

    # ───── Tampilkan & tangkap klik node ─────────────────────────────
    st.subheader("Sankey Diagram")
    clicked = plotly_events(fig_sankey, click_event=True, override_width="100%", hover_event=False, override_height=500)

    if clicked:
        st.subheader("Detail Flow Node (klik)")
        for ev in clicked:
            idx = ev.get("pointIndex")
            if idx is not None:
                st.markdown(f"**{node_labels[idx]}**")
                st.markdown(node_hovertext[idx])

    # ───── Expander tabel flow lengkap ───────────────────────────────
    with st.expander("Lihat tabel flow detail"):
        st.dataframe(
            sankey_df.sort_values("Count", ascending=False),
            use_container_width=True,
        )

    # ───── Dropdown filter untuk tabel detil origin / destination ────
    col_orig, col_dest = st.columns(2)
    with col_orig:
        orig_sel = st.selectbox(
            "Filter asal (OldCategory)", ["All"] + old_cats, key="orig"
        )
    with col_dest:
        dest_sel = st.selectbox(
            "Filter destinasi (NewCategory)", ["All"] + new_cats, key="dest"
        )

    detail_df = sankey_df.copy()
    if orig_sel != "All":
        detail_df = detail_df[detail_df["OldCategory"] == orig_sel]
    if dest_sel != "All":
        detail_df = detail_df[detail_df["NewCategory"] == dest_sel]

    st.subheader("Detail Flow (sesuai filter)")
    st.dataframe(detail_df.reset_index(drop=True), use_container_width=True)
