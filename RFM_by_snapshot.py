"""
Hitung metrik & segmen RFM dengan aturan diskret:
- Recency  <  90 hari        → R = High
- Recency ≤ 180 hari        → R = Medium
- lainnya                    → R = Low

- Frequency > 13 transaksi  → F = High
- Frequency ≥ 2 transaksi   → F = Medium
- lainnya                    → F = Low

- Monetary  > 5 juta        → M = High
- Monetary ≥ 1 juta         → M = Medium
- lainnya                    → M = Low
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ────────────────────────────────────────────────────────────────
# 1.  Utilitas kecil
# ────────────────────────────────────────────────────────────────
def norm(text):
    """Bersihkan nama (lowercase & strip, tangani NaN)."""
    return "" if pd.isna(text) else str(text).strip().lower()


def parse_dates(col: pd.Series) -> pd.Series:
    """Parser tanggal robust (default → dayfirst → 2 pola manual)."""
    parsed = pd.to_datetime(col, errors="coerce")
    if parsed.isnull().any():
        parsed = parsed.fillna(pd.to_datetime(col, errors="coerce", dayfirst=True))
    if parsed.isnull().any():
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            mask = parsed.isnull()
            if not mask.any():
                break
            parsed = parsed.fillna(
                pd.to_datetime(col[mask], format=fmt, errors="coerce")
            )
    return parsed


# ────────────────────────────────────────────────────────────────
# 2.  Aturan skoring custom
# ────────────────────────────────────────────────────────────────
def get_r_score(recency_days: int) -> str:
    if recency_days < 90:
        return "High"
    elif recency_days <= 180:
        return "Medium"
    return "Low"


def get_f_score(total_items: int) -> str:
    if total_items > 13:
        return "High"
    elif total_items >= 2:
        return "Medium"
    return "Low"


def get_m_score(total_spent: float) -> str:
    if total_spent > 5_000_000:
        return "High"
    elif total_spent >= 1_000_000:
        return "Medium"
    return "Low"


# ────────────────────────────────────────────────────────────────
# 3.  Fungsi utama
# ────────────────────────────────────────────────────────────────
def build_rfm(
    tx_file: str | Path,
    member_file: str | Path,
    snapshot: str | datetime,
    *,
    date_col="Date",
    customer_col="NAMA CUSTOMER",
    price_col="Total_Price",
    treatment_col="NAMA TREATMENT",
) -> pd.DataFrame:
    """
    • Memuat transaksi + data member
    • Memfilter transaksi ≤ snapshot_date
    • Menghitung Recency, Frequency, Monetary
    • Memberi skor R/F/M berdasar aturan custom
    • (Opsional) merangkum treatment & total item
    • Mengembalikan DataFrame tingkat customer
    """

    snapshot = pd.to_datetime(snapshot).normalize()
    snapshot = snapshot + pd.Timedelta(days=1)      

    # ---------- 3.1 Load & bersihkan ----------
    tx = pd.read_excel(tx_file)
    members = pd.read_excel(member_file)

    tx["cust_norm"] = tx[customer_col].apply(norm)
    members["member_norm"] = members["Name"].apply(norm)

    tx[date_col] = parse_dates(tx[date_col])
    tx = tx[tx[date_col] <= snapshot]  # filter future rows

    tx[price_col] = pd.to_numeric(tx[price_col], errors="coerce").fillna(0)

    # ---------- 3.2 Metrik dasar ----------
    next_day = snapshot + timedelta(days=1)

    rfm = (
        tx.groupby("cust_norm")
        .agg(
            Recency_Days=(date_col, lambda s: (next_day - s.max()).days),
            Frequency=(date_col, "nunique"),
            MonetaryValue=(price_col, "sum"),
            Nama_Pertama=(customer_col, "first"),
        )
        .reset_index()
    )
    
    # ---------- 3.3 Info treatment ----------
    tx["treat_norm"] = tx[treatment_col].apply(norm)
    treat_summary = (
        tx[tx["treat_norm"] != ""]
        .groupby(["cust_norm", "treat_norm"])
        .size()
        .groupby("cust_norm")
        .apply(lambda s: ", ".join(f"{t} ({n})" for t, n in s.items()))
        .rename("Treatment Summary")
    )

    total_items = (
        tx[tx["treat_norm"] != ""]
        .groupby("cust_norm")
        .size()
        .rename("Total_Items_Purchased")
    )

    rfm = rfm.merge(treat_summary, on="cust_norm", how="left").merge(
        total_items, on="cust_norm", how="left"
    )

    rfm["Treatment Summary"].fillna("", inplace=True)
    rfm["Total_Items_Purchased"].fillna(0, inplace=True)

    # ---------- 3.4 Skoring ----------
    rfm["R_Score"] = rfm["Recency_Days"].apply(get_r_score)
    rfm["F_Score"] = rfm["Total_Items_Purchased"].apply(get_f_score)
    rfm["M_Score"] = rfm["MonetaryValue"].apply(get_m_score)

    # ---------- 3.5 Segments ----------
    SEG_MAP = {
        ("High", "High", "High"): "Champion",
        ("High", "High", "Medium"): "Loyal",
        ("High", "High", "Low"): "Loyal",
        ("High", "Medium", "High"): "Loyal",
        ("High", "Medium", "Medium"): "Potential loyal",
        ("High", "Medium", "Low"): "Promising",
        ("High", "Low", "High"): "Recent Customers",
        ("High", "Low", "Medium"): "Recent Customers",
        ("High", "Low", "Low"): "Recent Customers",
        ("Medium", "High", "High"): "At risk",
        ("Medium", "High", "Medium"): "Need attention",
        ("Medium", "High", "Low"): "About to sleep",
        ("Medium", "Medium", "High"): "Need attention",
        ("Medium", "Medium", "Medium"): "About to sleep",
        ("Medium", "Medium", "Low"): "About to sleep",
        ("Medium", "Low", "High"): "Need attention",
        ("Medium", "Low", "Medium"): "About to sleep",
        ("Medium", "Low", "Low"): "About to sleep",
        ("Low", "High", "High"): "Cant lose them",
        ("Low", "High", "Medium"): "Cant lose them",
        ("Low", "High", "Low"): "Hibernating",
        ("Low", "Medium", "High"): "Cant lose them",
        ("Low", "Medium", "Medium"): "Hibernating",
        ("Low", "Medium", "Low"): "Hibernating",
        ("Low", "Low", "High"): "Cant lose them",
        ("Low", "Low", "Medium"): "Hibernating",
        ("Low", "Low", "Low"): "Lost",
    }
    rfm["RFM_Classification"] = rfm.apply(
        lambda r: SEG_MAP.get((r.R_Score, r.F_Score, r.M_Score), "Undefined"), axis=1
    )

    # ---------- 3.6 Status member ----------
    rfm["Status"] = (
        rfm["cust_norm"]
        .isin(members["member_norm"])
        .map({True: "Member", False: "Non-Member"})
    )

    # ---------- 3.7 Susun kolom akhir ----------
    return rfm[
        [
            "Nama_Pertama",
            "MonetaryValue",
            "Status",
            "Recency_Days",
            "Frequency",
            "Total_Items_Purchased",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Classification",
            "Treatment Summary",
        ]
    ].rename(columns={"Nama_Pertama": customer_col})


# ────────────────────────────────────────────────────────────────
# Contoh penggunaan (notebook / script lain)
# ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     df = build_rfm(
#         "mapped_treatments.xlsx",
#         "no_redundant_data.xlsx",
#         snapshot="2025-04-29",
#     )
#     print(df)
# df.to_excel("rfm_report_simple.xlsx", index=False)
# print("Berhasil menyimpan rfm_report_simple.xlsx")
