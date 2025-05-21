import pandas as pd
import re

# Load file
mapped_df = pd.read_excel("mapped_treatments.xlsx")
members_df = pd.read_excel("no_redundant_data.xlsx")

# Fungsi normalisasi nama
def normalize_name(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower()

# Normalisasi nama customer dan nama member
mapped_df['customer_norm'] = mapped_df['NAMA CUSTOMER'].apply(normalize_name)
members_df['member_norm'] = members_df['Name'].apply(normalize_name)

# === 1. MEMBER SPENDING: no_redundant + total spent ===
spending = mapped_df.groupby('customer_norm')['Total_Price'].sum().reset_index()
spending.columns = ['member_norm', 'Total_Spent']
member_spending_df = members_df.merge(spending, on='member_norm', how='left')
member_spending_df['Total_Spent'] = member_spending_df['Total_Spent'].fillna(0)

# === 2. ALL CUSTOMER SPENDING + STATUS ===
customer_spending_df = mapped_df.groupby('customer_norm').agg({
    'NAMA CUSTOMER': 'first',
    'Total_Price': 'sum'
}).reset_index()
customer_spending_df['Status'] = customer_spending_df['customer_norm'].isin(members_df['member_norm'])
customer_spending_df['Status'] = customer_spending_df['Status'].map({True: 'Member', False: 'Non-Member'})

# === 3. TREATMENT SUMMARY FOR EACH MEMBER ===
# Filter hanya transaksi dari customer yang adalah member
member_transactions = mapped_df[mapped_df['customer_norm'].isin(members_df['member_norm'])]

# Bersihkan nama treatment
member_transactions['treatment_clean'] = member_transactions['NAMA TREATMENT'].str.lower().str.strip()

# Hitung jumlah beli per treatment
treatment_summary = (
    member_transactions.groupby(['customer_norm', 'treatment_clean'])
    .size()
    .reset_index(name='Count')
)

# Gabungkan per customer
summary_dict = (
    treatment_summary.groupby('customer_norm')
    .apply(lambda x: ', '.join(f"{row['treatment_clean']} ({row['Count']})" for _, row in x.iterrows()))
    .reset_index(name='Treatment Summary')
)

# Gabungkan kembali dengan nama asli
summary_final = members_df[['Name']].copy()
summary_final['customer_norm'] = summary_final['Name'].apply(normalize_name)
summary_final = summary_final.merge(summary_dict, on='customer_norm', how='left')

# === SIMPAN FILE EXCEL ===
member_spending_df.to_excel("member_spending.xlsx", index=False)
customer_spending_df.to_excel("all_customer_spending_status.xlsx", index=False)
summary_final.to_excel("member_treatment_summary.xlsx", index=False)

print("✅ Selesai. Tiga file berhasil disimpan.")
