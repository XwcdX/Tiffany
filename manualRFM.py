import pandas as pd

file_path = "all_customer_combined_info_RFM_corrected_full.xlsx"
df = pd.read_excel(file_path)

def get_r_score(recency_days):
    if recency_days <= 60:
        return 'High'
    elif recency_days <= 180:
        return 'Medium'
    else:
        return 'Low'

def get_f_score(total_items):
    if total_items > 5:
        return 'High'
    elif total_items >= 3:
        return 'Medium'
    else:
        return 'Low'

def get_m_score(total_spent):
    if total_spent > 5_000_000:
        return 'High'
    elif total_spent >= 1_000_000:
        return 'Medium'
    else:
        return 'Low'

df['R_Score'] = df['Recency_Days'].apply(get_r_score)
df['F_Score'] = df['Total_Items_Purchased'].apply(get_f_score)
df['M_Score'] = df['Total_Spent'].apply(get_m_score)

output_file = "updated_RFM_scoring.xlsx"
df.to_excel(output_file, index=False)

print(f"File berhasil disimpan sebagai: {output_file}")
