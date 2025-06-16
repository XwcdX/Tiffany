import pandas as pd

try:
    file_path = "all_customer_combined_info_RFM_corrected_full.xlsx"
    df = pd.read_excel(file_path)
    print(f"Successfully loaded '{file_path}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the Excel file is in the same directory as the script.")
    exit()

if 'RFM_Classification' in df.columns:
    print("Found an old 'RFM_Classification' column. It will be removed and correctly regenerated.")
    df.drop(columns=['RFM_Classification'], inplace=True)

def get_r_score(recency_days):
    if recency_days < 90:
        return 'High'
    elif recency_days <= 180:
        return 'Medium'
    else:
        return 'Low'

def get_f_score(total_items):
    if total_items > 13:
        return 'High'
    elif total_items >= 2:
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
print("Generated new R, F, and M scores.")

rfm_logic_data = {
    'R': ['High', 'High', 'High', 'High', 'High', 'High', 'High', 'High', 'High',
          'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium',
          'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low'],
    'F': ['High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low',
          'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low',
          'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low'],
    'M': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low',
          'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low',
          'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low'],
    'RFM_Classification': ['Champion', 'Loyal', 'Loyal', 'Loyal', 'Potential loyal', 'Promising',
                           'Recent Customers', 'Recent Customers', 'Recent Customers', 'At risk',
                           'Need attention', 'About to sleep', 'Need attention', 'About to sleep',
                           'About to sleep', 'Need attention', 'About to sleep', 'About to sleep',
                           'Cant lose them', 'Cant lose them', 'Hibernating', 'Cant lose them',
                           'Hibernating', 'Hibernating', 'Cant lose them', 'Hibernating', 'Lost']
}
rfm_segment_map_df = pd.DataFrame(rfm_logic_data)

df = pd.merge(
    df,
    rfm_segment_map_df,
    left_on=['R_Score', 'F_Score', 'M_Score'],
    right_on=['R', 'F', 'M'],
    how='left'
)

df.drop(columns=['R', 'F', 'M'], inplace=True)
print("Applied new, correct classifications.")

output_file = "updated_RFM_with_classification.xlsx"
df.to_excel(output_file, index=False)