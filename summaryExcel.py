# import pandas as pd
# import re

# # Load file
# mapped_df = pd.read_excel("mapped_treatments.xlsx")
# members_df = pd.read_excel("no_redundant_data.xlsx")

# # Fungsi normalisasi nama
# def normalize_name(s):
#     if pd.isna(s):
#         return ''
#     return str(s).strip().lower()

# # Normalisasi nama customer dan nama member
# mapped_df['customer_norm'] = mapped_df['NAMA CUSTOMER'].apply(normalize_name)
# members_df['member_norm'] = members_df['Name'].apply(normalize_name)

# # === 1. MEMBER SPENDING: no_redundant + total spent ===
# spending = mapped_df.groupby('customer_norm')['Total_Price'].sum().reset_index()
# spending.columns = ['member_norm', 'Total_Spent']
# member_spending_df = members_df.merge(spending, on='member_norm', how='left')
# member_spending_df['Total_Spent'] = member_spending_df['Total_Spent'].fillna(0)

# # === 2. ALL CUSTOMER SPENDING + STATUS ===
# customer_spending_df = mapped_df.groupby('customer_norm').agg({
#     'NAMA CUSTOMER': 'first',
#     'Total_Price': 'sum'
# }).reset_index()
# customer_spending_df['Status'] = customer_spending_df['customer_norm'].isin(members_df['member_norm'])
# customer_spending_df['Status'] = customer_spending_df['Status'].map({True: 'Member', False: 'Non-Member'})

# # === 3. TREATMENT SUMMARY FOR EACH MEMBER ===
# # Filter hanya transaksi dari customer yang adalah member
# member_transactions = mapped_df[mapped_df['customer_norm'].isin(members_df['member_norm'])]

# # Bersihkan nama treatment
# member_transactions['treatment_clean'] = member_transactions['NAMA TREATMENT'].str.lower().str.strip()

# # Hitung jumlah beli per treatment
# treatment_summary = (
#     member_transactions.groupby(['customer_norm', 'treatment_clean'])
#     .size()
#     .reset_index(name='Count')
# )

# # Gabungkan per customer
# summary_dict = (
#     treatment_summary.groupby('customer_norm')
#     .apply(lambda x: ', '.join(f"{row['treatment_clean']} ({row['Count']})" for _, row in x.iterrows()))
#     .reset_index(name='Treatment Summary')
# )

# # Gabungkan kembali dengan nama asli
# summary_final = members_df[['Name']].copy()
# summary_final['customer_norm'] = summary_final['Name'].apply(normalize_name)
# summary_final = summary_final.merge(summary_dict, on='customer_norm', how='left')

# # === SIMPAN FILE EXCEL ===
# member_spending_df.to_excel("member_spending.xlsx", index=False)
# customer_spending_df.to_excel("all_customer_spending_status.xlsx", index=False)
# summary_final.to_excel("member_treatment_summary.xlsx", index=False)

# print("✅ Selesai. Tiga file berhasil disimpan.")






# import pandas as pd
# import re
# from datetime import datetime, timedelta

# # --- RFM Segmentation Logic (as provided) ---
# rfm_logic_data = {
#     'R': ['High', 'High', 'High', 'High', 'High', 'High', 'High', 'High', 'High',
#           'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium',
#           'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low'],
#     'F': ['High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low',
#           'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low',
#           'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Low'],
#     'M': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low',
#           'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low',
#           'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low'],
#     'Classification': ['Champion', 'Loyal', 'Loyal', 'Loyal', 'Potential loyal', 'Promising',
#                        'Recent Customers', 'Recent Customers', 'Recent Customers', 'At risk',
#                        'Need attention', 'About to sleep', 'Need attention', 'About to sleep',
#                        'About to sleep', 'Need attention', 'About to sleep', 'About to sleep',
#                        'Cant lose them', 'Cant lose them', 'Hibernating', 'Cant lose them',
#                        'Hibernating', 'Hibernating', 'Cant lose them', 'Hibernating', 'Lost']
# }
# rfm_segment_map_df = pd.DataFrame(rfm_logic_data)
# # -----------------------------------------

# # Function to normalize names
# def normalize_name(s):
#     if pd.isna(s):
#         return ''
#     return str(s).strip().lower()

# print("Loading data files...")
# try:
#     mapped_df = pd.read_excel("mapped_treatments.xlsx")
#     members_df = pd.read_excel("no_redundant_data.xlsx")
#     print("Data files loaded successfully.")
# except FileNotFoundError as e:
#     print(f"ERROR: Could not load data files. {e}")
#     print("Please ensure 'mapped_treatments.xlsx' and 'no_redundant_data.xlsx' are in the same directory as the script.")
#     exit()
# except Exception as e:
#     print(f"An unexpected error occurred while loading data files: {e}")
#     exit()

# # --- Data Preprocessing ---
# print("Normalizing names...")
# mapped_df['customer_norm'] = mapped_df['NAMA CUSTOMER'].apply(normalize_name)
# members_df['member_norm'] = members_df['Name'].apply(normalize_name)

# # --- RFM Calculation ---
# DATE_COLUMN_NAME = 'Date'
# PRICE_COLUMN_NAME = 'Total_Price'
# CUSTOMER_NAME_COLUMN = 'NAMA CUSTOMER'

# print(f"Processing date column '{DATE_COLUMN_NAME}' for RFM calculation...")
# if DATE_COLUMN_NAME not in mapped_df.columns:
#     print(f"ERROR: Date column '{DATE_COLUMN_NAME}' not found in mapped_treatments.xlsx.")
#     print(f"Available columns are: {mapped_df.columns.tolist()}")
#     print("Please check the DATE_COLUMN_NAME variable in the script or your Excel file.")
#     exit()

# try:
#     mapped_df[DATE_COLUMN_NAME] = pd.to_datetime(mapped_df[DATE_COLUMN_NAME])
#     print(f"Date column '{DATE_COLUMN_NAME}' converted to datetime successfully.")
# except Exception as e:
#     print(f"ERROR converting '{DATE_COLUMN_NAME}' to datetime: {e}.")
#     print("Please check the date format in your Excel file. It should be unambiguous (e.g., YYYY-MM-DD, MM/DD/YYYY).")
#     exit()

# if PRICE_COLUMN_NAME not in mapped_df.columns:
#     print(f"ERROR: Price column '{PRICE_COLUMN_NAME}' not found in mapped_treatments.xlsx.")
#     exit()


# print("Calculating RFM metrics...")
# snapshot_date = mapped_df[DATE_COLUMN_NAME].max() + timedelta(days=1)

# monetary_df = mapped_df.groupby('customer_norm')[PRICE_COLUMN_NAME].sum().reset_index()
# monetary_df.rename(columns={PRICE_COLUMN_NAME: 'MonetaryValue'}, inplace=True)

# frequency_df = mapped_df.groupby('customer_norm')[DATE_COLUMN_NAME].nunique().reset_index()
# frequency_df.rename(columns={DATE_COLUMN_NAME: 'Frequency'}, inplace=True)

# recency_df = mapped_df.groupby('customer_norm')[DATE_COLUMN_NAME].max().reset_index()
# recency_df['Recency_Days'] = (snapshot_date - recency_df[DATE_COLUMN_NAME]).dt.days
# recency_df.drop(DATE_COLUMN_NAME, axis=1, inplace=True)

# rfm_df = monetary_df.merge(frequency_df, on='customer_norm', how='left')
# rfm_df = rfm_df.merge(recency_df, on='customer_norm', how='left')
# rfm_df.fillna({'MonetaryValue': 0, 'Frequency': 0, 'Recency_Days': rfm_df['Recency_Days'].max()}, inplace=True) # Handle new customers

# # 2. Create R, F, M Scores
# r_labels = ['High', 'Medium', 'Low']
# f_labels = ['Low', 'Medium', 'High']
# m_labels = ['Low', 'Medium', 'High']

# score_bins = 3

# try:
#     rfm_df['R_Score'] = pd.qcut(rfm_df['Recency_Days'], q=score_bins, labels=r_labels, duplicates='drop')
# except ValueError:
#     rfm_df['R_Score'] = pd.cut(rfm_df['Recency_Days'], bins=score_bins, labels=r_labels, include_lowest=True, duplicates='drop')

# try:
#     rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=score_bins, labels=f_labels, duplicates='drop')
# except ValueError:
#     rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=score_bins, labels=f_labels, include_lowest=True, duplicates='drop')

# try:
#     rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'].rank(method='first'), q=score_bins, labels=m_labels, duplicates='drop')
# except ValueError:
#     rfm_df['M_Score'] = pd.cut(rfm_df['MonetaryValue'], bins=score_bins, labels=m_labels, include_lowest=True, duplicates='drop')

# # 3. Merge with RFM Segmentation Logic
# rfm_df = rfm_df.merge(rfm_segment_map_df,
#                       left_on=['R_Score', 'F_Score', 'M_Score'],
#                       right_on=['R', 'F', 'M'],
#                       how='left')
# rfm_df.rename(columns={'Classification': 'RFM_Classification'}, inplace=True)
# rfm_df.drop(['R', 'F', 'M'], axis=1, inplace=True, errors='ignore')
# print("RFM metrics and classification calculated.")

# print("Creating base customer DataFrame...")
# all_customer_base_df = mapped_df.groupby('customer_norm').agg(
#     NAMA_CUSTOMER_ORIGINAL=(CUSTOMER_NAME_COLUMN, 'first')
# ).reset_index()

# all_customer_base_df = all_customer_base_df.merge(
#     rfm_df[['customer_norm', 'MonetaryValue']],
#     on='customer_norm',
#     how='left'
# )
# all_customer_base_df.rename(columns={'MonetaryValue': 'Total_Spent',
#                                      'NAMA_CUSTOMER_ORIGINAL': 'NAMA CUSTOMER'}, inplace=True)
# all_customer_base_df['Total_Spent'] = all_customer_base_df['Total_Spent'].fillna(0)

# all_customer_base_df['Status'] = all_customer_base_df['customer_norm'].isin(members_df['member_norm'])
# all_customer_base_df['Status'] = all_customer_base_df['Status'].map({True: 'Member', False: 'Non-Member'})
# print("Base customer DataFrame created.")

# # --- Treatment Summary for All Customers ---
# print("Calculating treatment summaries for all customers...")
# if 'NAMA TREATMENT' not in mapped_df.columns:
#     print("ERROR: 'NAMA TREATMENT' column not found in mapped_treatments.xlsx. Cannot generate treatment summaries.")
#     all_customers_treatment_summary_str_df = pd.DataFrame(columns=['customer_norm', 'Treatment Summary'])
# else:
#     mapped_df['treatment_clean_all'] = mapped_df['NAMA TREATMENT'].apply(normalize_name)
#     mapped_df_valid_treatments = mapped_df[mapped_df['treatment_clean_all'] != '']
#     if not mapped_df_valid_treatments.empty:
#         all_treatment_counts = (
#             mapped_df_valid_treatments.groupby(['customer_norm', 'treatment_clean_all'])
#             .size()
#             .reset_index(name='Count')
#         )
#         all_customers_treatment_summary_str_df = (
#             all_treatment_counts.groupby('customer_norm')
#             .apply(lambda x: ', '.join(f"{row['treatment_clean_all']} ({row['Count']})" for _, row in x.iterrows()))
#             .reset_index(name='Treatment Summary')
#         )
#     else:
#         all_customers_treatment_summary_str_df = pd.DataFrame(columns=['customer_norm', 'Treatment Summary'])
# print("Treatment summaries calculated.")

# # --- Combine All Information ---
# print("Combining all data for the final report...")
# all_customer_combined_df = pd.merge(
#     all_customer_base_df,
#     all_customers_treatment_summary_str_df,
#     on='customer_norm',
#     how='left'
# )
# # Fill NaN for Treatment Summary if a customer had spending but no treatments
# all_customer_combined_df['Treatment Summary'] = all_customer_combined_df['Treatment Summary'].fillna('')

# # Merge the full RFM data
# all_customer_combined_df = pd.merge(
#     all_customer_combined_df,
#     rfm_df[['customer_norm', 'Recency_Days', 'Frequency', 'R_Score', 'F_Score', 'M_Score', 'RFM_Classification']],
#     on='customer_norm',
#     how='left'
# )

# # --- Final Output DataFrame ---
# final_columns_ordered = [
#     'NAMA CUSTOMER', 'Total_Spent', 'Status',
#     'Recency_Days', 'Frequency',
#     'R_Score', 'F_Score', 'M_Score',
#     'RFM_Classification',
#     'Treatment Summary'
# ]

# final_columns_to_select = [col for col in final_columns_ordered if col in all_customer_combined_df.columns]
# all_customer_combined_df = all_customer_combined_df[final_columns_to_select]

# # --- Save the Combined File ---
# output_filename = "all_customer_combined_info.xlsx"
# print(f"Saving combined data to '{output_filename}'...")
# try:
#     all_customer_combined_df.to_excel(output_filename, index=False)
#     print(f"✅ Selesai. File '{output_filename}' berhasil disimpan.")
#     print("\nColumns in the output file:")
#     print(all_customer_combined_df.columns.tolist())
# except Exception as e:
#     print(f"ERROR: Could not save the Excel file '{output_filename}'. {e}")







import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import traceback

# --- RFM Segmentation Logic (as provided) ---
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
    'Classification': ['Champion', 'Loyal', 'Loyal', 'Loyal', 'Potential loyal', 'Promising',
                       'Recent Customers', 'Recent Customers', 'Recent Customers', 'At risk',
                       'Need attention', 'About to sleep', 'Need attention', 'About to sleep',
                       'About to sleep', 'Need attention', 'About to sleep', 'About to sleep',
                       'Cant lose them', 'Cant lose them', 'Hibernating', 'Cant lose them',
                       'Hibernating', 'Hibernating', 'Cant lose them', 'Hibernating', 'Lost']
}
rfm_segment_map_df = pd.DataFrame(rfm_logic_data)
# -----------------------------------------

# --- Helper Functions ---
def normalize_name(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower()

def robust_rfm_date_parser(date_column_series, df_name_for_debug="transactions_df"):
    """
    Robust date parser adapted for a non-Streamlit script.
    Prints debug info to console.
    """
    print(f"\n--- Debug: Date Parsing for {df_name_for_debug} ---")
    original_dates_copy = date_column_series.copy() # Keep original for comparison
    print(f"Original 'Date' column sample (first 5 from Excel in {df_name_for_debug}):", original_dates_copy.head().tolist())
    print(f"Original 'Date' column dtype from Excel in {df_name_for_debug}:", original_dates_copy.dtype)

    # --- CRITICAL STEP: Convert ALL to string FIRST to neutralize Excel's pre-parsing ---
    print("Converting entire 'Date' column to string before parsing...")
    date_series_as_strings = original_dates_copy.astype(str)
    # print("Sample after astype(str):", date_series_as_strings.head().tolist()) # Can be verbose

    # Attempt 1: Parse string dates with dayfirst=True
    print("Attempting `pd.to_datetime` with `dayfirst=True` on string-converted dates...")
    parsed_dates = pd.to_datetime(date_series_as_strings, dayfirst=True, errors='coerce')
    print(f"After `dayfirst=True`: {parsed_dates.count()} valid dates, {parsed_dates.isnull().sum()} NaT.")

    # Attempt 2: If dayfirst=True left NaTs, try specific D/M/Y formats on those remaining NaTs
    remaining_nat_mask_after_dayfirst = parsed_dates.isnull()
    if remaining_nat_mask_after_dayfirst.any():
        print(f"Found {remaining_nat_mask_after_dayfirst.sum()} NaTs after `dayfirst=True`. Trying specific formats for these...")
        strings_for_specific_formats = date_series_as_strings[remaining_nat_mask_after_dayfirst]
        formats_to_try_for_strings = ['%d/%m/%Y', '%d/%m/%y']
        reparsed_from_specific_formats = pd.Series([pd.NaT] * len(strings_for_specific_formats),
                                                   index=strings_for_specific_formats.index,
                                                   dtype='datetime64[ns]')
        for fmt in formats_to_try_for_strings:
            if reparsed_from_specific_formats.isnull().any():
                current_nat_in_batch_mask = reparsed_from_specific_formats.isnull()
                temp_reparsed = pd.to_datetime(strings_for_specific_formats[current_nat_in_batch_mask],
                                               format=fmt, errors='coerce')
                reparsed_from_specific_formats = reparsed_from_specific_formats.fillna(temp_reparsed)
        parsed_dates = parsed_dates.fillna(reparsed_from_specific_formats)
        print(f"After trying specific D/M/Y formats: {parsed_dates.count()} total valid dates, {parsed_dates.isnull().sum()} remain NaT.")

    if parsed_dates.isnull().any():
        print(f"WARNING: {parsed_dates.isnull().sum()} dates still NaT after all parsing stages.")
        print("Sample of original values that resulted in NaT (first 10):")
        print(original_dates_copy[parsed_dates.isnull()].head(10).tolist())
    else:
        print("All date entries appear to be successfully parsed (no NaTs found after these stages).")
    
    print(f"--- End Date Parsing for {df_name_for_debug} ---")
    return parsed_dates

# --- Loading Data Files ---
print("Loading data files...")
try:
    mapped_df = pd.read_excel("mapped_treatments.xlsx")
    members_df = pd.read_excel("no_redundant_data.xlsx")
    print("Data files loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not load data files. {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading data files: {e}")
    print(traceback.format_exc())
    exit()

# --- Data Preprocessing ---
print("\nNormalizing names...")
if 'NAMA CUSTOMER' not in mapped_df.columns:
    print("ERROR: 'NAMA CUSTOMER' column not found in mapped_treatments.xlsx")
    exit()
mapped_df['customer_norm'] = mapped_df['NAMA CUSTOMER'].apply(normalize_name)

if 'Name' not in members_df.columns:
    print("ERROR: 'Name' column not found in no_redundant_data.xlsx")
    exit()
members_df['member_norm'] = members_df['Name'].apply(normalize_name)

# --- Date Column Processing for RFM ---
DATE_COLUMN_NAME = 'Date'
if DATE_COLUMN_NAME not in mapped_df.columns:
    print(f"ERROR: Date column '{DATE_COLUMN_NAME}' not found in mapped_treatments.xlsx.")
    exit()

print(f"\nProcessing date column '{DATE_COLUMN_NAME}' for RFM calculation using robust parser...")
mapped_df[DATE_COLUMN_NAME] = robust_rfm_date_parser(mapped_df[DATE_COLUMN_NAME], "mapped_df")

if mapped_df[DATE_COLUMN_NAME].isnull().any():
    nat_count_before_drop = mapped_df[DATE_COLUMN_NAME].isnull().sum()
    print(f"WARNING: {nat_count_before_drop} dates are NaT after parsing. These rows will be EXCLUDED from RFM analysis.")
    mapped_df.dropna(subset=[DATE_COLUMN_NAME], inplace=True)
    if mapped_df.empty:
        print("ERROR: All rows were dropped due to invalid dates. RFM analysis cannot proceed.")
        exit()
    print(f"{nat_count_before_drop} rows with NaT dates dropped. Remaining rows: {len(mapped_df)}")

# --- CRITICAL DATE DIAGNOSTICS FOR RFM ---
print("\n--- Date Diagnostics for RFM After Parsing & NaT Drop ---")
min_date_for_rfm = pd.NaT
max_date_for_rfm = pd.NaT

if not mapped_df.empty and DATE_COLUMN_NAME in mapped_df.columns and not mapped_df[DATE_COLUMN_NAME].isnull().all():
    min_date_for_rfm = mapped_df[DATE_COLUMN_NAME].min()
    max_date_for_rfm = mapped_df[DATE_COLUMN_NAME].max()
    print(f"Min date in mapped_df for RFM: {min_date_for_rfm}")
    print(f"Max date in mapped_df for RFM: {max_date_for_rfm}")

    if pd.isna(min_date_for_rfm) or pd.isna(max_date_for_rfm):
        print("ERROR: Min or Max date is NaT after processing. Cannot proceed with RFM.")
        exit()

    year_counts_for_rfm = mapped_df[DATE_COLUMN_NAME].dt.year.value_counts().sort_index()
    print("Value counts of years in mapped_df for RFM:")
    print(year_counts_for_rfm)

    EXPECTED_MAX_DATE_RFM = pd.to_datetime("2025-04-30") 
    RECENT_YEAR_THRESHOLD_RFM = 2023 

    if max_date_for_rfm > EXPECTED_MAX_DATE_RFM:
        print(f"ERROR-ALERT: Max date found ({max_date_for_rfm}) is BEYOND your expected max data date ({EXPECTED_MAX_DATE_RFM}).")
    if min_date_for_rfm.year < RECENT_YEAR_THRESHOLD_RFM:
        print(f"ERROR-ALERT: Minimum date year ({min_date_for_rfm.year}) is before threshold {RECENT_YEAR_THRESHOLD_RFM}.")
else:
    print("ERROR: mapped_df is empty or Date column is all NaT after processing. Cannot proceed with RFM.")
    exit()
print("--- End Date Diagnostics for RFM ---")


# --- Price Column Processing ---
PRICE_COLUMN_NAME = 'Total_Price'
if PRICE_COLUMN_NAME not in mapped_df.columns:
    print(f"WARNING: Price column '{PRICE_COLUMN_NAME}' not found. Using placeholder value 0 for calculations.")
    mapped_df[PRICE_COLUMN_NAME] = 0
mapped_df[PRICE_COLUMN_NAME] = pd.to_numeric(mapped_df[PRICE_COLUMN_NAME], errors='coerce').fillna(0)


# --- RFM Calculation ---
CUSTOMER_NAME_COLUMN = 'NAMA CUSTOMER'
TREATMENT_COLUMN_NAME = 'NAMA TREATMENT'

print("\nCalculating RFM base metrics...")
snapshot_date = max_date_for_rfm + timedelta(days=1)
print(f"Snapshot Date for Recency Calculation: {snapshot_date}")

monetary_df = mapped_df.groupby('customer_norm')[PRICE_COLUMN_NAME].sum().reset_index()
monetary_df.rename(columns={PRICE_COLUMN_NAME: 'MonetaryValue'}, inplace=True)

frequency_df = mapped_df.groupby('customer_norm')[DATE_COLUMN_NAME].nunique().reset_index()
frequency_df.rename(columns={DATE_COLUMN_NAME: 'Frequency'}, inplace=True)

recency_calc_df = mapped_df.groupby('customer_norm')[DATE_COLUMN_NAME].max().reset_index()
recency_calc_df.rename(columns={DATE_COLUMN_NAME: 'LastPurchaseDate'}, inplace=True)
recency_calc_df['Recency_Days'] = (snapshot_date - recency_calc_df['LastPurchaseDate']).dt.days
recency_df = recency_calc_df[['customer_norm', 'Recency_Days']]

if not recency_df.empty:
    max_calculated_recency = recency_df['Recency_Days'].max()
    EXPECTED_MAX_POSSIBLE_RECENCY = (max_date_for_rfm - min_date_for_rfm).days + 1
    print(f"Max calculated recency (before FillNA): {max_calculated_recency}")
    print(f"Expected max possible recency based on data span: {EXPECTED_MAX_POSSIBLE_RECENCY}")
    if max_calculated_recency > EXPECTED_MAX_POSSIBLE_RECENCY + 30: # Allow some buffer
        print(f"WARNING: Max calculated recency {max_calculated_recency} seems too high for data span!")
        print("Top 5 largest recencies:")
        print(recency_df.nlargest(5, 'Recency_Days'))

rfm_df = monetary_df.merge(frequency_df, on='customer_norm', how='left')
rfm_df = rfm_df.merge(recency_df, on='customer_norm', how='left')

rfm_df['Frequency'] = rfm_df['Frequency'].fillna(0).astype(int) # Ensure Frequency is int
rfm_df['MonetaryValue'] = rfm_df['MonetaryValue'].fillna(0)

if 'Recency_Days' in rfm_df.columns:
    if not rfm_df['Recency_Days'].isnull().all():
        max_observed_recency = rfm_df['Recency_Days'].max()
        rfm_df['Recency_Days'].fillna(max_observed_recency, inplace=True)
    else:
        rfm_df['Recency_Days'].fillna(EXPECTED_MAX_POSSIBLE_RECENCY, inplace=True)
else:
    rfm_df['Recency_Days'] = EXPECTED_MAX_POSSIBLE_RECENCY
rfm_df['Recency_Days'] = rfm_df['Recency_Days'].astype(int) # Ensure Recency is int

print(f"Final Max Recency in rfm_df after fillna: {rfm_df['Recency_Days'].max()}")

# --- RFM Score Calculations (R, F, M) ---
# (Using the robust scoring logic from your last complete script version)
r_labels = ['High', 'Medium', 'Low']
f_labels = ['Low', 'Medium', 'High']
m_labels = ['Low', 'Medium', 'High']
score_bins = 3

# R_Score
if not rfm_df.empty and 'Recency_Days' in rfm_df.columns and rfm_df['Recency_Days'].nunique() > 1:
    try:
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency_Days'], q=score_bins, labels=r_labels, duplicates='drop')
    except ValueError:
        print("R_Score: qcut failed. Using pd.cut as fallback.")
        try:
            rfm_df['R_Score'] = pd.cut(rfm_df['Recency_Days'], bins=score_bins, labels=r_labels, include_lowest=True, duplicates='drop')
        except ValueError:
            print("R_Score: pd.cut also failed. Defaulting logic.")
            median_r = rfm_df['Recency_Days'].median()
            rfm_df['R_Score'] = np.select(
                [rfm_df['Recency_Days'] <= rfm_df['Recency_Days'].quantile(1/3), 
                 rfm_df['Recency_Days'] <= rfm_df['Recency_Days'].quantile(2/3)],
                [r_labels[0], r_labels[1]], default=r_labels[2])
    except Exception as e:
        print(f"R_Score: Unexpected error {e}. Defaulting R_Score to Medium.")
        rfm_df['R_Score'] = r_labels[1]
elif not rfm_df.empty:
    rfm_df['R_Score'] = r_labels[1]

# M_Score
if not rfm_df.empty and 'MonetaryValue' in rfm_df.columns and rfm_df['MonetaryValue'].nunique() > 1:
    try:
        rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'].rank(method='first'), q=score_bins, labels=m_labels, duplicates='drop')
    except ValueError:
        print("M_Score: qcut failed. Using pd.cut as fallback.")
        try:
            rfm_df['M_Score'] = pd.cut(rfm_df['MonetaryValue'], bins=score_bins, labels=m_labels, include_lowest=True, duplicates='drop')
        except ValueError:
            print("M_Score: pd.cut also failed. Defaulting logic.")
            rfm_df['M_Score'] = np.select(
                [rfm_df['MonetaryValue'] <= rfm_df['MonetaryValue'].quantile(1/3),
                 rfm_df['MonetaryValue'] <= rfm_df['MonetaryValue'].quantile(2/3)],
                [m_labels[0], m_labels[1]], default=m_labels[2])
    except Exception as e:
        print(f"M_Score: Unexpected error {e}. Defaulting M_Score to Medium.")
        rfm_df['M_Score'] = m_labels[1]
elif not rfm_df.empty:
    rfm_df['M_Score'] = m_labels[1]

# F_Score (Fairness Logic)
print("Calculating F_Score with 'fairness' logic...")
f_score_map = {}
if not rfm_df.empty and 'Frequency' in rfm_df.columns:
    rfm_df['Frequency'] = rfm_df['Frequency'].astype(int) # Ensure int
    unique_f_values = sorted(rfm_df['Frequency'].unique())
    if not unique_f_values:
        rfm_df['F_Score'] = f_labels[1]
    elif len(unique_f_values) == 1:
        f_score_map = {unique_f_values[0]: f_labels[0]}
    elif len(unique_f_values) == 2:
        f_score_map = {unique_f_values[0]: f_labels[0], unique_f_values[1]: f_labels[1]}
    else:
        try:
            binned_unique_f_labeled = pd.qcut(pd.Series(unique_f_values), q=score_bins, labels=f_labels, duplicates='drop')
            f_score_map = dict(zip(unique_f_values, binned_unique_f_labeled))
        except ValueError:
            print("F_Score: qcut on unique values failed. Using manual split for labels.")
            n_unique = len(unique_f_values)
            assigned_labels = []
            for i in range(n_unique):
                label_idx = min(int(i * score_bins / n_unique), score_bins - 1)
                assigned_labels.append(f_labels[label_idx])
            f_score_map = dict(zip(unique_f_values, assigned_labels))
    if f_score_map: rfm_df['F_Score'] = rfm_df['Frequency'].map(f_score_map)
    if 'F_Score' not in rfm_df.columns and not rfm_df.empty : rfm_df['F_Score'] = f_labels[1] # Default if not created
    if 'F_Score' in rfm_df.columns: rfm_df['F_Score'].fillna(f_labels[1], inplace=True) # Fill any NaNs
elif not rfm_df.empty:
    rfm_df['F_Score'] = f_labels[1]


# Merge with RFM Segmentation Logic
if not rfm_df.empty and all(col in rfm_df.columns for col in ['R_Score', 'F_Score', 'M_Score']):
    rfm_df = rfm_df.merge(rfm_segment_map_df, left_on=['R_Score', 'F_Score', 'M_Score'], right_on=['R', 'F', 'M'], how='left')
    rfm_df.rename(columns={'Classification': 'RFM_Classification'}, inplace=True)
    rfm_df.drop(['R', 'F', 'M'], axis=1, inplace=True, errors='ignore')
    if 'RFM_Classification' in rfm_df.columns: rfm_df['RFM_Classification'].fillna('Undefined', inplace=True)
    else: rfm_df['RFM_Classification'] = 'Undefined'
else:
    if not rfm_df.empty: rfm_df['RFM_Classification'] = 'Undefined'
print("RFM metrics and classification calculated.")


# --- Creating Base Customer DataFrame ---
print("\nCreating base customer DataFrame...")
all_customer_base_df = pd.DataFrame() # Initialize
if not mapped_df.empty and 'customer_norm' in mapped_df.columns and CUSTOMER_NAME_COLUMN in mapped_df.columns:
    unique_customers = pd.DataFrame(mapped_df['customer_norm'].unique(), columns=['customer_norm'])
    if not unique_customers.empty:
        customer_names_map = mapped_df.groupby('customer_norm')[CUSTOMER_NAME_COLUMN].first().reset_index(name='NAMA CUSTOMER')
        all_customer_base_df = unique_customers.merge(customer_names_map, on='customer_norm', how='left')
else:
    print("Warning: mapped_df is empty or missing key columns for base_df creation.")

# Merge Total_Spent
if not rfm_df.empty and 'MonetaryValue' in rfm_df.columns and 'customer_norm' in rfm_df.columns:
    all_customer_base_df = all_customer_base_df.merge(rfm_df[['customer_norm', 'MonetaryValue']], on='customer_norm', how='left')
    if 'MonetaryValue' in all_customer_base_df.columns:
        all_customer_base_df.rename(columns={'MonetaryValue': 'Total_Spent'}, inplace=True)
if 'Total_Spent' not in all_customer_base_df.columns: all_customer_base_df['Total_Spent'] = 0.0
all_customer_base_df['Total_Spent'] = all_customer_base_df['Total_Spent'].fillna(0.0)

# Add Status
if not members_df.empty and 'member_norm' in members_df.columns and 'customer_norm' in all_customer_base_df.columns:
    all_customer_base_df['Status'] = all_customer_base_df['customer_norm'].isin(members_df['member_norm'])
    all_customer_base_df['Status'] = all_customer_base_df['Status'].map({True: 'Member', False: 'Non-Member'})
else:
    all_customer_base_df['Status'] = 'Unknown'
print("Base customer DataFrame created.")


# --- Treatment Summary & Total Items ---
all_customers_treatment_summary_str_df = pd.DataFrame(columns=['customer_norm', 'Treatment Summary'])
total_items_df = pd.DataFrame(columns=['customer_norm', 'Total_Items_Purchased'])
if TREATMENT_COLUMN_NAME in mapped_df.columns and not mapped_df.empty:
    mapped_df['treatment_clean_all'] = mapped_df[TREATMENT_COLUMN_NAME].apply(normalize_name)
    mapped_df_valid_treatments = mapped_df[mapped_df['treatment_clean_all'] != '']
    if not mapped_df_valid_treatments.empty:
        all_treatment_counts = mapped_df_valid_treatments.groupby(['customer_norm', 'treatment_clean_all']).size().reset_index(name='Count')
        all_customers_treatment_summary_str_df = all_treatment_counts.groupby('customer_norm').apply(
            lambda x: ', '.join(f"{row['treatment_clean_all']} ({row['Count']})" for _, row in x.iterrows())
        ).reset_index(name='Treatment Summary')
        total_items_df = mapped_df_valid_treatments.groupby('customer_norm').size().reset_index(name='Total_Items_Purchased')
        print("Treatment summaries and total items calculated.")
else:
    print(f"Warning: '{TREATMENT_COLUMN_NAME}' column not found or mapped_df empty. Summaries will be empty.")

# --- Combine All Information ---
print("\nCombining all data for the final report...")
all_customer_combined_df = all_customer_base_df.copy()

if not all_customers_treatment_summary_str_df.empty:
    all_customer_combined_df = pd.merge(all_customer_combined_df, all_customers_treatment_summary_str_df, on='customer_norm', how='left')
if 'Treatment Summary' not in all_customer_combined_df.columns: all_customer_combined_df['Treatment Summary'] = ''
all_customer_combined_df['Treatment Summary'] = all_customer_combined_df['Treatment Summary'].fillna('')

if not total_items_df.empty:
    all_customer_combined_df = pd.merge(all_customer_combined_df, total_items_df, on='customer_norm', how='left')
if 'Total_Items_Purchased' not in all_customer_combined_df.columns: all_customer_combined_df['Total_Items_Purchased'] = 0
all_customer_combined_df['Total_Items_Purchased'] = all_customer_combined_df['Total_Items_Purchased'].fillna(0).astype(int)

# Merge RFM data
rfm_cols_to_merge = ['customer_norm', 'Recency_Days', 'Frequency', 'R_Score', 'F_Score', 'M_Score', 'RFM_Classification']
if not rfm_df.empty and 'customer_norm' in rfm_df.columns:
    rfm_cols_present = [col for col in rfm_cols_to_merge if col in rfm_df.columns]
    if 'customer_norm' in rfm_cols_present: # Must have customer_norm
        all_customer_combined_df = pd.merge(all_customer_combined_df, rfm_df[rfm_cols_present], on='customer_norm', how='left')

# Ensure all RFM columns exist and fill NaNs
for col in ['Recency_Days', 'Frequency']:
    if col not in all_customer_combined_df.columns: all_customer_combined_df[col] = 0
    all_customer_combined_df[col] = all_customer_combined_df[col].fillna(0).astype(int)
for col in ['R_Score', 'F_Score', 'M_Score', 'RFM_Classification']:
    if col not in all_customer_combined_df.columns: all_customer_combined_df[col] = 'Undefined'
    all_customer_combined_df[col] = all_customer_combined_df[col].fillna('Undefined')

# --- Final Output DataFrame ---
final_columns_ordered = [
    'NAMA CUSTOMER', 'Total_Spent', 'Status',
    'Recency_Days', 'Frequency', 'Total_Items_Purchased',
    'R_Score', 'F_Score', 'M_Score',
    'RFM_Classification',
    'Treatment Summary'
]
for col_name in final_columns_ordered: # Ensure all final columns exist
    if col_name not in all_customer_combined_df.columns:
        if col_name in ['Total_Spent', 'Recency_Days', 'Frequency', 'Total_Items_Purchased']: all_customer_combined_df[col_name] = 0
        elif col_name in ['Status', 'R_Score', 'F_Score', 'M_Score', 'RFM_Classification']: all_customer_combined_df[col_name] = 'Undefined'
        elif col_name == 'Treatment Summary': all_customer_combined_df[col_name] = ''
        else: all_customer_combined_df[col_name] = 'Unknown'
all_customer_combined_df = all_customer_combined_df[final_columns_ordered]

recency_cutpoints = rfm_df['Recency_Days'].quantile([1/3, 2/3])
print(recency_cutpoints)

unique_f_values = sorted(rfm_df['Frequency'].unique())
f_labels = ['Low', 'Medium', 'High']
f_score_map = {}

if not unique_f_values:
    # no rows at all
    pass
elif len(unique_f_values) == 1:
    # only one distinct frequency → label it 'Low'
    f_score_map = { unique_f_values[0]: f_labels[0] }
elif len(unique_f_values) == 2:
    # two distinct frequencies → map smallest→Low, largest→Medium
    f_score_map = {
        unique_f_values[0]: f_labels[0],
        unique_f_values[1]: f_labels[1]
    }
else:
    # three or more distinct frequencies → try qcut on the unique values
    try:
        binned = pd.qcut(
            pd.Series(unique_f_values),
            q=3,
            labels=f_labels,
            duplicates='drop'
        )
        # pd.qcut returns a Categorical indexed by 0..(n_unique-1)
        f_score_map = {
            freq_val: binned.iloc[i]
            for i, freq_val in enumerate(unique_f_values)
        }
    except ValueError:
        # fallback: evenly assign labels across sorted unique_f_values
        n_unique = len(unique_f_values)
        assigned = []
        for i in range(n_unique):
            label_idx = min(int(i * 3 / n_unique), 2)
            assigned.append(f_labels[label_idx])
        f_score_map = dict(zip(unique_f_values, assigned))

# Now print out the mapping:
print("\nFrequency → F_Score mapping in this dataset:")
for freq_val, label in f_score_map.items():
    print(f"  • Frequency = {freq_val:>2} → F_Score = '{label}'")

mon_rank_quantiles = rfm_df['MonetaryValue'].rank(method='first').quantile([1/3, 2/3])
print("MonetaryValue rank cut-points (Q1/Q2):")
print(mon_rank_quantiles)

mon_value_quantiles = rfm_df['MonetaryValue'].quantile([1/3, 2/3])
print("MonetaryValue value cut-points (Q1/Q2):")
print(mon_value_quantiles)
# --- Save the Combined File ---
# output_filename = "all_customer_combined_info_RFM_corrected_full.xlsx"
# print(f"\nSaving combined data to '{output_filename}'...")
# try:
#     all_customer_combined_df.to_excel(output_filename, index=False)
#     print(f"✅ Selesai. File '{output_filename}' berhasil disimpan.")
#     print("\nColumns in the output file:")
#     print(all_customer_combined_df.columns.tolist())
#     print("\nSample of the output data (first 5 rows):")
#     print(all_customer_combined_df.head())
# except Exception as e:
#     print(f"ERROR: Could not save the Excel file '{output_filename}'. {e}")
#     print(traceback.format_exc())
# print("\nRFM Script Execution Finished.")