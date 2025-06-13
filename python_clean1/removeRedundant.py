import pandas as pd
import numpy as np

# --- 1. Load Data from Excel File ---
file_path = 'has_address_valid_district2.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Successfully loaded the Excel file.")
    actual_columns = df.columns.tolist()
    print("Columns in your Excel file:", actual_columns)

    # --- Determine critical column names ---
    # BirthDate Column
    birth_date_col_name = 'BirthDate' # Default
    if birth_date_col_name not in actual_columns:
        possible_bdate_cols = [col for col in actual_columns if 'birth' in col.lower() and 'date' in col.lower()]
        if possible_bdate_cols:
            birth_date_col_name = possible_bdate_cols[0]
            print(f"NOTE: Using '{birth_date_col_name}' as the birth date column based on fuzzy match.")
        else:
            print(f"WARNING: Birth date column '{birth_date_col_name}' not found, and no similar column detected. Birth date logic might not work as expected.")
            birth_date_col_name = None # Set to None if not found

    # Address Column
    address_col_name = 'Address' # Default
    if address_col_name not in actual_columns:
        print(f"WARNING: Address column '{address_col_name}' not found. Address logic might not work as expected.")
        address_col_name = None # Set to None if not found

    # 'No.' Column (for sorting and potential aggregation)
    no_col_name = 'No.' # Default, ensure this matches your Excel
    if no_col_name not in actual_columns:
        common_no_alternatives = ['No', 'ID', 'Index']
        found_alt = False
        for alt in common_no_alternatives:
            if alt in actual_columns:
                no_col_name = alt
                print(f"NOTE: Using '{no_col_name}' as the 'No.' column based on common alternatives.")
                found_alt = True
                break
        if not found_alt and actual_columns and pd.api.types.is_numeric_dtype(df[actual_columns[0]]):
            no_col_name = actual_columns[0]
            print(f"WARNING: Column 'No.' not found. Using first numeric column '{no_col_name}' for 'No.' functionalities.")
        elif not found_alt:
            print(f"WARNING: Column 'No.' not found and no suitable alternative detected. Sorting by 'No.' might fail or be skipped.")
            no_col_name = None

    # Standardize empty strings to NaN and process critical columns
    if birth_date_col_name and birth_date_col_name in df.columns:
        # Replace empty strings with NaN, keep as string for comparison
        df[birth_date_col_name] = df[birth_date_col_name].replace(r'^\s*$', np.nan, regex=True)
        # Ensure it's treated as string type for comparison if not NaN
        df[birth_date_col_name] = df[birth_date_col_name].astype(str).replace('nan', np.nan)


    if address_col_name and address_col_name in df.columns:
        df[address_col_name] = df[address_col_name].replace(r'^\s*$', np.nan, regex=True)
        # Ensure it's treated as string type for comparison if not NaN
        df[address_col_name] = df[address_col_name].astype(str).replace('nan', np.nan)


    print("First 5 rows of your data (after initial NaN conversion for BirthDate/Address):")
    print(df.head())
except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found. Please check the path and filename.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the Excel file: {e}")
    exit()

# Store original column order to restore it at the end
original_input_columns = df.columns.tolist()

# --- 2. Define Helper Function to Identify if Records Represent the Same Person ---
def are_records_same_person(name1, bdate1, addr1, name2, bdate2, addr2):
    if name1 != name2:
        return False

    # Compare birthdates as strings (or NaNs)
    birthdates_conflict = pd.notna(bdate1) and pd.notna(bdate2) and str(bdate1) != str(bdate2)
    
    # Address conflict check
    addr1_str = str(addr1).lower().strip() if pd.notna(addr1) else None
    addr2_str = str(addr2).lower().strip() if pd.notna(addr2) else None
    addresses_conflict = (addr1_str is not None and addr2_str is not None and addr1_str != addr2_str)

    if birthdates_conflict or addresses_conflict:
        return False
    return True

# --- 3. Assign a unique 'person_id' to Records That Should Be Merged ---
sort_key_initial = ['Name']
if no_col_name and no_col_name in df.columns and pd.api.types.is_numeric_dtype(df[no_col_name]):
    sort_key_initial.append(no_col_name)

df_processed = df.sort_values(by=sort_key_initial).reset_index(drop=True)
df_processed['person_id'] = -1
current_pid = 0

print("\nAssigning person_id for grouping...")
for i in range(len(df_processed)):
    if df_processed.loc[i, 'person_id'] == -1:
        df_processed.loc[i, 'person_id'] = current_pid
        
        ref_name = df_processed.loc[i, 'Name']
        # Get birthdate as string (or NaN)
        group_ref_bdate = df_processed.loc[i, birth_date_col_name] if birth_date_col_name and birth_date_col_name in df_processed.columns else np.nan
        group_ref_addr = df_processed.loc[i, address_col_name] if address_col_name and address_col_name in df_processed.columns else np.nan

        for j in range(i + 1, len(df_processed)):
            if df_processed.loc[j, 'person_id'] == -1:
                cand_name = df_processed.loc[j, 'Name']
                cand_bdate = df_processed.loc[j, birth_date_col_name] if birth_date_col_name and birth_date_col_name in df_processed.columns else np.nan
                cand_addr = df_processed.loc[j, address_col_name] if address_col_name and address_col_name in df_processed.columns else np.nan

                if are_records_same_person(ref_name, group_ref_bdate, group_ref_addr,
                                           cand_name, cand_bdate, cand_addr):
                    df_processed.loc[j, 'person_id'] = current_pid
                    
                    if pd.isna(group_ref_bdate) and pd.notna(cand_bdate): group_ref_bdate = cand_bdate
                    if pd.isna(group_ref_addr) and pd.notna(cand_addr): group_ref_addr = cand_addr
        current_pid += 1
print(f"Assigned {current_pid} unique person_ids based on Name, BirthDate, and Address logic.")

# --- 4. Define Column Lists for Aggregation ---
general_cols_user_defined = [
    no_col_name, 'Name', 'Code', 'VIP', 'Detail', birth_date_col_name, address_col_name, 'JobTitle',
    'Nationality', 'CategoryID', 'Category', 'Province', 'City', 'Country',
    'TotalCardIssued', 'Tip', 'Email', 'Fax', 'Profession', 'Hobby',
    'Status', 'Distrik', 'Bulan', 'Active', 'Kecamatan'
]
general_cols_user_defined = [col for col in general_cols_user_defined if col and col in df_processed.columns]

yes_no_cols_user_defined = [
    'Delivery', 'Donasi', 'Reward', 'RewardDeposit',
    'JualAtasNamaSendiri', 'JualBawahHarga', 'OfficerCheck'
]
yes_no_cols_user_defined = [col for col in yes_no_cols_user_defined if col and col in df_processed.columns]

all_cols_in_df = set(df_processed.columns)
all_cols_in_df.discard('person_id')
processed_cols_set = set(general_cols_user_defined) | set(yes_no_cols_user_defined)
remaining_cols_for_general_agg = list(all_cols_in_df - processed_cols_set)
final_general_cols = list(set(general_cols_user_defined + remaining_cols_for_general_agg))
final_yes_no_cols = yes_no_cols_user_defined

print(f"\nColumns for 'first_valid_or_else_first' aggregation: {final_general_cols}")
print(f"Columns for 'merge_yes_no' aggregation: {final_yes_no_cols}")

grouping_column_internal = 'person_id'

# --- 5. Define Aggregation Functions ---
def first_valid_or_else_first(series):
    first_val_encountered = None
    for x in series:
        if pd.notna(x): # This correctly handles np.nan
            if first_val_encountered is None: first_val_encountered = x
            # For strings, check if it's non-empty after stripping.
            # For non-strings (like numbers, or already cleaned date strings that aren't just whitespace), take them.
            if isinstance(x, str) and x.strip() != "": return x
            elif not isinstance(x, str): return x # Includes numbers, booleans, etc.
    # If loop finishes, return the first non-null value encountered (could be an empty string if that's all)
    if first_val_encountered is not None: return first_val_encountered
    return np.nan

def merge_yes_no(series):
    if series.isnull().all(): return np.nan
    if series.astype(str).str.strip().str.upper().eq('YES').any(): return 'Yes'
    return 'No'

# --- 6. Create Aggregation Dictionary ---
agg_dict = {}
for col in final_general_cols:
    if col != grouping_column_internal: agg_dict[col] = first_valid_or_else_first
for col in final_yes_no_cols:
    agg_dict[col] = merge_yes_no

if not agg_dict:
    print("ERROR: The aggregation dictionary is empty.")
    exit()

# --- 7. Perform Grouping and Aggregation ---
print(f"\nAttempting to group by '{grouping_column_internal}' and aggregate...")
if grouping_column_internal not in df_processed.columns:
    print(f"ERROR: Internal grouping column '{grouping_column_internal}' not found.")
    exit()

try:
    merged_df = df_processed.groupby(grouping_column_internal, as_index=False).agg(agg_dict)
    print("Grouping and aggregation successful.")
except Exception as e:
    print(f"An unexpected error occurred during grouping/aggregation: {e}")
    exit()

# --- 8. Reorder Columns to Match Original Input Order ---
final_ordered_columns = [col for col in original_input_columns if col in merged_df.columns]
merged_df = merged_df[final_ordered_columns]

# --- 9. Sort by 'No.' column ---
if no_col_name and no_col_name in merged_df.columns:
    print(f"\nSorting final DataFrame by '{no_col_name}'...")
    merged_df[no_col_name] = pd.to_numeric(merged_df[no_col_name], errors='coerce')
    merged_df.sort_values(by=no_col_name, inplace=True)
else:
    print(f"\nWARNING: Column '{no_col_name}' not found in merged_df or not defined. Skipping final sort by 'No.'.")


# --- 10. Output ---
print("\nOriginal DataFrame sample (first 5 rows):")
print(df.head())
print("\nMerged DataFrame (first 5 rows after potential sorting):")
print(merged_df.head())
print(f"\nNumber of rows in original DataFrame: {len(df)}")
print(f"Number of rows in merged DataFrame: {len(merged_df)}")

output_file_excel = 'cleaned_data_merged_sorted_string_dates.xlsx'
try:
    merged_df.to_excel(output_file_excel, index=False)
    print(f"\nCleaned, merged, and sorted data saved to Excel: {output_file_excel}")
except Exception as e:
    print(f"\nError saving the output file: {e}")