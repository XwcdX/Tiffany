import pandas as pd

input_file  = '2024SALES.xlsx'
output_file = 'merged_2024_transactions.xlsx'

xls = pd.ExcelFile(input_file)

sheet_names = [s for s in xls.sheet_names if s.strip().endswith('24')]

df_list = []
for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    df['Month'] = sheet 
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

merged_df.to_excel(output_file, index=False)

print(f"Merged {len(sheet_names)} sheets into '{output_file}'")