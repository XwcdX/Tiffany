import pandas as pd

file1 = 'merged_2024_transactions.xlsx'
file2 = 'merged_2025_transactions.xlsx'

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

merged = pd.concat([df1, df2], ignore_index=True)

merged.to_excel('final_transaction.xlsx', index=False)

print(f"Merged {len(df1)} + {len(df2)} rows into {len(merged)} total rows.")
