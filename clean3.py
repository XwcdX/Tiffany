import pandas as pd

df = pd.read_excel('cleaned_pelanggan_v2_2.xlsx')

df['Address'] = df['Address'].fillna('').astype(str).str.strip()

with_address = df[df['Address'] != '']
no_address   = df[df['Address'] == '']

with_address.to_excel('has_address2.xlsx', index=False)
no_address.to_excel('no_address2.xlsx', index=False)
