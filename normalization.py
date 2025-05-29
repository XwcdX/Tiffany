import pandas as pd
import re

# Load data
df = pd.read_excel('cleaned_final_transaction.xlsx')
pricelist = pd.read_csv('pricelist.csv')

def normalize_name(s):
    if pd.isna(s):
        return ''
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(package|laser|add|rabu|dj|5x|3x)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Normalize treatment and service names
df['treat_norm'] = df['NAMA TREATMENT'].apply(normalize_name)
pricelist['svc_norm'] = pricelist['Service'].apply(normalize_name)

# Deduplicate pricelist to prevent cartesian join explosion
pricelist = pricelist.drop_duplicates(subset='svc_norm', keep='first')

# Merge by normalized service name
df = df.merge(
    pricelist[['svc_norm', 'Price_IDR']],
    left_on='treat_norm', right_on='svc_norm',
    how='left'
)

# Convert Price_IDR to numeric
df['Price_IDR'] = pd.to_numeric(df['Price_IDR'], errors='coerce').fillna(0)

# Multiply by quantity
if 'QTY' in df.columns:
    df['Total_Price'] = df['Price_IDR'] * df['QTY']
else:
    df['Total_Price'] = df['Price_IDR']

# Drop helper columns
df = df.drop(columns=['treat_norm', 'svc_norm'])

# Export
df.to_excel('mapped_treatments.xlsx', index=False)
print(f"✅ Written {len(df)} rows to mapped_treatments.xlsx")
