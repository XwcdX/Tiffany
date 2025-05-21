import pandas as pd
import re


df = pd.read_excel('final_transaction.xlsx')

other_cols = df.columns.drop('Month')

mask_only_month = df['Month'].notna() & df[other_cols].isna().all(axis=1)
df = df.loc[~mask_only_month].copy()
df['Month'] = df['Month'].astype(str).str.lower().str.strip()

df['Month'] = df['Month'].str.replace(
    r'^(.*?)(?<!\s)(24|25)$',
    r'\1 \2',
    regex=True
)

Month_map = {
    'jan':   'january',
    'feb':   'february',
    'mar':   'march',
    'apr':   'april',
    'jun':   'june',
    'jul':   'july',
    'agst':  'august',
    'sept':  'september',
    'okt':   'october',
    'nov':   'november',
    'des':   'desember'
}

def expand_Month(val):
    """
    Extracts an alphabetic Month part and optional 2-digit year,
    maps the Month part, and recombines.
    """
    m = re.match(r'^([a-z]+)\s*(\d{2})?$', val)
    if not m:
        return val
    abbr, yr = m.groups()
    full = Month_map.get(abbr, abbr)
    return f"{full} {yr}" if yr else full

df['Month'] = df['Month'].apply(expand_Month)

for col in df.select_dtypes(include=['datetime64[ns]']):
    df[col] = df[col].dt.date

df.to_excel('cleaned_final_transaction.xlsx', index=False)
print(f"Result: {len(df)} rows saved to cleaned_merged.xlsx")
