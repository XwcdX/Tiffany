import re
import pandas as pd

def normalize_dmy(s):
    """
    Take a string like '09/05/1953', '9-5-1953', '29/01/1970'
    and return 'DD-MM-YYYY' without any swapping or time component.
    """
    if pd.isna(s):
        return ''
    s = str(s).strip()
    parts = re.split(r'[-/]', s)
    if len(parts) != 3:
        return ''
    d, m, y = parts
    d = d.zfill(2)
    m = m.zfill(2)
    if len(y) == 2 and y.isdigit():
        yi = int(y)
        y = f"19{y}" if yi > 30 else f"20{y}"
    return f"{d}-{m}-{y}"

df = pd.read_excel('tif2.xlsx').reset_index(drop=True)

if 'BirthDate' in df.columns:
    df['BirthDate'] = df['BirthDate'].apply(normalize_dmy)

records = []
first_col = df.columns[0]

for i in range(0, len(df), 2):
    top = df.iloc[i].copy()
    bot = df.iloc[i+1] if i+1 < len(df) else None

    if bot is not None:
        if pd.notna(bot.get('Birthplace')):
            top['Birthplace'] = bot['Birthplace']

        val = bot[first_col]
        if isinstance(val, str) and val.strip().lower().startswith('*address'):
            extra = val.split(':', 1)[1].strip()
            if pd.isna(top.get('Address')) or top['Address'] == '':
                top['Address'] = extra
            else:
                top['Address'] = f"{top['Address']}, {extra}"

    records.append(top)

clean_df = pd.DataFrame(records)
clean_df.to_excel('cleaned_pelanggan2.xlsx', index=False)
