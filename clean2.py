import pandas as pd
import re

df = pd.read_excel('cleaned_pelanggan2.xlsx')

df['BirthDate'] = pd.to_datetime(df['BirthDate'], errors='coerce') \
                  .dt.strftime('%d-%m-%Y')

def extract_vip(name):
    m = re.search(r'\(VIP\s*(\d+)\)', name, re.IGNORECASE)
    return m.group(1) if m else ''

def extract_detail(name):
    parts = re.findall(r'\((.*?)\)', name)
    details = [p for p in parts if not re.match(r'VIP\s*\d+', p, re.IGNORECASE)]
    return '; '.join(details) if details else ''

df['VIP'] = df['Name'].astype(str).apply(extract_vip)
df['detail'] = df['Name'].astype(str).apply(extract_detail)

df['Name'] = df['Name'].astype(str).str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()

cols = list(df.columns)
if 'Name' in cols:
    idx = cols.index('Name')
    for c in ['VIP', 'detail']:
        if c in cols:
            cols.remove(c)
    cols.insert(idx+1, 'VIP')
    cols.insert(idx+2, 'detail')
df = df[cols]

output_path = 'cleaned_pelanggan_v2_2.xlsx'
df.to_excel(output_path, index=False)