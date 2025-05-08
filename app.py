import streamlit as st
import pandas as pd
import json
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")

# --- Configuration ---
DATA_FILE = 'no_redundant_data.xlsx'
GEOJSON_FILE = 'GeoJSON/Indonesia_subdistricts.geojson'

# --- Helper Functions ---

def parse_date(date_str):
    if pd.isna(date_str) or date_str == '':
        return pd.NaT
    try:
        # Split date parts
        parts = date_str.split('-')
        if len(parts) != 3:
            return pd.NaT # Invalid format

        day, month, year_part = parts
        day = int(day)
        month = int(month)

        # Attempt to fix the year part
        if len(year_part) == 2: # YY format
            year = int(year_part)
            # Assumption: years < 30 are 20xx, others are 19xx
            # Adjust this logic if needed based on your data's actual range
            year = 2000 + year if year < 30 else 1900 + year
        elif len(year_part) == 3: # YYY format (assuming 19Y or 20Y?)
            year_prefix = year_part[0]
            year_suffix = int(year_part[1:])
            if year_prefix == '1':
                 year = 1900 + year_suffix
            elif year_prefix == '2' or year_prefix == '0': # e.g., 200 -> 2000
                 year = 2000 + year_suffix
            else: # Unknown 3-digit year format
                 return pd.NaT
            # Basic validation for 3-digit years - refine if necessary
            # For example, '195' likely means 1995, not 1905.
            # Let's assume YYY always implies 19YY if starts with 1, 20YY if starts with 2/0
            # Example specific fix: 195 -> 1995, 197 -> 1997, 199 -> 1999?
            if year < 1950 and year_part.startswith('1'): # Heuristic: 1XX likely means 19XX
                 year += 90 # Guessing 195 -> 1995
        elif len(year_part) == 4: # YYYY format
            year = int(year_part)
        else:
            return pd.NaT # Invalid year format

        return pd.Timestamp(year=year, month=month, day=day)

    except (ValueError, TypeError):
        return pd.NaT

# --- Load and Prepare Data ---
@st.cache_data
def load_data(file_path):
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            st.error(f"Unsupported file format: {file_path}")
            return pd.DataFrame()

        # --- Column Renaming and Cleaning (Adjust based on exact headers in your file) ---
        # Create a mapping from image headers to clean names
        column_mapping = {
            'No.': 'no',
            'Code': 'code',
            'Name': 'name',
            'VIP': 'vip',
            'detail': 'detail', # Keep if needed, otherwise remove later
            'Birthplace': 'birthplace',
            'BirthDate': 'birthdate_str', # Load as string first
            'Address': 'address',
            'JenisKulit': 'jeniskulit',
            'FitzPatrick': 'fitzpatrick',
            'Category': 'category',
            'DiscountType': 'discounttype',
            'Province': 'province',
            'City': 'city',
            'Country': 'country',
            'PostalCode': 'postalcode',
            'Hp': 'hp',
            'Fax': 'fax',
            'Email': 'email',
            'Profession': 'profession',
            'Hobby': 'hobby',
            'Delivery': 'delivery',
            'DineIn': 'dinein',
            'Deposit': 'deposit',
            'ewar': 'reward',
            'Points': 'points',
            'Receiv': 'received',
            'Officer': 'officer',
            'Check': 'check',
            'Status': 'status',
            'District': 'district'
        }

        df.columns = df.columns.str.strip()
        valid_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df[list(valid_columns.keys())].rename(columns=valid_columns)


        # Dates
        df['birthdate'] = df['birthdate_str'].apply(parse_date)
        # Calculate Age (handle NaT dates)
        now = pd.Timestamp.now()
        df['age'] = df['birthdate'].apply(lambda x: (now - x).days // 365 if pd.notna(x) else np.nan)

        # Numeric Columns (convert, coerce errors to NaN)
        numeric_cols = ['vip', 'jeniskulit', 'fitzpatrick']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Categorical/String Columns (fill NaNs, strip whitespace)
        string_cols = ['name', 'birthplace', 'address', 'category', 'discounttype',
                       'province', 'city', 'country', 'postalcode', 'hp', 'fax',
                       'email', 'profession', 'hobby', 'status', 'district']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace({'nan': 'Unknown', '': 'Unknown', 'None':'Unknown'})

        # Yes/No Columns (fill NaNs with 'No' or 'Unknown', standardize)
        yes_no_cols = ['delivery', 'dinein', 'deposit', 'reward', 'points', 'received', 'officer', 'check']
        for col in yes_no_cols:
            if col in df.columns:
                processed_col = df[col].fillna('no').astype(str).str.strip().str.lower()
                df[col] = processed_col.apply(lambda x: 'Yes' if x == 'yes' else 'No')

        # Drop columns if they are not needed (e.g., original row number, intermediate columns)
        # cols_to_drop = ['no', 'birthdate_str', 'detail', 'unnamed_1'] # Example
        # df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        return df

    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        return pd.DataFrame()

# --- Load Data ---
df = load_data(DATA_FILE)

# --- Streamlit App Layout ---
st.title("Customer Data Dashboard")

if df.empty:
    st.warning("Data could not be loaded. Please check the file and try again.")
    st.stop() # Stop execution if data loading failed

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Demographics
# Store user selections in separate variables first
professions_sel_user = st.sidebar.multiselect("Profession", sorted(df['profession'].unique()))
cities_sel_user = st.sidebar.multiselect("City", sorted(df['city'].unique()))
districts_sel_user = st.sidebar.multiselect("District", sorted(df['district'].unique()))

# Age (Keep as is)
min_age = int(df['age'].min()) if df['age'].notna().any() else 0
max_age = int(df['age'].max()) if df['age'].notna().any() else 100
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age), disabled=not df['age'].notna().any())

# Customer Attributes
categories_sel_user = st.sidebar.multiselect("Category", sorted(df['category'].unique()))
statuses_sel_user = st.sidebar.multiselect("Status", sorted(df['status'].unique()))
vip_levels_sel_user = st.sidebar.multiselect("VIP Level", sorted(df['vip'].dropna().unique().astype(int)))

# Skin Attributes
skin_types_sel_user = st.sidebar.multiselect("Jenis Kulit (Skin Type)", sorted(df['jeniskulit'].dropna().unique().astype(int)))
fitzpatrick_sel_user = st.sidebar.multiselect("FitzPatrick Scale", sorted(df['fitzpatrick'].dropna().unique().astype(int)))

# Service Usage (Example Yes/No Filters)
delivery_sel_user = st.sidebar.multiselect("Delivery", ['Yes', 'No'])
dinein_sel_user = st.sidebar.multiselect("DineIn", ['Yes', 'No'])

# --- Apply Filters ---
# Start with a boolean True index
filtered_idx = pd.Series(True, index=df.index)

# Apply filters conditionally based on USER SELECTION (check if the list is not empty)

if professions_sel_user: # Check if the user selected any profession
    filtered_idx &= df['profession'].isin(professions_sel_user)

if cities_sel_user: # Check if the user selected any city
    filtered_idx &= df['city'].isin(cities_sel_user) # <<< Use the user selection variable

if districts_sel_user: # Check if the user selected any district
    filtered_idx &= df['district'].isin(districts_sel_user)

if df['age'].notna().any(): # Only filter age if age data exists
     filtered_idx &= df['age'].between(age_range[0], age_range[1], inclusive='both')

if categories_sel_user: # Check if the user selected any category
    filtered_idx &= df['category'].isin(categories_sel_user)

if statuses_sel_user: # Check if the user selected any status
    filtered_idx &= df['status'].isin(statuses_sel_user)

if vip_levels_sel_user: # Check if the user selected any VIP level
     # Handle NaN in VIP column if filtering is applied
     filtered_idx &= df['vip'].isin(vip_levels_sel_user) # No need for NaN check here if user selected specific levels

if skin_types_sel_user: # Check if the user selected any skin type
     filtered_idx &= df['jeniskulit'].isin(skin_types_sel_user)

if fitzpatrick_sel_user: # Check if the user selected any Fitzpatrick scale
     filtered_idx &= df['fitzpatrick'].isin(fitzpatrick_sel_user)

if delivery_sel_user: # Check if the user selected specific Delivery options
    # If user can select BOTH 'Yes' and 'No', this check is fine.
    # If user selecting nothing means "show all", then this check is also fine.
    # If user selecting nothing means "show none", the logic needs adjustment. Assuming "show all" if nothing selected.
    filtered_idx &= df['delivery'].isin(delivery_sel_user)

if dinein_sel_user: # Check if the user selected specific DineIn options
    filtered_idx &= df['dinein'].isin(dinein_sel_user)


filtered_df = df[filtered_idx].copy()

# --- Data Preview & Download ---
st.subheader(f"Filtered Data Preview ({len(filtered_df)} records)")
# Select subset of columns for better preview if needed
preview_cols = ['name', 'age', 'city', 'district', 'profession', 'category', 'status', 'jeniskulit', 'fitzpatrick', 'vip']
st.dataframe(filtered_df[[col for col in preview_cols if col in filtered_df.columns]], use_container_width=True)

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Filtered Data (CSV)", csv, "filtered_customer_data.csv", "text/csv")

# --- Summary Charts ---
st.subheader("Distributions and Summaries")

if not filtered_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        # Age Distribution
        st.markdown("**Age Distribution**")
        if filtered_df['age'].notna().any():
            fig_age = px.histogram(filtered_df.dropna(subset=['age']), x="age", title="Customer Age Distribution")
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No age data available for the selected filters.")

        # Category Distribution
        st.markdown("**Customer Category**")
        df_cat = filtered_df['category'].value_counts().reset_index(name="Count")
        df_cat.columns = ["Category", "Count"]
        fig_cat = px.pie(df_cat, names="Category", values="Count", title="Customer Category")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Jenis Kulit (Skin Type) Distribution
        st.markdown("**Jenis Kulit (Skin Type)**")
        if filtered_df['jeniskulit'].notna().any():
            df_skin = filtered_df['jeniskulit'].dropna().astype(int).value_counts().reset_index(name="Count")
            df_skin.columns = ["Jenis Kulit", "Count"]
            fig_skin = px.bar(df_skin, x="Jenis Kulit", y="Count", title="Distribution by Jenis Kulit")
            st.plotly_chart(fig_skin, use_container_width=True)
        else:
             st.info("No skin type data available for the selected filters.")

    with col2:
        # Profession Distribution
        st.markdown("**Profession Distribution (Top 10)**")
        df_prof = filtered_df['profession'].value_counts().reset_index(name="Count").head(10)
        df_prof.columns = ["Profession", "Count"]
        fig_prof = px.bar(df_prof, y="Profession", x="Count", orientation='h', title="Top 10 Professions")
        st.plotly_chart(fig_prof, use_container_width=True)

        # Status Distribution
        st.markdown("**Customer Status**")
        df_stat = filtered_df['status'].value_counts().reset_index(name="Count")
        df_stat.columns = ["Status", "Count"]
        fig_stat = px.pie(df_stat, names="Status", values="Count", title="Customer Status")
        st.plotly_chart(fig_stat, use_container_width=True)

        # Fitzpatrick Scale Distribution
        st.markdown("**FitzPatrick Scale**")
        if filtered_df['fitzpatrick'].notna().any():
            df_fitz = filtered_df['fitzpatrick'].dropna().astype(int).value_counts().reset_index(name="Count")
            df_fitz.columns = ["FitzPatrick", "Count"]
            fig_fitz = px.bar(df_fitz, x="FitzPatrick", y="Count", title="Distribution by FitzPatrick Scale")
            st.plotly_chart(fig_fitz, use_container_width=True)
        else:
            st.info("No Fitzpatrick data available for the selected filters.")

else:
    st.info("No data matches the current filters.")


# --- Choropleth Map based on District ---
st.subheader("Customer Distribution by District Map")
if not filtered_df.empty:
    try:
        with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        # Aggregate data by district
        district_counts = filtered_df.groupby('district').size().reset_index(name='Customer Count')

        # Check for districts present in data but potentially missing in GeoJSON or vice-versa
        geojson_districts = {feature['properties'].get('NAME_3') for feature in geojson_data['features']}
        data_districts = set(district_counts['district'])

        missing_in_geojson = data_districts - geojson_districts
        # missing_in_data = geojson_districts - data_districts # Less relevant for plotting

        if missing_in_geojson:
             st.warning(f"Note: The following districts found in the data could not be matched to the GeoJSON 'NAME_3' property and won't appear on the map: {', '.join(list(missing_in_geojson))}. Check for exact name matches (case-sensitive) and ensure 'Not Found' or similar placeholders are handled.")


        # Create the choropleth map
        fig_map = px.choropleth_mapbox(
            district_counts,
            geojson=geojson_data,
            locations='district',          # Column in DataFrame matching featureidkey
            featureidkey='properties.NAME_3', # Path to the district name in GeoJSON properties
            color='Customer Count',        # Column defining the color intensity
            color_continuous_scale="Viridis", # Or any other color scale
            mapbox_style='carto-positron', # Map style
            center={'lat': -2.5, 'lon': 118.0}, # Centered roughly on Indonesia
            zoom=3.5,                         # Adjust zoom level as needed
            opacity=0.6,
            title='Customer Count by District (Kecamatan)',
            hover_name='district',          # Show district name on hover
            hover_data={'Customer Count': True} # Show count on hover
        )
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    except FileNotFoundError:
        st.error(f"GeoJSON file not found at: {GEOJSON_FILE}")
    except KeyError:
         st.error("Error accessing 'properties.NAME_3' in the GeoJSON file. Please verify the GeoJSON structure.")
    except Exception as e:
        st.error(f"Could not generate the choropleth map: {e}")
else:
    st.info("No data available to display the map based on current filters.")

# --- Patient Detail Expander ---
st.subheader("Customer Detail")
if not filtered_df.empty:
    with st.expander("Select a customer to view full details", expanded=False): # Start collapsed
        # --- Customer Selection (Same as before) ---
        customer_list = filtered_df['name'].tolist()
        if 'code' in filtered_df.columns:
             customer_display_list = filtered_df.apply(lambda row: f"{row['name']} ({row.get('code', 'N/A')})", axis=1).tolist()
             display_to_index_map = {display_name: index for index, display_name in zip(filtered_df.index, customer_display_list)}
        else:
             customer_display_list = customer_list
             display_to_index_map = {name: index for index, name in zip(filtered_df.index, customer_list)}

        selected_display_name = st.selectbox(
            "Select Customer",
            options=customer_display_list,
            key="customer_selector_dense" # Use a different key if needed
        )

        # --- Get Selected Customer Data (Same as before) ---
        if selected_display_name and selected_display_name in display_to_index_map:
            selected_index = display_to_index_map[selected_display_name]
            p = filtered_df.loc[selected_index]
            
            code_val = p.get('code', 'N/A')
            name_val = p.get('name', 'N/A')
            birthplace_val = p.get('birthplace', 'N/A')
            birthdate_str = p['birthdate'].strftime('%d-%m-%Y') if pd.notna(p['birthdate']) else 'N/A'
            age_val = f"{int(p['age'])}" if pd.notna(p['age']) else 'N/A'
            address_val = p.get('address', 'N/A')
            city_val = p.get('city', 'N/A')
            district_val = p.get('district', 'N/A')
            profession_val = p.get('profession', 'N/A')
            hp_val = p.get('hp', 'N/A')
            email_val = p.get('email', 'N/A')
            category_val = p.get('category', 'N/A')
            status_val = p.get('status', 'N/A')
            vip_val = f"{int(p['vip'])}" if pd.notna(p['vip']) else 'N/A'
            skin_val = f"{int(p['jeniskulit'])}" if pd.notna(p['jeniskulit']) else 'N/A'
            fitz_val = f"{int(p['fitzpatrick'])}" if pd.notna(p['fitzpatrick']) else 'N/A'
            delivery_val = p.get('delivery', 'N/A')
            dinein_val = p.get('dinein', 'N/A')
            deposit_val = p.get('deposit', 'N/A')
            
            details_style = "font-size: 14px; line-height: 1.6;"

            st.markdown(f"""
            <div style='{details_style}'>
                Code: {code_val} | Name: {name_val} <br>
                Birthplace: {birthplace_val} | Birthdate: {birthdate_str} | Age: {age_val} <br>
                Address: {address_val} <br>
                City: {city_val} | District: {district_val} | Profession: {profession_val} <br>
                Phone (Hp): {hp_val} | Email: {email_val}
                <hr style='margin-top: 5px; margin-bottom: 5px;'>
                Category: {category_val} | Status: {status_val} | VIP Level: {vip_val} <br>
                Jenis Kulit: {skin_val} | FitzPatrick: {fitz_val}
                <hr style='margin-top: 5px; margin-bottom: 5px;'>
                Delivery: {delivery_val} | DineIn: {dinein_val} | Deposit: {deposit_val} <br>
                <!-- Add other Yes/No fields similarly here -->
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Please select a customer.")

else:
    st.info("No customers match the current filters to display details.")