import streamlit as st
import pandas as pd
import random
import datetime
import json
import plotly.express as px

# ─── Dummy data generation ────────────────────────────────────────────────────────
@st.cache_data
def generate_dummy_data(n=200):
    first_names = ["Andi","Budi","Citra","Dedi","Eka","Fajar","Gita","Hadi","Intan","Joko",
                   "Karina","Laksmi","Mira","Niko","Putri","Randy","Sari","Tono","Umi","Vina"]
    last_names = ["Santoso","Widodo","Pratama","Susilo","Wijaya","Hartono","Kusuma","Rahmad",
                  "Saputra","Anggraini","Sukma","Setiawan","Prasetyo","Nugroho","Yulianto",
                  "Amin","Sihombing","Rachman","Putra","Sari"]
    cities = ["Jakarta","Bandung","Surabaya","Yogyakarta","Medan","Makassar","Semarang","Palembang"]
    jobs = ["Doctor","Nurse","Teacher","Engineer","Student","Police","Artist","Lawyer","Farmer","Entrepreneur"]
    statuses = ["single","marriage","divorce"]
    skin_types = ["kering","normal","berminyak","kombinasi","sensitif"]
    fitz = ["I","II","III","IV","V","VI"]
    diseases = [
        "pori-pori besar","hiperpigmentasi","hipopigmentasi","komedo","milia","acne","kutil",
        "scar","keloid","bayangan gelap sekitar mata","kantung mata","tahi lalat/mole","keratosis"
    ]
    allergies = ["None","Penicillin","Aspirin","Latex","Peanuts","Shellfish","Pollen"]

    data = []
    for _ in range(n):
        name     = f"{random.choice(first_names)} {random.choice(last_names)}"
        city     = random.choice(cities)
        bd       = datetime.date.today() - datetime.timedelta(days=random.randint(18*365,65*365))
        bd_str   = bd.strftime("%d %m %Y")
        bd_place = f"{city}, {bd_str}"
        addr     = f"{random.randint(1,200)} Jl. {random.choice(['Merdeka','Sudirman','Thamrin','Gatot Subroto'])}, {city}"
        data.append({
            "name": name,
            "birth_date_and_place": bd_place,
            "address": addr,
            "job": random.choice(jobs),
            "phone": f"08{random.randint(100000000,999999999)}",
            "drug allergy": random.choice(allergies),
            "status": random.choice(statuses),
            "skin type": random.choice(skin_types),
            "skin type FITZ PATRICK": random.choice(fitz),
            "skin disease(s)": ", ".join(random.sample(diseases, random.randint(0,3))) or "None"
        })
    return pd.DataFrame(data)

# ─── Load / generate data ────────────────────────────────────────────────────────
df = generate_dummy_data(200)

# ─── FIXED split on ', ' to avoid leading space ─────────────────────────────────
df[['birth_place','birth_date_str']] = (
    df['birth_date_and_place']
      .str.split(', ', n=1, expand=True)
)

df['birth_date'] = pd.to_datetime(df['birth_date_str'], format='%d %m %Y')
df['age'] = (pd.Timestamp.now() - df['birth_date']).dt.days // 365

st.title("Patient Data Dashboard")

# ─── Sidebar: Basic Filters ───────────────────────────────────────────────────────
st.sidebar.header("Basic Filters")
jobs_sel  = st.sidebar.multiselect("Job", df['job'].unique())
areas_sel = st.sidebar.multiselect("Area (City)", df['birth_place'].unique())
min_age, max_age = int(df['age'].min()), int(df['age'].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# ─── Sidebar: Clinical Filters ──────────────────────────────────────────────────
st.sidebar.header("Clinical Filters")
allergies_sel = st.sidebar.multiselect("Drug Allergy", df["drug allergy"].unique())
status_sel    = st.sidebar.multiselect("Marital Status", df["status"].unique())
stype_sel     = st.sidebar.multiselect("Skin Type", df["skin type"].unique())
fitz_sel      = st.sidebar.multiselect("Fitzpatrick Scale", df["skin type FITZ PATRICK"].unique())
all_ds        = sorted({d.strip() for cell in df["skin disease(s)"].str.split(",") for d in cell if d.strip()})
diseases_sel  = st.sidebar.multiselect("Skin Disease(s)", ["None"]+all_ds)

if not jobs_sel:  jobs_sel  = df['job'].unique()
if not areas_sel: areas_sel = df['birth_place'].unique()
if not allergies_sel: allergies_sel = df["drug allergy"].unique()
if not status_sel:    status_sel    = df["status"].unique()
if not stype_sel:     stype_sel     = df["skin type"].unique()
if not fitz_sel:      fitz_sel      = df["skin type FITZ PATRICK"].unique()
if not diseases_sel:  diseases_sel  = ["None"] + all_ds

# ─── Apply filters ───────────────────────────────────────────────────────────────
filtered = df[
    df['job'].isin(jobs_sel) &
    df['birth_place'].isin(areas_sel) &
    df['age'].between(*age_range) &
    df['drug allergy'].isin(allergies_sel) &
    df['status'].isin(status_sel) &
    df['skin type'].isin(stype_sel) &
    df['skin type FITZ PATRICK'].isin(fitz_sel)
].loc[lambda d: d["skin disease(s)"].apply(lambda cell: any(sd in cell for sd in diseases_sel))]

# ─── Data preview & download ────────────────────────────────────────────────────
st.subheader("Filtered Data Preview")
st.dataframe(filtered, use_container_width=True)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "filtered_patients.csv", "text/csv")

# ─── Summary Charts ───────────────────────────────────────────────────────────────
st.subheader("Distributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Skin Type Distribution**")
    df1 = (
        filtered["skin type"]
          .value_counts()
          .reset_index(name="Count")
    )
    df1.columns = ["Skin Type","Count"]
    fig1 = px.bar(df1, x="Skin Type", y="Count", title="Skin Type Distribution")
    st.plotly_chart(fig1)

    st.markdown("**Fitzpatrick Scale**")
    df2 = (
        filtered["skin type FITZ PATRICK"]
          .value_counts()
          .reset_index(name="Count")
    )
    df2.columns = ["Fitzpatrick","Count"]
    fig2 = px.pie(
        df2,
        names="Fitzpatrick",
        values="Count",
        title="Fitzpatrick Scale"
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("**Drug Allergy Distribution**")
    df3 = (
        filtered["drug allergy"]
          .value_counts()
          .reset_index(name="Count")
    )
    df3.columns = ["Allergy","Count"]
    fig3 = px.bar(
        df3,
        x="Allergy",
        y="Count",
        title="Drug Allergies"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Top Skin Diseases**")
    exploded = filtered["skin disease(s)"].str.split(", ").explode()
    df4 = (
        exploded
          .value_counts()
          .reset_index(name="Count")
          .head(10)
    )
    df4.columns = ["Disease","Count"]
    fig4 = px.bar(
        df4,
        x="Disease",
        y="Count",
        title="Top 10 Skin Diseases"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─── Choropleth Map ───────────────────────────────────────────────────────────────
st.subheader("Patients by Area Map")
try:
    with open('GeoJSON/Indonesia_subdistricts.geojson', 'r', encoding='utf-8') as f:
        geojson = json.load(f)
    grp = filtered.groupby('birth_place').size().reset_index(name='Count')
    fig_map = px.choropleth_mapbox(
        grp,
        geojson=geojson,
        locations='birth_place',
        featureidkey='properties.NAME_3',
        color='Count',
        mapbox_style='carto-positron',
        center={'lat': -2.5, 'lon': 118.0},
        zoom=4,
        title='Patients by Area'
    )
    st.plotly_chart(fig_map)
except Exception as e:
    st.warning(f"Could not load choropleth: {e}")

# ─── Patient Detail Expander ─────────────────────────────────────────────────────
st.subheader("Patient Detail")
if not filtered.empty:
    with st.expander("Select a patient to view full details"):
        patient = st.selectbox("Patient", filtered["name"])
        p = filtered[filtered["name"] == patient].iloc[0]
        st.markdown(f"""
        **Name:** {p['name']}  
        **Birth (Place & Date):** {p['birth_date_and_place']}  
        **Age:** {p['age']}  
        **Address:** {p['address']}  
        **Job:** {p['job']}  
        **Phone:** {p['phone']}  
        **Drug Allergy:** {p['drug allergy']}  
        **Status:** {p['status']}  
        **Skin Type:** {p['skin type']}  
        **Fitzpatrick:** {p['skin type FITZ PATRICK']}  
        **Skin Diseases:** {p['skin disease(s)']}  
        """)
else:
    st.info("No patients match the current filters.")
