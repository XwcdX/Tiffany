import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from collections import Counter
import re

# --- Page Configuration (Keep this at the top) ---
st.set_page_config(page_title="Customer Analysis Dashboard", layout="wide")

st.title("Customer Analysis Dashboard")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df_member = pd.read_excel("member_spending.xlsx")
        df_all = pd.read_excel("all_customer_combined_final.xlsx")
        # df_treat = pd.read_excel("member_treatment_summary.xlsx")
        return df_member, df_all #, df_treat
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please ensure 'member_spending.xlsx' and 'all_customer_combined_info.xlsx' are in the same directory.")
        return pd.DataFrame(), pd.DataFrame() #, pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while reading Excel files: {e}")
        return pd.DataFrame(), pd.DataFrame() #, pd.DataFrame()

df_member, df_all = load_data()

# --- Helper function for download buttons ---
def create_download_button(df, filename_stem, button_label, key_suffix=""):
    if df.empty:
        st.info(f"No data to download for {button_label}.")
        return
    # CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"{button_label} (CSV)",
        csv_data,
        f"{filename_stem}_filtered.csv",
        "text/csv",
        key=f"csv_{filename_stem}{key_suffix}"
    )
    # Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtered Data')
    st.download_button(
        f"{button_label} (Excel)",
        output.getvalue(),
        f"{filename_stem}_filtered.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"excel_{filename_stem}{key_suffix}"
    )

if df_all.empty : # Only checking df_all as it's crucial for Section 2
    st.warning("Core customer data ('all_customer_combined_info.xlsx') not loaded. Dashboard may not function correctly.")
    st.stop()

# =========================
# SECTION 1: MEMBER SPENDING (Optional - kept for now, can be removed if not needed)
# =========================
# You can comment out this entire section if Section 1 is not needed.
if not df_member.empty:
    st.header("Section 1: Detailed Member Spending Analysis")
    col1_filter_s1, col2_display_s1 = st.columns([1, 3])

    with col1_filter_s1:
        st.subheader("Filters (Section 1)")
        if "Name" in df_member.columns:
            names_s1 = sorted(df_member["Name"].dropna().unique())
            selected_names_s1 = st.multiselect("Filter by Member Name (S1)", names_s1, default=[], help="Leave empty to select all names.")
        else:
            selected_names_s1 = []

        if "Province" in df_member.columns:
            provinces_s1 = sorted(df_member["Province"].dropna().unique())
            selected_provinces_s1 = st.multiselect("Filter by Province (S1)", provinces_s1, default=[], help="Leave empty to select all provinces.")
        else:
            selected_provinces_s1 = []

        min_spent_s1, max_spent_s1 = 0, 0
        if "Total_Spent" in df_member.columns and not df_member["Total_Spent"].empty:
            min_spent_s1 = int(df_member["Total_Spent"].min())
            max_spent_s1 = int(df_member["Total_Spent"].max())
        selected_range_s1 = st.slider(
            "Filter by Total Spent Range (S1)",
            min_value=min_spent_s1, max_value=max_spent_s1, value=(min_spent_s1, max_spent_s1),
            disabled=(min_spent_s1 == max_spent_s1), key="slider_s1"
        )

    filtered_member_s1 = df_member.copy()
    if selected_names_s1 and "Name" in filtered_member_s1.columns:
        filtered_member_s1 = filtered_member_s1[filtered_member_s1["Name"].isin(selected_names_s1)]
    if selected_provinces_s1 and "Province" in filtered_member_s1.columns:
        filtered_member_s1 = filtered_member_s1[filtered_member_s1["Province"].isin(selected_provinces_s1)]
    if "Total_Spent" in filtered_member_s1.columns:
        filtered_member_s1 = filtered_member_s1[
            (filtered_member_s1["Total_Spent"] >= selected_range_s1[0]) &
            (filtered_member_s1["Total_Spent"] <= selected_range_s1[1])
        ]

    with col2_display_s1:
        st.subheader("Filtered Member Data (Section 1)")
        if not filtered_member_s1.empty:
            num_members = filtered_member_s1["Name"].nunique() if "Name" in filtered_member_s1.columns else 0
            total_spending = filtered_member_s1["Total_Spent"].sum() if "Total_Spent" in filtered_member_s1.columns else 0
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Filtered Members (S1)", f"{num_members:,}")
            m_col2.metric("Total Spending (S1)", f"Rp {total_spending:,.0f}")

            with st.expander("View Filtered Member Data Table (S1)", expanded=False):
                st.dataframe(filtered_member_s1, use_container_width=True)
            # ... (Charts for Section 1 can be added back here if needed) ...
            create_download_button(filtered_member_s1, "member_spending_s1", "Download S1 Data", "_s1")
        else:
            st.info("No members match S1 filter criteria.")
    st.divider()
# End of Optional Section 1

# =========================
# SECTION 2: ALL CUSTOMER RFM & TREATMENT ANALYSIS
# =========================
st.header("Customer Segmentation and Analysis (RFM & Treatments)")

if not df_all.empty:
    # --- Check for essential RFM columns ---
    rfm_cols = ['NAMA CUSTOMER', 'Total_Spent', 'Status',
                'Recency_Days', 'Frequency', 'R_Score', 'F_Score', 'M_Score', 'RFM_Classification',
                'Treatment Summary']
    missing_cols = [col for col in rfm_cols if col not in df_all.columns]
    if missing_cols:
        st.error(f"The 'all_customer_combined_info.xlsx' file is missing required columns for RFM analysis: {', '.join(missing_cols)}. Please regenerate the file with all RFM fields.")
        st.stop()

    col1_filter_s2, col2_display_s2 = st.columns([1, 2.5])

    with col1_filter_s2:
        st.subheader("Filters")

        customer_names_s2 = sorted(df_all["NAMA CUSTOMER"].dropna().unique())
        selected_customer_names_s2 = st.multiselect(
            "Filter by Customer Name", customer_names_s2, default=[], key="customer_name_filter_s2"
        )

        # Filter by Status (Member/Non-Member)
        statuses = sorted(df_all["Status"].dropna().unique())
        selected_statuses = st.multiselect("Filter by Status", statuses, default=statuses, key="status_filter_s2")

        # Filter by RFM Classification
        rfm_classifications = sorted(df_all["RFM_Classification"].dropna().unique())
        selected_rfm_class = st.multiselect("Filter by RFM Segment", rfm_classifications, default=[], key="rfm_class_filter")

        # Filter by R_Score, F_Score, M_Score
        r_scores = sorted(df_all["R_Score"].dropna().unique())
        selected_r_scores = st.multiselect("Filter by Recency Score (R)", r_scores, default=[], key="r_score_filter")

        f_scores = sorted(df_all["F_Score"].dropna().unique())
        selected_f_scores = st.multiselect("Filter by Frequency Score (F)", f_scores, default=[], key="f_score_filter")

        m_scores = sorted(df_all["M_Score"].dropna().unique())
        selected_m_scores = st.multiselect("Filter by Monetary Score (M)", m_scores, default=[], key="m_score_filter")


        # Filter by Total Spent
        min_spent_s2 = 0
        if not df_all["Total_Spent"].empty: min_spent_s2 = int(df_all["Total_Spent"].min())
        max_spent_s2 = 0
        if not df_all["Total_Spent"].empty: max_spent_s2 = int(df_all["Total_Spent"].max())
        selected_spent_range_s2 = st.slider(
            "Filter by Total Spent Range",
            min_value=min_spent_s2, max_value=max_spent_s2, value=(min_spent_s2, max_spent_s2),
            key="spent_slider_s2", disabled=(min_spent_s2 == max_spent_s2)
        )
        
        # Filter by Recency Days
        min_recency_s2 = 0
        if not df_all["Recency_Days"].empty: min_recency_s2 = int(df_all["Recency_Days"].min())
        max_recency_s2 = 0
        if not df_all["Recency_Days"].empty: max_recency_s2 = int(df_all["Recency_Days"].max())
        selected_recency_range_s2 = st.slider(
            "Filter by Recency (Days)",
            min_value=min_recency_s2, max_value=max_recency_s2, value=(min_recency_s2, max_recency_s2),
            key="recency_slider_s2", disabled=(min_recency_s2 == max_recency_s2)
        )

        # Filter by Frequency
        min_freq_s2 = 0
        if not df_all["Frequency"].empty: min_freq_s2 = int(df_all["Frequency"].min())
        max_freq_s2 = 0
        if not df_all["Frequency"].empty: max_freq_s2 = int(df_all["Frequency"].max())
        selected_freq_range_s2 = st.slider(
            "Filter by Frequency (Visits)",
            min_value=min_freq_s2, max_value=max_freq_s2, value=(min_freq_s2, max_freq_s2),
            key="freq_slider_s2", disabled=(min_freq_s2 == max_freq_s2)
        )


    # --- Apply filters to df_all ---
    filtered_all_s2 = df_all.copy()
    if selected_customer_names_s2:
        filtered_all_s2 = filtered_all_s2[filtered_all_s2["NAMA CUSTOMER"].isin(selected_customer_names_s2)]
    if selected_statuses:
        filtered_all_s2 = filtered_all_s2[filtered_all_s2["Status"].isin(selected_statuses)]
    if selected_rfm_class:
        filtered_all_s2 = filtered_all_s2[filtered_all_s2["RFM_Classification"].isin(selected_rfm_class)]
    if selected_r_scores:
        filtered_all_s2 = filtered_all_s2[filtered_all_s2["R_Score"].isin(selected_r_scores)]
    if selected_f_scores:
        filtered_all_s2 = filtered_all_s2[filtered_all_s2["F_Score"].isin(selected_f_scores)]
    if selected_m_scores:
        filtered_all_s2 = filtered_all_s2[filtered_all_s2["M_Score"].isin(selected_m_scores)]

    filtered_all_s2 = filtered_all_s2[
        (filtered_all_s2["Total_Spent"] >= selected_spent_range_s2[0]) &
        (filtered_all_s2["Total_Spent"] <= selected_spent_range_s2[1])
    ]
    filtered_all_s2 = filtered_all_s2[
        (filtered_all_s2["Recency_Days"] >= selected_recency_range_s2[0]) &
        (filtered_all_s2["Recency_Days"] <= selected_recency_range_s2[1])
    ]
    filtered_all_s2 = filtered_all_s2[
        (filtered_all_s2["Frequency"] >= selected_freq_range_s2[0]) &
        (filtered_all_s2["Frequency"] <= selected_freq_range_s2[1])
    ]


    with col2_display_s2:
        st.subheader("Filtered Customer Data Overview")
        if not filtered_all_s2.empty:
            total_filtered_customers_s2 = filtered_all_s2["NAMA CUSTOMER"].nunique()
            total_filtered_spent_s2 = filtered_all_s2["Total_Spent"].sum()
            avg_recency = filtered_all_s2["Recency_Days"].mean()
            avg_frequency = filtered_all_s2["Frequency"].mean()
            avg_monetary = filtered_all_s2["Total_Spent"].mean()


            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Filtered Customers", f"{total_filtered_customers_s2:,}")
            m_col2.metric("Total Spending (Filtered)", f"Rp {total_filtered_spent_s2:,.0f}")
            
            m_col3, m_col4, m_col5 = st.columns(3)
            m_col3.metric("Avg. Recency", f"{avg_recency:,.1f} days")
            m_col4.metric("Avg. Frequency", f"{avg_frequency:,.1f} visits")
            m_col5.metric("Avg. Monetary Value", f"Rp {avg_monetary:,.0f}")


            with st.expander("View Filtered Customer Data Table (with RFM)", expanded=True):
                display_cols = ['NAMA CUSTOMER', 'Status', 'RFM_Classification', 'R_Score', 'F_Score', 'M_Score',
                                'Recency_Days', 'Frequency', 'Total_Spent', 'Treatment Summary']
                st.dataframe(filtered_all_s2[display_cols], use_container_width=True)

            # --- RFM Analysis Visualizations ---
            st.subheader("RFM Segmentation Analysis")
            
            # Distribution of Customers by RFM Classification
            rfm_dist_data = filtered_all_s2['RFM_Classification'].value_counts().reset_index()
            rfm_dist_data.columns = ['RFM_Classification', 'Number of Customers']
            if not rfm_dist_data.empty:
                chart_rfm_dist = alt.Chart(rfm_dist_data).mark_bar().encode(
                    x=alt.X('Number of Customers:Q'),
                    y=alt.Y('RFM_Classification:N', sort='-x', title="RFM Segment"),
                    tooltip=['RFM_Classification', 'Number of Customers']
                ).properties(
                    title='Customer Distribution by RFM Segment'
                )
                st.altair_chart(chart_rfm_dist, use_container_width=True)

            # Average Monetary Value by RFM Classification
            avg_monetary_by_rfm = filtered_all_s2.groupby('RFM_Classification')['Total_Spent'].mean().reset_index()
            avg_monetary_by_rfm.columns = ['RFM_Classification', 'Average Total Spent']
            if not avg_monetary_by_rfm.empty:
                chart_avg_monetary_rfm = alt.Chart(avg_monetary_by_rfm).mark_bar().encode(
                    x=alt.X('Average Total Spent:Q', title='Average Spending (Rp)'),
                    y=alt.Y('RFM_Classification:N', sort=alt.EncodingSortField(field="Average Total Spent", op="sum", order='descending'), title="RFM Segment"),
                    tooltip=['RFM_Classification', alt.Tooltip('Average Total Spent:Q', format=',.0f')]
                ).properties(
                    title='Average Spending by RFM Segment'
                )
                st.altair_chart(chart_avg_monetary_rfm, use_container_width=True)

            # RFM Scatter Plot (e.g., Recency vs Frequency, color by Monetary or Segment)
            if len(filtered_all_s2) > 1 and len(filtered_all_s2) < 5000:
                st.subheader("RFM Score Scatter Plot")
                scatter_rfm = alt.Chart(filtered_all_s2).mark_circle(size=60).encode(
                    x=alt.X('Recency_Days:Q', scale=alt.Scale(zero=False), title="Recency (Days)"),
                    y=alt.Y('Frequency:Q', scale=alt.Scale(zero=False), title="Frequency (Visits)"),
                    color=alt.Color('M_Score:N', title="Monetary Score"),
                    size=alt.Size('Total_Spent:Q', title="Total Spent", legend=None),
                    tooltip=['NAMA CUSTOMER', 'RFM_Classification', 'Recency_Days', 'Frequency', 'Total_Spent']
                ).properties(
                    title='Recency vs. Frequency (Colored by Monetary Score)'
                ).interactive()
                st.altair_chart(scatter_rfm, use_container_width=True)
            elif len(filtered_all_s2) >= 5000:
                st.info("Scatter plot is not displayed for >5000 data points for performance reasons. Please apply more filters.")


            # --- Spending Analysis (Original from your code, adapted) ---
            st.subheader("Overall Spending Analysis")
            agg_status_spending = filtered_all_s2.groupby("Status")["Total_Spent"].sum().reset_index()
            if not agg_status_spending.empty and agg_status_spending["Total_Spent"].sum() > 0:
                pie_status_spending = alt.Chart(agg_status_spending).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("Total_Spent:Q", title="Total Spending"),
                    color=alt.Color("Status:N", title="Status"),
                    tooltip=['Status', alt.Tooltip('Total_Spent:Q', title='Total Spent', format=',.0f')]
                ).properties(title='Total Spending by Customer Status')
                st.altair_chart(pie_status_spending, use_container_width=True)

            # --- Treatment Analysis (Original from your code, adapted) ---
            st.subheader("Treatment Analysis")
            treatment_total_counts = Counter()
            for summary_str in filtered_all_s2["Treatment Summary"].dropna():
                if not str(summary_str).strip(): continue
                parts = str(summary_str).split(',')
                for part in parts:
                    part = part.strip()
                    if not part: continue
                    match = re.match(r"(.+?)\s*\((\d+)\)", part)
                    if match:
                        treatment_name = match.group(1).strip().title()
                        count = int(match.group(2))
                        treatment_total_counts[treatment_name] += count
            
            if treatment_total_counts:
                top_n_treatments = st.slider("Number of Top Treatments to Display", 5, 20, 10, key="top_n_treat_s2")
                top_sold_treatments_df = pd.DataFrame(
                    treatment_total_counts.most_common(top_n_treatments),
                    columns=["Treatment", "Total Times Sold"]
                )
                if not top_sold_treatments_df.empty:
                    chart_top_sold_treatments = alt.Chart(top_sold_treatments_df).mark_bar().encode(
                        x=alt.X('Total Times Sold:Q'),
                        y=alt.Y('Treatment:N', sort='-x', title="Treatment Name"),
                        tooltip=['Treatment', 'Total Times Sold']
                    ).properties(title=f'Top {top_n_treatments} Most Sold Treatments')
                    st.altair_chart(chart_top_sold_treatments, use_container_width=True)
            else:
                st.info("No treatment summary data available or parsed for the selected customers.")

            create_download_button(filtered_all_s2, "all_customer_analysis", "Download Full Analysis Data", "_s2")
        else:
            st.info("No customers match the current filter criteria.")
else:
    st.warning("All customer combined data (`all_customer_combined_info.xlsx`) with RFM fields is not available or empty.")