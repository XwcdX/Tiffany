import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from collections import Counter

st.set_page_config(page_title="Customer Spending Dashboard", layout="wide")
st.title("Customer Spending and Treatment Dashboard")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df_member = pd.read_excel("member_spending.xlsx")
        df_all = pd.read_excel("all_customer_spending_status.xlsx")
        df_treat = pd.read_excel("member_treatment_summary.xlsx")
        return df_member, df_all, df_treat
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please make sure the Excel files are in the same directory.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_member, df_all, df_treat = load_data()

# --- Helper function for download buttons ---
def create_download_button(df, filename_stem, button_label):
    # CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"{button_label} (CSV)",
        csv_data,
        f"{filename_stem}_filtered.csv",
        "text/csv",
        key=f"csv_{filename_stem}"
    )
    # Excel
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    st.download_button(
        f"{button_label} (Excel)",
        output.getvalue(),
        f"{filename_stem}_filtered.xlsx",
        key=f"excel_{filename_stem}"
    )

if df_member.empty and df_all.empty and df_treat.empty:
    st.warning("No data loaded. Dashboard will not function correctly.")
    st.stop()

# =========================
# SECTION 1: MEMBER SPENDING
# =========================
st.header("Section 1: Member Spending Analysis")

if not df_member.empty:
    col1_filter, col2_display = st.columns([1, 3])

    with col1_filter:
        st.subheader("Filters")
        names = sorted(df_member["Name"].dropna().unique())
        selected_names = st.multiselect("Filter by Name", names, default=[], help="Leave empty to select all names.")

        provinces = sorted(df_member["Province"].dropna().unique()) if "Province" in df_member.columns else []
        if provinces:
            selected_provinces = st.multiselect("Filter by Province", provinces, default=[], help="Leave empty to select all provinces.")
        else:
            selected_provinces = []

        min_spent = 0
        max_spent = 0
        if "Total_Spent" in df_member.columns and not df_member["Total_Spent"].empty:
            min_spent = int(df_member["Total_Spent"].min())
            max_spent = int(df_member["Total_Spent"].max())
        
        selected_range = st.slider(
            "Filter by Total Spent Range",
            min_value=min_spent,
            max_value=max_spent,
            value=(min_spent, max_spent),
            disabled=(min_spent == max_spent)
        )

    # Apply filters
    filtered_member = df_member.copy()

    if selected_names:
        filtered_member = filtered_member[filtered_member["Name"].isin(selected_names)]
    
    if selected_provinces and provinces:
        filtered_member = filtered_member[filtered_member["Province"].isin(selected_provinces)]
    
    if "Total_Spent" in filtered_member.columns:
        filtered_member = filtered_member[
            (filtered_member["Total_Spent"] >= selected_range[0]) &
            (filtered_member["Total_Spent"] <= selected_range[1])
        ]

    with col2_display:
        st.subheader("Filtered Member Data")
        if not filtered_member.empty:
            total_filtered_members = filtered_member["Name"].nunique()
            total_filtered_spending = filtered_member["Total_Spent"].sum() if "Total_Spent" in filtered_member.columns else 0

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Filtered Members", f"{total_filtered_members:,}")
            metric_col2.metric("Total Spending (Filtered)", f"Rp.{total_filtered_spending:,.2f}")
            
            with st.expander("View Filtered Member Data Table", expanded=False):
                st.dataframe(filtered_member, use_container_width=True)

            if "Total_Spent" in filtered_member.columns and not filtered_member.empty:
                st.subheader("Top 10 Members by Spending")
                top10 = filtered_member.nlargest(10, "Total_Spent").sort_values("Total_Spent", ascending=True)
                
                chart_top10 = alt.Chart(top10).mark_bar().encode(
                    x=alt.X('Total_Spent:Q', title="Total Spent"),
                    y=alt.Y('Name:N', sort='-x', title="Member Name"),
                    tooltip=['Name', 'Total_Spent']
                ).properties(
                    title='Top 10 Spenders'
                )
                st.altair_chart(chart_top10, use_container_width=True)

                st.subheader("Spending Distribution")
                hist_spending = alt.Chart(filtered_member).mark_bar().encode(
                    alt.X('Total_Spent:Q', bin=alt.Bin(maxbins=20), title="Total Spent Bins"),
                    alt.Y('count()', title="Number of Members"),
                    tooltip=[alt.X('Total_Spent:Q', bin=alt.Bin(maxbins=20)), 'count()']
                ).properties(
                    title='Distribution of Total Spending per Member'
                )
                st.altair_chart(hist_spending, use_container_width=True)
            
            create_download_button(filtered_member, "member_spending", "Download Member Spending")
        else:
            st.info("No members match the current filter criteria.")
else:
    st.warning("Member spending data (`member_spending.xlsx`) is not available or empty.")

st.divider()

# =========================
# SECTION 2: CUSTOMER STATUS
# =========================
st.header("Section 2: All Customer Spending and Status")

if not df_all.empty:
    col1_filter_s2, col2_display_s2 = st.columns([1, 3])

    with col1_filter_s2:
        st.subheader("Filters")
        statuses = sorted(df_all["Status"].dropna().unique()) if "Status" in df_all.columns else ["Member", "Non-Member"] # Provide default if column missing
        selected_statuses = st.multiselect("Filter by Status", statuses, default=[], help="Leave empty to select all statuses.")
        
        min_price = 0
        max_price = 0
        if "Total_Price" in df_all.columns and not df_all["Total_Price"].empty:
            min_price = int(df_all["Total_Price"].min())
            max_price = int(df_all["Total_Price"].max())

        selected_price_range = st.slider(
            "Filter by Total Price Range",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            key="price_slider_s2",
            disabled=(min_price == max_price)
        )

    filtered_all = df_all.copy()

    if selected_statuses and "Status" in filtered_all.columns:
        filtered_all = filtered_all[filtered_all["Status"].isin(selected_statuses)]
    
    if "Total_Price" in filtered_all.columns:
        filtered_all = filtered_all[
            (filtered_all["Total_Price"] >= selected_price_range[0]) &
            (filtered_all["Total_Price"] <= selected_price_range[1])
        ]
    
    with col2_display_s2:
        st.subheader("Filtered Customer Data")
        if not filtered_all.empty:
            unique_customer_col = "Customer_ID" if "Customer_ID" in filtered_all.columns else None
            total_filtered_customers = filtered_all[unique_customer_col].nunique() if unique_customer_col else len(filtered_all)
            total_filtered_price_s2 = filtered_all["Total_Price"].sum() if "Total_Price" in filtered_all.columns else 0

            metric_col1_s2, metric_col2_s2 = st.columns(2)
            metric_col1_s2.metric("Filtered Customers", f"{total_filtered_customers:,}")
            metric_col2_s2.metric("Total Price (Filtered)", f"Rp.{total_filtered_price_s2:,.2f}")

            with st.expander("View Filtered Customer Data Table", expanded=False):
                st.dataframe(filtered_all, use_container_width=True)

            if "Status" in filtered_all.columns and "Total_Price" in filtered_all.columns:
                st.subheader("Spending Distribution by Status")
                agg_status_spending = filtered_all.groupby("Status")["Total_Price"].sum().reset_index()
                if not agg_status_spending.empty:
                    pie_status_spending = alt.Chart(agg_status_spending).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta("Total_Price:Q", title="Total Spending"),
                        color=alt.Color("Status:N", title="Status"),
                        tooltip=['Status', 'Total_Price']
                    ).properties(
                        title='Total Spending by Customer Status'
                    )
                    st.altair_chart(pie_status_spending, use_container_width=True)
                else:
                    st.info("Not enough data to display spending distribution by status chart.")

                st.subheader("Average Spending by Status")
                avg_spending_status = filtered_all.groupby("Status")["Total_Price"].mean().reset_index()
                avg_spending_status = avg_spending_status.sort_values("Total_Price", ascending=False)
                if not avg_spending_status.empty:
                    bar_avg_spending = alt.Chart(avg_spending_status).mark_bar().encode(
                        x=alt.X("Status:N", title="Status", sort='-y'),
                        y=alt.Y("Total_Price:Q", title="Average Total Price"),
                        color="Status:N",
                        tooltip=['Status', 'Total_Price']
                    ).properties(
                        title='Average Spending by Customer Status'
                    )
                    st.altair_chart(bar_avg_spending, use_container_width=True)
                else:
                    st.info("Not enough data to display average spending by status chart.")

            create_download_button(filtered_all, "all_customer_spending", "Download All Customer Data")
        else:
            st.info("No customers match the current filter criteria.")
else:
    st.warning("All customer spending data (`all_customer_spending_status.xlsx`) is not available or empty.")

st.divider()

# =========================
# SECTION 3: TREATMENT SUMMARY
# =========================
st.header("Section 3: Member Treatment Summary")

if not df_treat.empty:
    col1_filter_s3, col2_display_s3 = st.columns([1, 3])

    with col1_filter_s3:
        st.subheader("Filters")
        names3 = sorted(df_treat["Name"].dropna().unique()) if "Name" in df_treat.columns else []
        selected_names3 = st.multiselect(
            "Filter by Member Name", 
            names3, 
            default=[], 
            key="name_filter_s3", 
            help="Leave empty to select all names."
        )

    filtered_treat = df_treat.copy()
    if selected_names3 and "Name" in filtered_treat.columns:
        filtered_treat = filtered_treat[filtered_treat["Name"].isin(selected_names3)]
    
    with col2_display_s3:
        st.subheader("Filtered Treatment Data")
        if not filtered_treat.empty:
            metric_col1_s3 = st.columns(1)[0]
            metric_col1_s3.metric("Filtered Members", f"{filtered_treat['Name'].nunique():,}")

            if len(selected_names3) == 1 and "Name" in filtered_treat.columns and "Treatment Summary" in filtered_treat.columns:
                member_summary = filtered_treat[filtered_treat["Name"] == selected_names3[0]]
                if not member_summary.empty:
                    st.subheader(f"Treatment Summary for {selected_names3[0]}")
                    st.text_area("Summary", member_summary["Treatment Summary"].iloc[0], height=100, disabled=True)
            
            with st.expander("View Filtered Treatment Data Table", expanded=False):
                st.dataframe(filtered_treat[["Name", "Treatment Summary"]], use_container_width=True)

            if "Treatment Summary" in filtered_treat.columns:
                treatment_counter = Counter()
                for summary in filtered_treat["Treatment Summary"].dropna():
                    treatments = [part.split("(", 1)[0].strip().title() for part in summary.split(",") if part.strip()]
                    treatment_counter.update(set(treatments))

                if treatment_counter:
                    st.subheader("Top 10 Most Common Treatments (among filtered members)")
                    common_treatments_df = pd.DataFrame(treatment_counter.most_common(10), columns=["Treatment", "Number of Members"])
                    
                    chart_common_treatments = alt.Chart(common_treatments_df).mark_bar().encode(
                        x=alt.X('Number of Members:Q'),
                        y=alt.Y('Treatment:N', sort='-x', title="Treatment"),
                        tooltip=['Treatment', 'Number of Members']
                    ).properties(
                        title='Top 10 Common Treatments'
                    )
                    st.altair_chart(chart_common_treatments, use_container_width=True)
                else:
                    st.info("No treatment data to analyze for the selected members.")
            
            create_download_button(filtered_treat, "member_treatment_summary", "Download Treatment Summary")
        else:
            st.info("No members match the current filter criteria for treatments.")
else:
    st.warning("Member treatment summary data (`member_treatment_summary.xlsx`) is not available or empty.")

st.sidebar.info("Dashboard by Your Name/Team")