import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import traceback

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Customer Analytics Dashboard")

@st.cache_data
def load_data():
    try:
        transactions_df = pd.read_excel("mapped_treatments.xlsx")
        rfm_summary_df = pd.read_excel("all_customer_combined_info_RFM_corrected_full.xlsx")
    except FileNotFoundError as e:
        st.error(f"ERROR: Could not load data files. {e}")
        return pd.DataFrame(), pd.DataFrame()

    if not transactions_df.empty and 'Date' in transactions_df.columns:
        st.markdown("---")
        st.subheader("Debug: Date Parsing in `load_data()`")

        original_dates_copy = transactions_df['Date'].copy()

        st.write("Original 'Date' column sample (first 5 from Excel):", original_dates_copy.head().tolist())
        st.write("Original 'Date' column dtype from Excel:", original_dates_copy.dtype)

        st.write("Converting entire 'Date' column to string before parsing...")
        date_series_as_strings = original_dates_copy.astype(str)
        st.write("Sample after astype(str):", date_series_as_strings.head().tolist())

        st.write("Attempting `pd.to_datetime` with `dayfirst=True` on string-converted dates...")
        parsed_dates = pd.to_datetime(date_series_as_strings, dayfirst=True, errors='coerce')
        st.write(f"After `dayfirst=True`: {parsed_dates.count()} valid dates, {parsed_dates.isnull().sum()} NaT.")
        st.write("Sample of parsed dates after `dayfirst=True`:", parsed_dates.dropna().head().tolist())

        remaining_nat_mask_after_dayfirst = parsed_dates.isnull()
        if remaining_nat_mask_after_dayfirst.any():
            st.write(f"Found {remaining_nat_mask_after_dayfirst.sum()} NaTs after `dayfirst=True`. Trying specific formats for these...")
            
            strings_for_specific_formats = date_series_as_strings[remaining_nat_mask_after_dayfirst]
            
            formats_to_try_for_strings = [
                '%d/%m/%Y',
                '%d/%m/%y',
                # '%Y-%m-%d %H:%M:%S',
                # '%Y-%m-%d',
            ]
            
            reparsed_from_specific_formats = pd.Series([pd.NaT] * len(strings_for_specific_formats), 
                                                       index=strings_for_specific_formats.index, 
                                                       dtype='datetime64[ns]')

            for fmt in formats_to_try_for_strings:
                if reparsed_from_specific_formats.isnull().any():
                    current_nat_in_batch_mask = reparsed_from_specific_formats.isnull()
                    temp_reparsed = pd.to_datetime(strings_for_specific_formats[current_nat_in_batch_mask], 
                                                   format=fmt, errors='coerce')
                    reparsed_from_specific_formats = reparsed_from_specific_formats.fillna(temp_reparsed)
                    st.write(f"Using format '{fmt}' on remaining strings: {reparsed_from_specific_formats.count()} of this batch now parsed.")
            
            parsed_dates = parsed_dates.fillna(reparsed_from_specific_formats)
            st.write(f"After trying specific D/M/Y formats: {parsed_dates.count()} total valid dates, {parsed_dates.isnull().sum()} remain NaT.")

        transactions_df['Date'] = parsed_dates
        
        # --- Final NaT Handling & Year Check ---
        if transactions_df['Date'].isnull().any():
            num_nat_final = transactions_df['Date'].isnull().sum()
            st.warning(f"After all attempts, {num_nat_final} dates remain NaT and will be dropped.")
            st.write("Original values for rows that remained NaT:")
            st.dataframe(pd.DataFrame({
                'Original_Date_Value': original_dates_copy[transactions_df['Date'].isnull()]
            }).head(20))
            transactions_df.dropna(subset=['Date'], inplace=True)
            if transactions_df.empty:
                st.error("All rows dropped after date parsing. Check Excel file for date consistency.")
                return pd.DataFrame(), pd.DataFrame()
        else:
            st.success("All date entries processed successfully (no NaTs remaining).")

        st.markdown("---")
        st.subheader("Debug: Final Processed Date Column Statistics")
        if not transactions_df.empty:
            st.write("Sample of final 'Date' column:", transactions_df['Date'].head().tolist())
            min_date_final = transactions_df['Date'].min()
            max_date_final = transactions_df['Date'].max()
            st.write(f"Min Date in final DataFrame: {min_date_final}")
            st.write(f"Max Date in final DataFrame: {max_date_final}")
            if pd.notna(min_date_final):
                year_counts = transactions_df['Date'].dt.year.value_counts().sort_index()
                st.write("Value counts of years in final 'Date' column:")
                st.dataframe(year_counts)
                
                # Explicitly check for dates beyond your known data range
                EXPECTED_MAX_DATE = pd.to_datetime("2025-04-30")
                if max_date_final > EXPECTED_MAX_DATE:
                    st.error(f"WARNING: Max date found ({max_date_final}) is BEYOND your expected max data date ({EXPECTED_MAX_DATE}). Showing problematic original rows:")
                    problem_rows_max = transactions_df[transactions_df['Date'] > EXPECTED_MAX_DATE]
                    problem_original_values_max = original_dates_copy.loc[problem_rows_max.index]
                    st.dataframe(pd.DataFrame({
                        'Original_Value_From_Excel': problem_original_values_max,
                        'Parsed_As_This_Date': problem_rows_max['Date']
                    }).head(50))
                
                # Also check for very old dates again
                RECENT_YEAR_THRESHOLD = 2023
                if min_date_final.year < RECENT_YEAR_THRESHOLD:
                    st.error(f"WARNING: Minimum date year {min_date_final.year} is before threshold {RECENT_YEAR_THRESHOLD}. Showing problematic original rows:")
                    problem_rows_min = transactions_df[transactions_df['Date'].dt.year < RECENT_YEAR_THRESHOLD]
                    problem_original_values_min = original_dates_copy.loc[problem_rows_min.index]
                    st.dataframe(pd.DataFrame({
                        'Original_Value_From_Excel': problem_original_values_min,
                        'Parsed_As_This_Date': problem_rows_min['Date']
                    }).head(50))
        else:
            st.warning("Transaction DataFrame is empty after NaT drop.")
        st.markdown("---")

        transactions_df['TransactionMonth'] = transactions_df['Date'].dt.to_period('M').astype(str)
        transactions_df['NAMA CUSTOMER'] = transactions_df['NAMA CUSTOMER'].astype(str).str.strip().str.lower()
        
        if 'Total_Price' not in transactions_df.columns:
            st.warning("'Total_Price' column not found. Using placeholder.")
            qty_series = transactions_df.get('QTY', pd.Series(1, index=transactions_df.index)).fillna(1)
            transactions_df['Total_Price'] = qty_series * 100000
            
    elif 'Date' not in transactions_df.columns:
        st.error("'Date' column is missing from the transactions file.")
        return pd.DataFrame(), pd.DataFrame()

    if not rfm_summary_df.empty and 'NAMA CUSTOMER' in rfm_summary_df.columns:
        rfm_summary_df['NAMA CUSTOMER'] = rfm_summary_df['NAMA CUSTOMER'].astype(str).str.strip().str.lower()

    return transactions_df, rfm_summary_df

# --- get_month_year function ---
def get_month_year(period_str):
    try:
        return datetime.strptime(period_str, "%Y-%m").strftime("%b %Y")
    except ValueError:
        try:
            dt_obj = pd.to_datetime(period_str)
            return dt_obj.strftime("%b %Y")
        except:
            return str(period_str)

# --- Load Data ---
if "mapped_treatments.xlsx" in pd.ExcelFile("mapped_treatments.xlsx").sheet_names:
    transactions_df_original_dates = pd.read_excel("mapped_treatments.xlsx", sheet_name="mapped_treatments", usecols=['Date'])['Date']
else:
    transactions_df_original_dates = pd.Series(dtype='object')


transactions_df, rfm_summary_df = load_data()

# --- Main Application ---
st.title("📈 Customer Analytics Dashboard")

if transactions_df.empty:
    st.error("Transaction data is empty after loading and processing. Dashboard functionality will be severely limited.")
    st.stop()
if rfm_summary_df.empty:
    st.warning("RFM summary data could not be loaded. Some parts of the dashboard might not work.")


# --- Descriptive Analytics Tabs ---
st.header("1. Descriptive & Diagnostic Analytics")
descriptive_tabs = st.tabs([
    "📊 RFM Segmentation Overview",
    "🔄 Cohort Analysis & Retention",
    "📈 Time-Series Analysis"
])

with descriptive_tabs[0]:
    st.subheader("RFM Classification Distribution")
    if not rfm_summary_df.empty and 'RFM_Classification' in rfm_summary_df.columns:
        segment_counts = rfm_summary_df['RFM_Classification'].value_counts().reset_index()
        segment_counts.columns = ['RFM_Classification', 'Number of Customers']
        fig_rfm_dist = px.bar(segment_counts, x='RFM_Classification', y='Number of Customers',
                              title="Customer Count by RFM Segment",
                              color='RFM_Classification',
                              labels={'Number of Customers': 'Count'})
        st.plotly_chart(fig_rfm_dist, use_container_width=True)

        display_cols_rfm = ['NAMA CUSTOMER', 'RFM_Classification', 'Total_Spent', 'Recency_Days', 'Frequency']
        if 'Status' in rfm_summary_df.columns: display_cols_rfm.append('Status')
        st.dataframe(rfm_summary_df[display_cols_rfm].head())
    else:
        st.warning("RFM_Classification column not found or RFM summary data is empty.")

with descriptive_tabs[1]: # Cohort Analysis
    st.subheader("Cohort Analysis: Customer Retention")

    if 'Date' not in transactions_df.columns or 'NAMA CUSTOMER' not in transactions_df.columns:
        st.warning("Required columns ('Date', 'NAMA CUSTOMER') missing for Cohort Analysis.")
    elif transactions_df['Date'].isnull().all():
        st.error("All dates in the transaction data are invalid after parsing. Cannot perform Cohort Analysis.")
    else:
        try:
            if transactions_df['Date'].dtype != 'datetime64[ns]':
                st.error("Date column is not in datetime format for Cohort Analysis. Trying to re-convert.")
                transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], errors='coerce')
                transactions_df.dropna(subset=['Date'], inplace=True)
            
            if transactions_df.empty:
                st.error("Transaction data became empty after ensuring Date column is datetime. Cannot perform Cohort Analysis.")
            else:
                transactions_df['CohortMonth'] = transactions_df.groupby('NAMA CUSTOMER')['Date'].transform('min').dt.to_period('M').astype(str)
                transactions_df['TransactionDateObj'] = transactions_df['Date'].dt.to_period('M')
                transactions_df['CohortMonthDateObj'] = pd.to_datetime(transactions_df['CohortMonth']).dt.to_period('M')

                transactions_df['CohortIndex'] = (transactions_df['TransactionDateObj'].dt.year - transactions_df['CohortMonthDateObj'].dt.year) * 12 + \
                                                 (transactions_df['TransactionDateObj'].dt.month - transactions_df['CohortMonthDateObj'].dt.month)

                cohort_data = transactions_df.groupby(['CohortMonth', 'CohortIndex'])['NAMA CUSTOMER'].nunique().reset_index()
                
                cohort_data['CohortMonth_dt'] = pd.to_datetime(cohort_data['CohortMonth'])
                cohort_data = cohort_data.sort_values('CohortMonth_dt')
                
                cohort_counts_abs = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='NAMA CUSTOMER').fillna(0).astype(int)

                if not cohort_counts_abs.empty:
                    cohort_sizes = cohort_counts_abs.iloc[:, 0]
                    retention_matrix_pct = cohort_counts_abs.divide(cohort_sizes, axis=0)
                    
                    y_axis_labels_with_size = []
                    for cohort_month_str, size in cohort_sizes.items():
                        formatted_month = get_month_year(cohort_month_str)
                        y_axis_labels_with_size.append(f"{formatted_month} (N={size})")
                    
                    text_labels = []
                    for i in range(len(retention_matrix_pct.index)):
                        row_labels = []
                        for j in range(len(retention_matrix_pct.columns)):
                            pct_val = retention_matrix_pct.iloc[i, j]
                            count_val = cohort_counts_abs.iloc[i, j]
                            if pd.isna(pct_val) or cohort_sizes.iloc[i] == 0:
                                row_labels.append("0% (0)")
                            else:
                                row_labels.append(f"{pct_val * 100:.1f}% ({count_val})")
                        text_labels.append(row_labels)

                    st.write("Retention Rate (%) and (Absolute Count) by Cohort:")
                    fig_retention_heatmap = go.Figure(data=go.Heatmap(
                        z=retention_matrix_pct.mul(100).round(1).fillna(0).values,
                        x=[f"Month {i}" for i in retention_matrix_pct.columns],
                        y=y_axis_labels_with_size,
                        colorscale='Viridis',
                        text=text_labels, 
                        texttemplate="%{text}", 
                        hoverinfo='z+text', 
                        hovertext=text_labels, 
                        hoverongaps=False))
                    
                    fig_retention_heatmap.update_layout(
                        title='Monthly Customer Retention by Cohort',
                        xaxis_title='Months Since First Purchase (Cohort Index)',
                        yaxis_title='Cohort (First Purchase Month, N = Initial Size)',
                        yaxis_autorange='reversed',
                        height=max(400, len(y_axis_labels_with_size) * 30 + 150) 
                    )
                    st.plotly_chart(fig_retention_heatmap, use_container_width=True)

                    st.write("Initial Cohort Sizes (Number of new customers in Month 0):")
                    cohort_sizes_display = cohort_sizes.copy()
                    cohort_sizes_display.index = [get_month_year(idx) for idx in cohort_sizes_display.index.astype(str)]
                    st.dataframe(cohort_sizes_display.rename("Cohort Size (Month 0)"))

                else:
                    st.warning("Could not generate cohort counts. Check data (especially after date parsing).")
        except Exception as e:
            st.error(f"Error during Cohort Analysis: {e}")
            st.error(traceback.format_exc())

with descriptive_tabs[2]:
    st.subheader("Revenue and Transaction Volume Over Time")

    if 'TransactionMonth' not in transactions_df.columns or 'Total_Price' not in transactions_df.columns:
        st.warning("Required columns ('TransactionMonth', 'Total_Price') missing for Time-Series Revenue Analysis.")
    elif transactions_df['TransactionMonth'].isnull().all() or transactions_df['Date'].isnull().all():
        st.error("TransactionMonth or Date data is all invalid. Cannot perform Time-Series Analysis.")
    else:
        try:
            # Ensure TransactionMonthDate can be created
            monthly_revenue = transactions_df.groupby('TransactionMonth')['Total_Price'].sum().reset_index()
            monthly_revenue['TransactionMonthDate'] = pd.to_datetime(monthly_revenue['TransactionMonth'], errors='coerce')
            monthly_revenue.dropna(subset=['TransactionMonthDate'], inplace=True)
            
            if monthly_revenue.empty:
                st.warning("No valid monthly revenue data after processing TransactionMonth.")
            else:
                monthly_revenue = monthly_revenue.sort_values('TransactionMonthDate')
                fig_revenue = px.line(monthly_revenue, x='TransactionMonthDate', y='Total_Price', title='Monthly Revenue', markers=True)
                fig_revenue.update_xaxes(title_text='Month')
                fig_revenue.update_yaxes(title_text='Total Revenue')
                st.plotly_chart(fig_revenue, use_container_width=True)

            # Monthly Transaction Volume
            monthly_transactions = transactions_df.drop_duplicates(subset=['Date', 'NAMA CUSTOMER']) \
                                                .groupby('TransactionMonth')['NAMA CUSTOMER'].count().reset_index()
            monthly_transactions.rename(columns={'NAMA CUSTOMER': 'TransactionVolume'}, inplace=True)
            monthly_transactions['TransactionMonthDate'] = pd.to_datetime(monthly_transactions['TransactionMonth'], errors='coerce')
            monthly_transactions.dropna(subset=['TransactionMonthDate'], inplace=True)

            if monthly_transactions.empty:
                st.warning("No valid monthly transaction volume data after processing TransactionMonth.")
            else:
                monthly_transactions = monthly_transactions.sort_values('TransactionMonthDate')
                fig_transactions = px.line(monthly_transactions, x='TransactionMonthDate', y='TransactionVolume', title='Monthly Transaction Volume', markers=True)
                fig_transactions.update_xaxes(title_text='Month')
                fig_transactions.update_yaxes(title_text='Number of Unique Customer Transactions')
                st.plotly_chart(fig_transactions, use_container_width=True)
        except Exception as e_ts:
            st.error(f"Error in Time Series plotting: {e_ts}")
            st.error(traceback.format_exc())

    st.subheader("Seasonality Decomposition (Monthly Revenue)")
    if 'monthly_revenue' in locals() and not monthly_revenue.empty and 'TransactionMonthDate' in monthly_revenue.columns:
        if len(monthly_revenue['TransactionMonthDate'].dropna()) >= 24 :
            period = 12
        elif len(monthly_revenue['TransactionMonthDate'].dropna()) >= 4:
            period = max(2, len(monthly_revenue['TransactionMonthDate'].dropna()) // 2)
        else:
            period = 0

        if period > 1 and len(monthly_revenue['TransactionMonthDate'].dropna()) >= 2 * period:
            revenue_series = monthly_revenue.set_index('TransactionMonthDate')['Total_Price']
            try:
                decomposition = seasonal_decompose(revenue_series, model='additive', period=period)
                fig_decomp = go.Figure()
                fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
                fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'))
                fig_decomp.update_layout(title_text=f'Revenue Seasonality Decomposition (Period: {period})', height=600)
                st.plotly_chart(fig_decomp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not perform seasonality decomposition: {e}")
                st.info("This might happen if there are not enough distinct months or too much missing data in the series.")
        else:
            st.warning("Not enough data points (months) for meaningful seasonality decomposition.")
    else:
        st.warning("Monthly revenue data not available or invalid for seasonality decomposition.")


st.header("2. Predictive Analytics")
predictive_tabs = st.tabs([
    "💰 Customer Lifetime Value (CLV)",
    "📉 Churn/Attrition Modeling",
    "⏳ Next-Purchase Timing (Basic)"
])
with predictive_tabs[0]: # CLV
    st.subheader("Probabilistic CLV Modeling (BG/NBD & Gamma-Gamma)")
    clv_results = pd.DataFrame(columns=['NAMA CUSTOMER', 'Predicted_CLV', 'RFM_Classification', 'Total_Spent'])

    if 'Total_Price' not in transactions_df.columns or transactions_df['Total_Price'].isnull().all() or transactions_df['Total_Price'].sum() == 0:
        st.warning("CLV modeling requires a 'Total_Price' column with valid, positive monetary values.")
    elif 'Date' not in transactions_df.columns or 'NAMA CUSTOMER' not in transactions_df.columns:
        st.warning("CLV modeling requires 'Date' and 'NAMA CUSTOMER' columns.")
    elif transactions_df['Date'].isnull().all():
        st.error("All dates in the transaction data are invalid after parsing. Cannot perform CLV modeling.")
    else:
        transactions_df_clv = transactions_df[transactions_df['Total_Price'] > 0].copy()
        if transactions_df_clv.empty:
             st.warning("No transactions with positive Total_Price found for CLV calculation.")
        elif transactions_df_clv['Date'].isnull().all():
             st.warning("All dates in positive Total_Price transactions are invalid. Cannot perform CLV.")
        else:
            max_date = transactions_df_clv['Date'].max()
            lifetimes_df = summary_data_from_transaction_data(
                transactions_df_clv,
                customer_id_col='NAMA CUSTOMER',
                datetime_col='Date',
                monetary_value_col='Total_Price',
                observation_period_end=max_date,
                freq='D'
            )
            lifetimes_df_for_ggf = lifetimes_df[(lifetimes_df['frequency'] > 0) & (lifetimes_df['monetary_value'] > 0)].copy()

            if lifetimes_df.empty or len(lifetimes_df) < 10:
                st.warning("Not enough overall transaction data for robust BGF modeling.")
            elif lifetimes_df_for_ggf.empty or len(lifetimes_df_for_ggf) < 10:
                st.warning("Not enough repeat customers with positive average transaction value for Gamma-Gamma model.")
            else:
                try:
                    bgf = BetaGeoFitter(penalizer_coef=0.01)
                    bgf.fit(lifetimes_df['frequency'], lifetimes_df['recency'], lifetimes_df['T'])
                    
                    ggf = GammaGammaFitter(penalizer_coef=0.01)
                    ggf.fit(lifetimes_df_for_ggf['frequency'], lifetimes_df_for_ggf['monetary_value'])

                    clv_prediction_days = st.slider("Predict CLV for next (days):", 30, 730, 365, 30, key="clv_days_slider_full_code")
                    
                    predicted_clv_series = ggf.customer_lifetime_value(
                        bgf,
                        lifetimes_df_for_ggf['frequency'],
                        lifetimes_df_for_ggf['recency'],
                        lifetimes_df_for_ggf['T'],
                        lifetimes_df_for_ggf['monetary_value'],
                        time=clv_prediction_days / 30.44,
                        freq='D',
                        discount_rate=0.01
                    )
                    
                    clv_ggf_customers = predicted_clv_series.reset_index()
                    
                    if 0 in clv_ggf_customers.columns and 'Predicted_CLV' not in clv_ggf_customers.columns:
                        clv_ggf_customers.rename(columns={0: 'Predicted_CLV'}, inplace=True)
                    elif predicted_clv_series.name is not None and predicted_clv_series.name != 'NAMA CUSTOMER' and 'Predicted_CLV' not in clv_ggf_customers.columns:
                         clv_ggf_customers.rename(columns={predicted_clv_series.name: 'Predicted_CLV'}, inplace=True)
                    elif 'Predicted_CLV' not in clv_ggf_customers.columns and len(clv_ggf_customers.columns) > 1:
                        value_col_name = clv_ggf_customers.columns[-1]
                        if value_col_name != 'NAMA CUSTOMER':
                             clv_ggf_customers.rename(columns={value_col_name: 'Predicted_CLV'}, inplace=True)
                        else:
                             st.error("Could not reliably identify the predicted CLV value column.")
                    elif 'Predicted_CLV' not in clv_ggf_customers.columns:
                         st.error("Predicted CLV column could not be identified or created.")


                    if not rfm_summary_df.empty and 'NAMA CUSTOMER' in rfm_summary_df.columns and \
                       'NAMA CUSTOMER' in clv_ggf_customers.columns and 'Predicted_CLV' in clv_ggf_customers.columns:
                         rfm_summary_df['NAMA CUSTOMER'] = rfm_summary_df['NAMA CUSTOMER'].astype(str)
                         clv_ggf_customers['NAMA CUSTOMER'] = clv_ggf_customers['NAMA CUSTOMER'].astype(str)
                         clv_results = rfm_summary_df.merge(clv_ggf_customers[['NAMA CUSTOMER', 'Predicted_CLV']], 
                                                            on='NAMA CUSTOMER', how='left')
                    elif 'NAMA CUSTOMER' in clv_ggf_customers.columns and 'Predicted_CLV' in clv_ggf_customers.columns:
                         clv_results = clv_ggf_customers[['NAMA CUSTOMER', 'Predicted_CLV']].copy()
                    else:
                        st.warning("Could not prepare `clv_ggf_customers` with 'Predicted_CLV' for merge or final result.")

                    if 'Predicted_CLV' in clv_results.columns:
                        clv_results['Predicted_CLV'] = clv_results['Predicted_CLV'].fillna(0).round(2)
                        clv_results['Predicted_CLV'] = 0 
                        st.warning("'Predicted_CLV' column was not present after processing. Displaying with zeros.")
                
                except Exception as e:
                    st.error(f"An error occurred during CLV modeling: {e}")
                    st.error(traceback.format_exc())
                    st.info("This can happen with insufficient data or model assumptions not met.")

    if not clv_results.empty and 'Predicted_CLV' in clv_results.columns:
        days_to_display = "N/A"
        if 'clv_prediction_days' in locals():
            days_to_display = clv_prediction_days

        st.write(f"Top Customers by Predicted CLV (next {days_to_display} days):")

        display_cols_clv = ['NAMA CUSTOMER', 'Predicted_CLV']
        if 'RFM_Classification' in clv_results.columns: display_cols_clv.insert(1, 'RFM_Classification')
        if 'Total_Spent' in clv_results.columns: display_cols_clv.append('Total_Spent')
        
        final_display_cols_clv = [col for col in display_cols_clv if col in clv_results.columns]

        if final_display_cols_clv and 'NAMA CUSTOMER' in final_display_cols_clv and 'Predicted_CLV' in final_display_cols_clv:
            st.dataframe(clv_results[final_display_cols_clv]
                        .sort_values(by='Predicted_CLV', ascending=False).head(10))

            if clv_results['Predicted_CLV'].sum() > 0:
                fig_clv_dist = px.histogram(clv_results[clv_results['Predicted_CLV'] > 0], x='Predicted_CLV',
                                            nbins=50, title="Distribution of Predicted CLV")
                st.plotly_chart(fig_clv_dist, use_container_width=True)
            else:
                st.info("No positive Predicted CLV values to display in histogram (or all are zero).")
        else:
             st.info("Essential columns for CLV display are missing from the results.")
    else:
        st.info("CLV results are empty or 'Predicted_CLV' column is missing after processing. Cannot display CLV table or distribution.")


with predictive_tabs[1]:
    st.subheader("Churn/Attrition Prediction")
    st.info("""
    **Churn Definition Approach:**
    1.  **Observation Period:** Define a period to observe customer behavior.
    2.  **Churn Window:** Define a subsequent period.
    3.  A customer is 'Churned' if they made purchases in the Observation Period but NOT in the Churn Window.
    """)
    
    churn_model_possible = True
    if 'Date' not in transactions_df.columns or transactions_df['Date'].isnull().all():
        st.warning("Date column is missing or invalid. Cannot proceed with Churn Modeling.")
        churn_model_possible = False
    elif transactions_df['Date'].max() < pd.to_datetime("2025-01-01", dayfirst=True):
        st.warning("Not enough future data (e.g., post Dec 2024) to define a churn window for this example based on Jan 2025 start.")
        churn_model_possible = False
    elif 'NAMA CUSTOMER' not in transactions_df.columns or 'Total_Price' not in transactions_df.columns:
        st.warning("Required columns ('NAMA CUSTOMER', 'Total_Price') missing for Churn Modeling.")
        churn_model_possible = False

    if churn_model_possible:
        try:
            observation_end_date = pd.to_datetime("2024-12-31", dayfirst=True)
            churn_window_start_date = pd.to_datetime("2025-01-01", dayfirst=True)
            churn_window_end_date = transactions_df['Date'].max()

            obs_period_customers = transactions_df[transactions_df['Date'] <= observation_end_date]['NAMA CUSTOMER'].unique()
            
            churn_window_customers = transactions_df[
                (transactions_df['Date'] >= churn_window_start_date) &
                (transactions_df['Date'] <= churn_window_end_date)
            ]['NAMA CUSTOMER'].unique()

            churn_df = pd.DataFrame({'NAMA CUSTOMER': obs_period_customers})
            churn_df['Churned'] = 1
            churn_df.loc[churn_df['NAMA CUSTOMER'].isin(churn_window_customers), 'Churned'] = 0

            obs_period_transactions = transactions_df[transactions_df['Date'] <= observation_end_date]
            if obs_period_transactions.empty:
                st.warning("No transactions found in the defined observation period for churn modeling.")
            else:
                rfm_churn_features = summary_data_from_transaction_data(
                    obs_period_transactions,
                    customer_id_col='NAMA CUSTOMER',
                    datetime_col='Date',
                    monetary_value_col='Total_Price',
                    observation_period_end=observation_end_date,
                    freq='D'
                )
                
                final_churn_df = churn_df.merge(rfm_churn_features.reset_index(), on='NAMA CUSTOMER', how='left').fillna(0)

                if len(final_churn_df) < 20 or final_churn_df['Churned'].nunique() < 2:
                    st.warning("Not enough data or distinct churn outcomes after feature engineering to build a robust churn model.")
                else:
                    features = ['frequency', 'recency', 'T', 'monetary_value']
                    X = final_churn_df[features]
                    y = final_churn_df['Churned']

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
                    
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                         st.warning("Stratified split resulted in a class with no samples in train or test set. Churn model cannot be trained.")
                    else:
                        model = RandomForestClassifier(random_state=42, class_weight='balanced')
                        model.fit(X_train, y_train)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                        st.write("Churn Model Performance (Random Forest):")
                        y_pred_class = (y_pred_proba >= 0.5).astype(int)
                        st.text(classification_report(y_test, y_pred_class, zero_division=0))
                        auc = roc_auc_score(y_test, y_pred_proba)
                        st.write(f"AUC Score: {auc:.2f}")
                        
                        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.2f})'))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
                        fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                                              xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                        st.plotly_chart(fig_roc, use_container_width=True)

                        importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
                        importance_df = importance_df.sort_values('importance', ascending=False)
                        fig_imp = px.bar(importance_df, x='feature', y='importance', title="Feature Importances for Churn")
                        st.plotly_chart(fig_imp, use_container_width=True)
        except ValueError as ve:
            st.error(f"A ValueError occurred during churn modeling: {ve}")
            st.error(traceback.format_exc())
        except Exception as e:
            st.error(f"An unexpected error occurred in churn modeling: {e}")
            st.error(traceback.format_exc())


with predictive_tabs[2]:
    st.subheader("Next-Purchase Timing Insights (Basic)")
    st.info("Insights into time between purchases for repeat customers.")

    if 'Date' not in transactions_df.columns or 'NAMA CUSTOMER' not in transactions_df.columns:
        st.warning("Required columns ('Date', 'NAMA CUSTOMER') missing for Inter-Purchase Time analysis.")
    elif transactions_df['Date'].isnull().all():
        st.warning("Date column is all invalid. Cannot perform Inter-Purchase Time analysis.")
    elif transactions_df.empty or len(transactions_df) < 2:
        st.warning("Not enough transaction data for inter-purchase time analysis.")
    else:
        try:
            inter_purchase_df = transactions_df.sort_values(['NAMA CUSTOMER', 'Date'])
            inter_purchase_df['Prev_Date'] = inter_purchase_df.groupby('NAMA CUSTOMER')['Date'].shift(1)
            inter_purchase_df.dropna(subset=['Prev_Date'], inplace=True)
            
            if not inter_purchase_df.empty:
                inter_purchase_df['Inter_Purchase_Time_Days'] = (inter_purchase_df['Date'] - inter_purchase_df['Prev_Date']).dt.days
                
                inter_purchase_df_filtered = inter_purchase_df[
                    (inter_purchase_df['Inter_Purchase_Time_Days'] >= 0) &
                    (inter_purchase_df['Inter_Purchase_Time_Days'] < 1000)
                ]

                if not inter_purchase_df_filtered.empty:
                    fig_inter_purchase = px.histogram(inter_purchase_df_filtered,
                                                    x='Inter_Purchase_Time_Days',
                                                    title='Distribution of Inter-Purchase Times (Days)',
                                                    nbins=50, labels={'Inter_Purchase_Time_Days': 'Days Between Purchases'})
                    st.plotly_chart(fig_inter_purchase, use_container_width=True)
                    
                    avg_inter_purchase_time = inter_purchase_df_filtered['Inter_Purchase_Time_Days'].mean()
                    if pd.notna(avg_inter_purchase_time):
                        st.write(f"Average inter-purchase time (filtered): {avg_inter_purchase_time:.2f} days")
                    else:
                        st.write("Could not calculate average inter-purchase time (possibly all NaNs after filtering).")
                else:
                    st.warning("No valid inter-purchase times found after filtering. Check for data inconsistencies.")
            else:
                st.warning("No repeat purchases found to calculate inter-purchase times.")
        except Exception as e_ipt:
            st.error(f"Error during Inter-Purchase Time analysis: {e_ipt}")
            st.error(traceback.format_exc())

    st.markdown("---")
    st.write("For more advanced 'Next Purchase Timing', survival models like Cox Proportional Hazards would be appropriate.")