import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import warnings
import calendar
from scipy import stats

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Sales Analytics Dashboard | لوحة تحليلات المبيعات",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Dashboard ---
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Cards styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .metric-card h3 {
        font-size: 12px;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        font-size: 24px;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 5px;
    }

    .metric-card .change {
        font-size: 12px;
        display: flex;
        align-items: center;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Custom titles */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Better chart containers */
    .plot-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Multilingual Support ---
translations = {
    "en": {
        "title": "📊 Sales Analytics Dashboard",
        "upload": "Upload Historical Sales Data",
        "date_col": "Date Column",
        "sales_col": "Sales Column",
        "process": "Process & Analyze Data",
        "forecast_days": "Days to Forecast",
        "generate": "Generate Forecast",
        "model": "Forecasting Model",
        "prophet": "Prophet",
        "lstm": "LSTM",
        "data_analysis": "Data Analysis",
        "forecasting": "Forecasting",
        # New Translations added below:
        "eda_title": "📈 Exploratory Data Analysis",
        "summary_stats": "📊 Summary Statistics",
        "total_sales": "Total Sales",
        "avg_daily_sales": "Average Daily Sales",
        "peak_sales": "Peak Sales Day",
        "sales_var": "Sales Variability",
        "sales_trends": "📈 Sales Trends Over Time",
        "view_by": "View by",
        "show_trend": "Show Trend Line",
        "seasonality": "🌊 Seasonality Analysis",
        "distribution": "📊 Sales Distribution Analysis",
        "outliers": "🔍 Outlier Detection",
        "growth": "📈 Growth Analysis",
        "data_quality": "📋 Data Quality Report",
        "insights": "💡 Key Insights & Recommendations",
        "forecast_title": "🔮 Sales Forecasting",
        "export_btn": "📥 Export Forecast to CSV",
"additional_cols": "➕ Additional Analysis Columns",
"select_additional": "Select additional columns for analysis (optional):"

    },
    "ar": {
        "title": "📊 لوحة تحليلات المبيعات",
        "upload": "رفع بيانات المبيعات التاريخية",
        "date_col": "عمود التاريخ",
        "sales_col": "عمود المبيعات",
        "process": "معالجة وتحليل البيانات",
        "forecast_days": "عدد أيام التوقع",
        "generate": "إنشاء التوقع",
        "model": "نموذج التنبؤ",
        "prophet": "Prophet",
        "lstm": "LSTM",
        "data_analysis": "تحليل البيانات",
        "forecasting": "التنبؤ",
        # الترجمات الجديدة:
        "eda_title": "📈 تحليل البيانات الاستكشافي",
        "summary_stats": "📊 الإحصائيات العامة",
        "total_sales": "إجمالي المبيعات",
        "avg_daily_sales": "متوسط المبيعات اليومية",
        "peak_sales": "أعلى يوم مبيعات",
        "sales_var": "تباين المبيعات",
        "sales_trends": "📈 اتجاهات المبيعات بمرور الوقت",
        "view_by": "عرض حسب",
        "show_trend": "إظهار خط الاتجاه",
        "seasonality": "🌊 تحليل الموسمية",
        "distribution": "📊 تحليل توزيع المبيعات",
        "outliers": "🔍 الكشف عن القيم الشاذة",
        "growth": "📈 تحليل النمو",
        "data_quality": "📋 تقرير جودة البيانات",
        "insights": "💡 رؤى وتوصيات رئيسية",
        "forecast_title": "🔮 التنبؤ بالمبيعات",
        "export_btn": "📥 تصدير التوقعات (CSV)",
"additional_cols": "➕ أعمدة تحليل إضافية",
"select_additional": "اختر أعمدة إضافية للتحليل (اختياري)"
    }
}

# Language switcher
if "lang" not in st.session_state:
    st.session_state.lang = "en"

col1, col2 = st.columns([1, 10])
with col1:
    if st.button("العربية" if st.session_state.lang == "en" else "English"):
        st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
        st.rerun()

lang = st.session_state.lang
t = translations[lang]

# RTL for Arabic
if lang == "ar":
    st.markdown("<style> body {direction: rtl; text-align: right;} </style>", unsafe_allow_html=True)

st.title(t["title"])
# كود لقلب الواجهة لليمين
if lang == "ar":
    st.markdown("""
        <style>
        /* الصفحة كاملة من اليمين لليسار */
        .stApp { direction: rtl; }

        /* محاذاة كل النصوص والعناوين والقوائم لليمين */
        h1, h2, h3, h4, h5, p, span, label, div { text-align: right !important; }
        .stSelectbox, .stButton, .stNumberInput, .stSlider { direction: rtl; text-align: right; }

        /* تعديل التبويبات (Tabs) لتكون من اليمين */
        .stTabs [data-baseweb="tab-list"] { 
            direction: rtl; 
            display: flex; 
            justify-content: flex-start; 
        }

        /* منع الرسوم البيانية من الانقلاب لتبقى الأرقام صحيحة */
        .js-plotly-plot .plotly { direction: ltr; }
        </style>
    """, unsafe_allow_html=True)
# --- Data Upload ---
uploaded_file = st.file_uploader(t["upload"], type=["csv", "xlsx"],
                                 help="Upload your sales data in CSV or Excel format")

if uploaded_file is not None:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith(".csv"):
            try:
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding='latin1')
        else:  # Excel file
            df_raw = pd.read_excel(uploaded_file)

        st.success(f"✅ File loaded successfully! {len(df_raw)} records found.")

        # Display raw data preview
        with st.expander("📄 Raw Data Preview", expanded=True):
            st.dataframe(df_raw.head(100), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_raw))
            with col2:
                st.metric("Columns", len(df_raw.columns))
            with col3:
                st.metric("Missing Values", df_raw.isnull().sum().sum())

        # Check for duplicate column names
        if df_raw.columns.duplicated().any():
            st.warning("⚠️ Duplicate column names detected. They will be renamed.")
            cols = pd.Series(df_raw.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols == dup] = [dup + f'_{i}' if i != 0 else dup for i in range(sum(cols == dup))]
            df_raw.columns = cols

        # Column selection
        st.subheader("⚙️ Configure Data Columns")
        cols = df_raw.columns.tolist()

        # Auto-detect date columns
        date_candidates = []
        for col in cols:
            if df_raw[col].dtype == 'object':
                try:
                    pd.to_datetime(df_raw[col].head())
                    date_candidates.append(col)
                except:
                    if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
                        date_candidates.append(col)
            elif 'date' in col.lower() or 'time' in col.lower():
                date_candidates.append(col)

        # Auto-detect numeric columns for sales
        numeric_candidates = []
        for col in cols:
            if pd.api.types.is_numeric_dtype(df_raw[col]):
                numeric_candidates.append(col)

        col1, col2 = st.columns(2)
        with col1:
            if date_candidates:
                default_date = date_candidates[0]
                st.info(f"Suggested date columns: {', '.join(date_candidates[:3])}")
            else:
                default_date = cols[0]
            date_col = st.selectbox(
                t["date_col"],
                cols,
                index=cols.index(default_date) if default_date in cols else 0,
                help="Select the column containing dates"
            )

        with col2:
            if numeric_candidates:
                # Prioritize columns with sales-related names
                sales_keywords = ['sales', 'amount', 'price', 'quantity', 'qty', 'total', 'revenue', 'profit', 'value']
                prioritized = [col for col in numeric_candidates if
                               any(keyword in col.lower() for keyword in sales_keywords)]
                default_sales = prioritized[0] if prioritized else numeric_candidates[0]
                st.info(f"Suggested sales columns: {', '.join(numeric_candidates[:3])}")
            else:
                default_sales = cols[1] if len(cols) > 1 else cols[0]
            sales_col = st.selectbox(
                t["sales_col"],
                cols,
                index=cols.index(default_sales) if default_sales in cols else 0,
                help="Select the column containing sales values"
            )

        # Additional analysis columns
        with st.expander(t["additional_cols"]):
            st.write(t["select_additional"])
            additional_cols = st.multiselect(
                "Select columns for deeper analysis",
                [col for col in cols if col not in [date_col, sales_col]],
                help="Select product categories, regions, or other dimensions"
            )

        if st.button(t["process"], type="primary", use_container_width=True):
            with st.spinner("Processing and analyzing data..."):
                try:
                    # Create a copy for processing
                    df_processing = df_raw.copy()

                    # Convert date column
                    df_processing[date_col] = pd.to_datetime(df_processing[date_col], errors='coerce')

                    # Remove rows with invalid dates or sales
                    mask = df_processing[date_col].notna() & df_processing[sales_col].notna()
                    df_clean = df_processing[mask].copy()

                    if len(df_clean) == 0:
                        st.error("❌ No valid data found after cleaning. Please check your date and sales columns.")
                    else:
                        # Sort by date
                        df_clean = df_clean.sort_values(date_col)

                        # Create daily aggregated dataset
                        df_daily = df_clean.groupby(date_col)[sales_col].sum().reset_index()
                        df_daily = df_daily.rename(columns={date_col: 'ds', sales_col: 'y'})

                        # Store in session state
                        st.session_state.df_raw = df_raw
                        st.session_state.df_clean = df_clean
                        st.session_state.df_daily = df_daily
                        st.session_state.date_col = date_col
                        st.session_state.sales_col = sales_col
                        st.session_state.additional_cols = additional_cols

                        st.success(
                            f"✅ Data processed successfully! {len(df_daily)} unique dates available for analysis.")

                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# =============== DATA ANALYSIS SECTION ===============
if "df_daily" in st.session_state:
    df_daily = st.session_state.df_daily
    df_clean = st.session_state.df_clean
    date_col_name = st.session_state.date_col
    sales_col_name = st.session_state.sales_col

    st.markdown("---")

    # Create tabs for analysis and forecasting
    tab1, tab2 = st.tabs([t["data_analysis"], t["forecasting"]])

    with tab1:
        # =============== EXPLORATORY DATA ANALYSIS ===============
        st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

        # Summary Statistics Cards
        st.subheader("📊 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Sales</h3>
                <div class="value">${:,.2f}</div>
                <div class="change positive">📈 All time</div>
            </div>
            """.format(df_daily['y'].sum()), unsafe_allow_html=True)

        with col2:
            avg_sales = df_daily['y'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Average Daily Sales</h3>
                <div class="value">${:,.2f}</div>
                <div class="change">📊 Per day</div>
            </div>
            """.format(avg_sales), unsafe_allow_html=True)

        with col3:
            max_sales = df_daily['y'].max()
            max_date = df_daily.loc[df_daily['y'].idxmax(), 'ds']
            st.markdown("""
            <div class="metric-card">
                <h3>Peak Sales Day</h3>
                <div class="value">${:,.2f}</div>
                <div class="change">📅 {}</div>
            </div>
            """.format(max_sales, max_date.strftime('%Y-%m-%d')), unsafe_allow_html=True)

        with col4:
            std_sales = df_daily['y'].std()
            cv = (std_sales / avg_sales) * 100 if avg_sales > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <h3>Sales Variability</h3>
                <div class="value">{:.1f}%</div>
                <div class="change">📊 Coefficient of Variation</div>
            </div>
            """.format(cv), unsafe_allow_html=True)

        # =============== TIME SERIES VISUALIZATION ===============
        st.subheader("📈 Sales Trends Over Time")

        col1, col2 = st.columns([3, 1])
        with col1:
            time_granularity = st.selectbox(
                "View by",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                index=0
            )

        with col2:
            show_trend = st.checkbox("Show Trend Line", value=True)

        # Prepare data based on granularity
        df_plot = df_daily.copy()
        df_plot['week'] = df_plot['ds'].dt.isocalendar().week
        df_plot['month'] = df_plot['ds'].dt.month
        df_plot['quarter'] = df_plot['ds'].dt.quarter
        df_plot['year'] = df_plot['ds'].dt.year

        if time_granularity == "Weekly":
            df_grouped = df_plot.groupby(['year', 'week'])['y'].sum().reset_index()
            df_grouped['period'] = df_grouped['year'].astype(str) + '-W' + df_grouped['week'].astype(str).str.zfill(2)
            x_col = 'period'
        elif time_granularity == "Monthly":
            df_grouped = df_plot.groupby(['year', 'month'])['y'].sum().reset_index()
            df_grouped['period'] = df_grouped['year'].astype(str) + '-' + df_grouped['month'].astype(str).str.zfill(2)
            df_grouped['date'] = pd.to_datetime(df_grouped['period'] + '-01')
            x_col = 'date'
        elif time_granularity == "Quarterly":
            df_grouped = df_plot.groupby(['year', 'quarter'])['y'].sum().reset_index()
            df_grouped['period'] = df_grouped['year'].astype(str) + '-Q' + df_grouped['quarter'].astype(str)
            x_col = 'period'
        elif time_granularity == "Yearly":
            df_grouped = df_plot.groupby(['year'])['y'].sum().reset_index()
            df_grouped['period'] = df_grouped['year'].astype(str)
            x_col = 'period'
        else:  # Daily
            df_grouped = df_daily.copy()
            x_col = 'ds'

        # Create time series plot
        fig = px.line(df_grouped, x=x_col, y='y',
                      title=f'Sales Trend ({time_granularity} View)',
                      labels={'y': 'Sales Amount ($)', x_col: 'Time Period'},
                      template='plotly_white')

        if show_trend and time_granularity in ["Daily", "Weekly"]:
            # Add trend line using rolling average
            window_size = min(30, len(df_grouped) // 4)
            if window_size > 1:
                df_grouped['trend'] = df_grouped['y'].rolling(window=window_size, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=df_grouped[x_col],
                    y=df_grouped['trend'],
                    mode='lines',
                    name=f'{window_size}-period Moving Average',
                    line=dict(color='red', width=3, dash='dash')
                ))

        fig.update_layout(
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # =============== SEASONALITY ANALYSIS ===============
        st.subheader(f"🌊 Seasonality Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Monthly seasonality
            df_monthly = df_clean.copy()
            df_monthly['month'] = df_monthly[date_col_name].dt.month
            df_monthly['month_name'] = df_monthly[date_col_name].dt.strftime('%B')
            monthly_avg = df_monthly.groupby(['month', 'month_name'])[sales_col_name].mean().reset_index()
            monthly_avg = monthly_avg.sort_values('month')

            fig_monthly = px.bar(monthly_avg, x='month_name', y=sales_col_name,
                                 title=f'Average Sales by Month',
                                 labels={sales_col_name: 'Average Sales ($)', 'month_name': 'Month'},
                                 color_discrete_sequence=['#636efa'])
            fig_monthly.update_layout(height=300)
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            # Day of week seasonality
            df_weekly = df_clean.copy()
            df_weekly['day_of_week'] = df_weekly[date_col_name].dt.day_name()
            df_weekly['day_num'] = df_weekly[date_col_name].dt.dayofweek
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_avg = df_weekly.groupby(['day_num', 'day_of_week'])[sales_col_name].mean().reset_index()
            weekly_avg = weekly_avg.sort_values('day_num')

            fig_weekly = px.bar(weekly_avg, x='day_of_week', y=sales_col_name,
                                title='Average Sales by Day of Week',
                                labels={sales_col_name: 'Average Sales ($)', 'day_of_week': 'Day'},
                                color_discrete_sequence=['#ef553b'])
            fig_weekly.update_layout(height=300)
            st.plotly_chart(fig_weekly, use_container_width=True)

        # =============== DISTRIBUTION ANALYSIS ===============
        st.subheader(f"📊 Sales Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Histogram with KDE
            fig_hist = px.histogram(df_daily, x='y',
                                    title='Sales Distribution Histogram',
                                    labels={'y': 'Daily Sales ($)', 'count': 'Frequency'},
                                    nbins=30,
                                    marginal='box',
                                    opacity=0.7)
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Distribution statistics
            with st.expander("📈 Distribution Statistics"):
                skewness = stats.skew(df_daily['y'].dropna())
                kurtosis = stats.kurtosis(df_daily['y'].dropna())

                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Skewness", f"{skewness:.3f}")
                with stat_col2:
                    st.metric("Kurtosis", f"{kurtosis:.3f}")
                with stat_col3:
                    q1, q3 = np.percentile(df_daily['y'].dropna(), [25, 75])
                    iqr = q3 - q1
                    st.metric("IQR", f"${iqr:.2f}")

        with col2:
            # Box plot by month
            df_box = df_clean.copy()
            df_box['month'] = df_box[date_col_name].dt.strftime('%Y-%m')
            df_box = df_box.sort_values('month')

            fig_box = px.box(df_box, x='month', y=sales_col_name,
                             title='Monthly Sales Distribution',
                             labels={sales_col_name: 'Sales ($)', 'month': 'Month'},
                             color_discrete_sequence=['#00cc96'])
            fig_box.update_layout(
                height=350,
                xaxis_tickangle=45,
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Percentile analysis
            with st.expander("📊 Percentile Analysis"):
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(df_daily['y'].dropna(), percentiles)

                for p, val in zip(percentiles, percentile_values):
                    col_left, col_right = st.columns([1, 3])
                    with col_left:
                        st.write(f"{p}th:")
                    with col_right:
                        st.write(f"${val:,.2f}")

        # =============== OUTLIER DETECTION ===============
        st.subheader("🔍 Outlier Detection")

        # Calculate outliers using IQR method
        Q1 = df_daily['y'].quantile(0.25)
        Q3 = df_daily['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_daily[(df_daily['y'] < lower_bound) | (df_daily['y'] > upper_bound)]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lower Bound", f"${lower_bound:,.2f}")
        with col2:
            st.metric("Upper Bound", f"${upper_bound:,.2f}")
        with col3:
            st.metric("Outliers Found", len(outliers))

        if len(outliers) > 0:
            # Plot with outliers highlighted
            fig_outliers = go.Figure()

            # Normal points
            normal_points = df_daily[(df_daily['y'] >= lower_bound) & (df_daily['y'] <= upper_bound)]
            fig_outliers.add_trace(go.Scatter(
                x=normal_points['ds'],
                y=normal_points['y'],
                mode='markers',
                name='Normal Sales',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))

            # Outlier points
            fig_outliers.add_trace(go.Scatter(
                x=outliers['ds'],
                y=outliers['y'],
                mode='markers',
                name='Potential Outliers',
                marker=dict(color='red', size=10, symbol='diamond')
            ))

            # Add threshold lines
            fig_outliers.add_hline(y=lower_bound, line_dash="dash", line_color="orange",
                                   annotation_text="Lower Bound", annotation_position="bottom right")
            fig_outliers.add_hline(y=upper_bound, line_dash="dash", line_color="orange",
                                   annotation_text="Upper Bound", annotation_position="top right")

            fig_outliers.update_layout(
                title='Sales Data with Outliers Highlighted',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_outliers, use_container_width=True)

            # Show outlier table
            with st.expander("📋 View Outlier Details"):
                outlier_table = outliers.copy()
                outlier_table['ds'] = outlier_table['ds'].dt.strftime('%Y-%m-%d')
                outlier_table['y'] = outlier_table['y'].round(2)
                st.dataframe(outlier_table, use_container_width=True)
        else:
            st.success("✅ No outliers detected using the IQR method!")

        # =============== GROWTH ANALYSIS ===============
        st.subheader("📈 Growth Analysis")

        # Calculate month-over-month growth
        df_growth = df_clean.copy()
        df_growth['year_month'] = df_growth[date_col_name].dt.to_period('M')
        monthly_sales = df_growth.groupby('year_month')[sales_col_name].sum().reset_index()
        monthly_sales['year_month'] = monthly_sales['year_month'].astype(str)
        monthly_sales['growth'] = monthly_sales[sales_col_name].pct_change() * 100

        fig_growth = go.Figure()

        # Sales bars
        fig_growth.add_trace(go.Bar(
            x=monthly_sales['year_month'],
            y=monthly_sales[sales_col_name],
            name='Monthly Sales',
            marker_color='#636efa'
        ))

        # Growth line on secondary axis
        fig_growth.add_trace(go.Scatter(
            x=monthly_sales['year_month'],
            y=monthly_sales['growth'],
            name='Growth Rate %',
            yaxis='y2',
            line=dict(color='#ff7f0e', width=3)
        ))

        fig_growth.update_layout(
            title='Monthly Sales with Growth Rate',
            xaxis_title='Month',
            yaxis_title='Sales ($)',
            yaxis2=dict(
                title='Growth Rate (%)',
                overlaying='y',
                side='right'
            ),
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_growth, use_container_width=True)

        # =============== DATA QUALITY REPORT ===============
        st.subheader("📋 Data Quality Report")

        report_col1, report_col2, report_col3 = st.columns(3)

        with report_col1:
            st.metric("Date Range",
                      f"{df_daily['ds'].min().strftime('%Y-%m-%d')} to {df_daily['ds'].max().strftime('%Y-%m-%d')}")
            st.metric("Total Days", len(df_daily))
            st.metric("Missing Days",
                      (df_daily['ds'].max() - df_daily['ds'].min()).days + 1 - len(df_daily))

        with report_col2:
            st.metric("Min Daily Sales", f"${df_daily['y'].min():,.2f}")
            st.metric("Median Daily Sales", f"${df_daily['y'].median():,.2f}")
            st.metric("Max Daily Sales", f"${df_daily['y'].max():,.2f}")

        with report_col3:
            st.metric("Data Completeness",
                      f"{(1 - df_daily['y'].isnull().sum() / len(df_daily)) * 100:.1f}%")
            st.metric("Zero Sales Days",
                      len(df_daily[df_daily['y'] == 0]))
            st.metric("Best Month",
                      df_clean[date_col_name].dt.strftime('%Y-%m').value_counts().idxmax())

        # =============== INSIGHTS AND RECOMMENDATIONS ===============
        st.subheader("💡 Key Insights & Recommendations")

        insights = []

        # Check for seasonality
        monthly_std = monthly_avg[sales_col_name].std()
        monthly_mean = monthly_avg[sales_col_name].mean()
        if monthly_std > monthly_mean * 0.3:  # Strong seasonality
            best_month = monthly_avg.loc[monthly_avg[sales_col_name].idxmax(), 'month_name']
            worst_month = monthly_avg.loc[monthly_avg[sales_col_name].idxmin(), 'month_name']
            insights.append(f"**Strong seasonality detected**: Best month is {best_month}, worst is {worst_month}")

        # Check for trend
        if len(df_daily) > 30:
            recent_mean = df_daily.tail(30)['y'].mean()
            earlier_mean = df_daily.head(30)['y'].mean()
            trend_pct = ((recent_mean - earlier_mean) / earlier_mean) * 100 if earlier_mean > 0 else 0
            if abs(trend_pct) > 10:
                trend_dir = "increasing" if trend_pct > 0 else "decreasing"
                insights.append(
                    f"**Clear {trend_dir} trend**: Sales {trend_dir} by {abs(trend_pct):.1f}% comparing first and last 30 days")

        # Check for outliers
        if len(outliers) > 0:
            insights.append(f"**{len(outliers)} outliers detected**: Consider investigating these unusual sales days")

        # Check for variability
        if cv > 50:
            insights.append("**High sales variability**: Consider implementing more stable inventory management")
        elif cv < 10:
            insights.append("**Low sales variability**: Good for stable forecasting and inventory planning")

        # Display insights
        for insight in insights:
            st.info(insight)

        # Forecasting recommendations
        st.subheader("🎯 Forecasting Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("**Based on your data analysis:**")
            if len(df_daily) < 30:
                st.error(
                    "⚠️ **Insufficient data**: Collect more historical data (at least 30 days) for reliable forecasting")
            elif monthly_std > monthly_mean * 0.3:
                st.success(
                    "✅ **Prophet recommended**: Strong seasonality detected - Prophet handles seasonal patterns well")
            elif len(outliers) > len(df_daily) * 0.1:
                st.warning(
                    "⚠️ **Consider LSTM with preprocessing**: Many outliers detected - LSTM can handle irregularities")
            else:
                st.success("✅ **Both models viable**: Data shows moderate seasonality and good consistency")

        with rec_col2:
            st.markdown("**Next steps:**")
            st.write("1. Review the data quality report above")
            st.write("2. Note any patterns or anomalies")
            st.write("3. Select forecasting model based on recommendations")
            st.write("4. Configure forecast parameters")
            st.write("5. Generate and validate predictions")

    with tab2:
        # =============== FORECASTING SECTION ===============
        st.markdown('<div class="section-title">🔮 Sales Forecasting</div>', unsafe_allow_html=True)

        st.info("💡 Based on your data analysis, here are forecasting recommendations.")

        # Forecasting configuration
        col1, col2 = st.columns(2)

        with col1:
            model_choice = st.selectbox(
                t["model"],
                [t["prophet"], t["lstm"]],
                help="Select forecasting model based on data characteristics"
            )

            if model_choice == t["prophet"]:
                st.success("✅ Prophet selected - Best for seasonal patterns")
            else:
                st.info("📊 LSTM selected - Good for complex patterns")

        with col2:
            forecast_days = st.number_input(
                t["forecast_days"],
                min_value=1,
                max_value=365,
                value=max(1, min(30, len(df_daily) // 3)),  # <- ensure value >= 1
                help="Number of future days to forecast"
            )
            st.caption(f"Historical days: {len(df_daily)}, Forecast days: {forecast_days}")

        # Advanced settings expander
        with st.expander("⚙️ Advanced Settings"):
            if model_choice == t["prophet"]:
                seasonality_mode = st.selectbox(
                    "Seasonality Mode",
                    ["additive", "multiplicative"],
                    help="Additive: constant seasonality, Multiplicative: increasing seasonality"
                )
                changepoint_prior = st.slider(
                    "Flexibility",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.05,
                    step=0.001,
                    help="Higher values make the trend more flexible"
                )
            else:  # LSTM settings
                seq_length = st.slider(
                    "Sequence Length",
                    min_value=7,
                    max_value=min(90, len(df_daily) - 10),
                    value=min(30, len(df_daily) // 4),
                    help="Number of past days to consider for prediction"
                )
                lstm_units = st.slider(
                    "LSTM Units",
                    min_value=20,
                    max_value=200,
                    value=100,
                    step=20
                )

        if st.button(t["generate"], type="primary", use_container_width=True):
            with st.spinner("Training model and generating forecast..."):
                if model_choice == t["prophet"]:
                    # Prophet implementation
                    try:
                        m = Prophet(
                            yearly_seasonality=True if len(df_daily) > 365 else False,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            seasonality_mode=seasonality_mode,
                            changepoint_prior_scale=changepoint_prior
                        )
                        m.fit(df_daily)
                        future = m.make_future_dataframe(periods=forecast_days, freq='D')
                        forecast = m.predict(future)

                        # Store results
                        st.session_state.forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                        st.session_state.model_type = "prophet"

                        # Display results
                        st.success("✅ Forecast generated successfully!")

                        # Plot forecast
                        fig_forecast = go.Figure()

                        # Historical data
                        fig_forecast.add_trace(go.Scatter(
                            x=df_daily['ds'],
                            y=df_daily['y'],
                            mode='lines',
                            name='Historical Sales',
                            line=dict(color='blue', width=2)
                        ))

                        # Forecast
                        future_forecast = forecast[forecast['ds'] > df_daily['ds'].max()]
                        fig_forecast.add_trace(go.Scatter(
                            x=future_forecast['ds'],
                            y=future_forecast['yhat'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', width=3, dash='dash')
                        ))

                        # Confidence interval
                        fig_forecast.add_trace(go.Scatter(
                            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
                            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval'
                        ))

                        fig_forecast.update_layout(
                            title='Sales Forecast with Prophet',
                            xaxis_title='Date',
                            yaxis_title='Sales ($)',
                            height=500,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Error with Prophet: {str(e)}")

                else:
                    # LSTM implementation
                    try:
                        if len(df_daily) < seq_length + 10:
                            st.error(f"❌ Need at least {seq_length + 10} days of data for LSTM")
                        else:
                            # LSTM implementation code here
                            st.warning("⚠️ LSTM implementation requires more computation time...")
                            # (You can add the full LSTM implementation from previous code here)

                    except Exception as e:
                        st.error(f"❌ Error with LSTM: {str(e)}")

        # Export functionality
        if "forecast_df" in st.session_state:
            st.markdown("---")
            st.subheader("💾 Export Results")

            forecast_df = st.session_state.forecast_df

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Forecast Preview:**")
                preview_df = forecast_df.copy()
                preview_df['ds'] = preview_df['ds'].dt.strftime('%Y-%m-%d')
                st.dataframe(preview_df.head(10), use_container_width=True)

            with col2:
                # Convert to CSV
                csv = forecast_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download Forecast (CSV)",
                    data=csv,
                    file_name=f"sales_forecast_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Summary metrics
                if 'yhat' in forecast_df.columns:
                    st.metric("Average Forecast", f"${forecast_df['yhat'].mean():,.2f}")
                    st.metric("Total Forecast", f"${forecast_df['yhat'].sum():,.2f}")

# Footer
st.markdown("---")
st.caption("© 2025 Sales Analytics Dashboard | Developed by Almaarefa University Students CSIS Department")
