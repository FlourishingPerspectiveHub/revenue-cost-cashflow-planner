# ============================================================================
# revenue_cost_cashflow
# ============================================================================
# 
# Revenue, Cost and Cash Flow Planner
# Visualize cash flow, test different scenarios, and forecast business finances
#
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit.components.v1 as components

# ============================================================================
# ACCESS CONTROL
# ============================================================================
def check_password():
    """Returns True if user entered correct password."""
    
    def password_entered():
        if st.session_state["password"] == st.secrets.get("app_password", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("### 🔒 Access Required")
        st.info("This tool is currently in testing. Enter password to continue.")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.markdown("### 🔒 Access Required")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Incorrect password")
        st.stop()

# Uncomment to enable password protection:
# check_password()

# ============================================================================
# CONFIGURATION FLAGS
# ============================================================================
DETAILED_MODE = False  # Set to True to show formulas and projected values in export

# Warning System Configuration
SHOW_WARNING_INSTRUCTIONS = True
SHOW_WARNING_EXPORT = True

WARNING_INSTRUCTIONS="💡 **Educational tool** — Explore scenarios with simplified estimates. Always verify calculations and consult professionals before making financial decisions."
WARNING_EXPORT ="⚠️ **Reminder**: Educational estimates only. Best practice is to assume results may be inaccurate. Verify all calculations independently and consult qualified professionals before making financial decisions."

# ============================================================================
# DISPLAY LABELS (Easy to update in future)
# ============================================================================
OVERLAY_LABELS = {
    "linear_fit_cash": "Linear Trend Based",  # For cash balance chart
    "avg_fit_cash": "Historical Average Based",        # Not used - Historical Average has no cash balance overlay
    "linear_fit_rev": "Linear Fit Rev",  # For revenue/cost chart
    "linear_fit_cost": "Linear Fit Cost",
    "avg_fit_rev": "Avg Rev",
    "avg_fit_cost": "Avg Cost"
}

# Professional color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'neutral': '#7f7f7f',
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'highlight': '#e67e22'
}

# Forecast method descriptions
FORECAST_METHODS = {
    "Historical Average": "Calculates average revenue and costs from lookback period, then projects forward",
    "Linear Trend": "Fits a linear regression line to historical data to identify growth/decline trends",
    "Last Month Repeat": "Simply repeats the last month's revenue and costs for all forecast months",
    "Compound Growth": "Applies monthly growth percentage starting from last month's values"
}

# Inspirational quotes
INSPIRATION_QUOTES = [
    {
        "quote": "Money's greatest intrinsic value—and this can't be overstated—is its ability to give you control over your time.",
        "author": "Morgan Housel",
        "source": "The Psychology of Money",
        "link": "https://www.goodreads.com/quotes/10517294-money-s-greatest-intrinsic-value-and-this-can-t-be-overstated-is-its-ability"
    },
]

# --------------------------
# Demo Business Data
# --------------------------
DEMO_BUSINESSES = {
    "📦 Retail / Café / Food Biz": {
        "initial_cash": 20000,
        "display_description": "Seasonal product cash flow with off-season negative months.",
        "data": pd.DataFrame({
            'Month': [f"M{i:02d}" for i in range(1, 13)],
            'Revenue': [25000, 30000, 2000, 5000, 4000, 20000, 21000, 18000, 65000, 60000, 70000, 65000],
            'Total_Costs': [35000, 32000, 30000, 33000, 38000, 36000, 40000, 38000, 42000, 45000, 50000, 55000]
        })
    },
    "🏠 Freelancer / Coaching / Rental": {
        "initial_cash": 10000,
        "display_description": "Project-based cash flow with fluctuating monthly revenue.",
        "data": pd.DataFrame({
            'Month': [f"M{i:02d}" for i in range(1, 13)],
            'Revenue': [9000, 7500, 6000, 12000, 5000, 8000, 10000, 6000, 6000, 9000, 7000, 12000],
            'Total_Costs': [6000, 5500, 5500, 6500, 4500, 5000, 6000, 5500, 5500, 6000, 5500, 6500]
        })
    },
    "🔥 Startup SaaS": {
        "initial_cash": 50000,
        "display_description": "Growing subscription revenue with early burn months.",
        "data": pd.DataFrame({
            'Month': [f"M{i:02d}" for i in range(1, 13)],
            'Revenue': [2000, 3500, 5000, 7000, 9500, 12000, 15000, 18000, 22000, 27000, 33000, 40000],
            'Total_Costs': [15000, 14000, 13500, 13000, 12500, 12000, 12000, 12500, 13000, 13500, 14000, 14500]
        })
    }
}

SAMPLE_CSV = pd.DataFrame({
    'Month': ['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
    'Revenue': [50000, 55000, 52000, 58000, 60000, 62000],
    'Total_Costs': [35000, 36000, 35500, 37000, 38000, 39000]
})

# --------------------------
# Helper Functions
# --------------------------
def standardize_months(df):
    """Convert various month formats to standardized Month_Num (1-12+)"""
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    def parse_month(x):
        try:
            s = str(x).strip().lower()
            if s.startswith('m') and len(s) >= 2:
                return int(s[1:])
            return month_map.get(s[:3], None)
        except:
            return None
    
    df = df.copy()
    df["Month"] = df["Month"].astype(str)
    df["Month_Num"] = df["Month"].apply(parse_month)
    
    if df["Month_Num"].isna().all():
        df["Month_Num"] = list(range(1, len(df) + 1))
    else:
        if df["Month_Num"].isna().any():
            max_known = int(df["Month_Num"].dropna().max() or 0)
            missing_idx = df[df["Month_Num"].isna()].index
            for i, idx in enumerate(missing_idx, start=1):
                df.at[idx, "Month_Num"] = max_known + i
    
    df["Month_Num"] = df["Month_Num"].astype(int)
    return df

def compute_financials(df, init_cash, rev_mult=1.0, cost_mult=1.0, rev_add=0, cost_add=0):
    """Calculate net cash flow and running cash balance with optional adjustments"""
    df = df.copy()
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    
    if "Total_Costs" in df.columns:
        df["Total_Costs"] = pd.to_numeric(df["Total_Costs"], errors="coerce").fillna(0.0)
    elif "Fixed_Costs" in df.columns and "Variable_Costs" in df.columns:
        df["Fixed_Costs"] = pd.to_numeric(df["Fixed_Costs"], errors="coerce").fillna(0.0)
        df["Variable_Costs"] = pd.to_numeric(df["Variable_Costs"], errors="coerce").fillna(0.0)
        df["Total_Costs"] = df["Fixed_Costs"] + df["Variable_Costs"]
    else:
        df["Total_Costs"] = 0.0

    df["Revenue"] = ((df["Revenue"] * rev_mult) + rev_add).round(2)
    df["Total_Costs"] = ((df["Total_Costs"] * cost_mult) + cost_add).round(2)
    df["Net_Cash_Flow"] = (df["Revenue"] - df["Total_Costs"]).round(2)
    df["Cash_Balance"] = df["Net_Cash_Flow"].cumsum().round(2) + init_cash
    return df

def generate_forecast(df_hist, months, method, flat_rev=0, flat_cost=0, lookback=None, rev_growth=0, cost_growth=0):
    """Generate forward-looking forecast using specified method
    Now uses actual Month_Num values for linear trend fitting"""
    if months <= 0:
        return pd.DataFrame(), None
    
    if lookback:
        df_base = df_hist.tail(lookback).copy()
    else:
        df_base = df_hist.copy()

    last_idx = df_hist["Month_Num"].iloc[-1]
    last_cash = df_hist["Cash_Balance"].iloc[-1]
    rows = []
    formulas = None
    
    if method == "Historical Average":
        avg_rev = df_base["Revenue"].mean() if len(df_base) > 0 else 0
        avg_cost = df_base["Total_Costs"].mean() if len(df_base) > 0 else 0
        for i in range(1, months + 1):
            rev = round(avg_rev + flat_rev, 2)
            cost = round(avg_cost + flat_cost, 2)
            net = round(rev - cost, 2)
            last_cash = round(last_cash + net, 2)
            rows.append({"Month": f"F{i:02d}", "Month_Num": last_idx + i, "Revenue": rev, 
                        "Total_Costs": cost, "Net_Cash_Flow": net, "Cash_Balance": last_cash})
    
    elif method == "Linear Trend":
        if len(df_base) < 2:
            return generate_forecast(df_hist, months, "Historical Average", flat_rev, flat_cost, lookback, 0, 0)
        
        #  Use actual Month_Num values for fitting
        x = df_base["Month_Num"].values
        rev_trend = np.polyfit(x, df_base["Revenue"], 1)
        cost_trend = np.polyfit(x, df_base["Total_Costs"], 1)
        
        # Store formulas for display
        formulas = {
            "revenue": f"Revenue = {rev_trend[0]:.2f} × Month + {rev_trend[1]:.2f}",
            "cost": f"Cost = {cost_trend[0]:.2f} × Month + {cost_trend[1]:.2f}"
        }
        
        # Forecast uses next sequential month numbers (13, 14, 15...)
        for i in range(1, months + 1):
            future_month = last_idx + i
            rev = round(max(0, rev_trend[0] * future_month + rev_trend[1]) + flat_rev, 2)
            cost = round(max(0, cost_trend[0] * future_month + cost_trend[1]) + flat_cost, 2)
            net = round(rev - cost, 2)
            last_cash = round(last_cash + net, 2)
            rows.append({"Month": f"F{i:02d}", "Month_Num": future_month, "Revenue": rev,
                        "Total_Costs": cost, "Net_Cash_Flow": net, "Cash_Balance": last_cash})
    
    elif method == "Last Month Repeat":
        last_rev = round(df_hist["Revenue"].iloc[-1] + flat_rev, 2)
        last_cost = round(df_hist["Total_Costs"].iloc[-1] + flat_cost, 2)
        net = round(last_rev - last_cost, 2)
        for i in range(1, months + 1):
            last_cash += net
            rows.append({"Month": f"F{i:02d}", "Month_Num": last_idx + i, "Revenue": last_rev,
                        "Total_Costs": last_cost, "Net_Cash_Flow": net, "Cash_Balance": last_cash})
    
    elif method == "Compound Growth":
        last_rev = df_hist["Revenue"].iloc[-1]
        last_cost = df_hist["Total_Costs"].iloc[-1]
        for i in range(1, months + 1):
            last_rev = round(last_rev * (1 + rev_growth), 2)
            last_cost = round(last_cost * (1 + cost_growth), 2)
            rev = round(last_rev + flat_rev, 2)
            cost = round(last_cost + flat_cost, 2)
            net = round(rev - cost, 2)
            last_cash = round(last_cash + net, 2)
            rows.append({"Month": f"F{i:02d}", "Month_Num": last_idx + i, "Revenue": rev,
                        "Total_Costs": cost, "Net_Cash_Flow": net, "Cash_Balance": last_cash})
    
    return pd.DataFrame(rows), formulas

def generate_projected_historical(df_hist, method, lookback=None, init_cash=0):
    """Generate backward projection to show what the forecast method 'sees' in historical data
    NOTE: This shows PURE mathematical fit WITHOUT any flat_rev/flat_cost adjustments
    Now uses actual Month_Num values for linear trend fitting"""
    
    if lookback:
        df_base = df_hist.tail(lookback).copy()
    else:
        df_base = df_hist.copy()
    
    if method == "Historical Average":
        avg_rev = df_base["Revenue"].mean() if len(df_base) > 0 else 0
        avg_cost = df_base["Total_Costs"].mean() if len(df_base) > 0 else 0
        
        df_projected = df_base.copy()
        df_projected["Projected_Revenue"] = avg_rev
        df_projected["Projected_Cost"] = avg_cost
        
    elif method == "Linear Trend":
        if len(df_base) < 2:
            return None
        
        #  Use actual Month_Num values for fitting
        x = df_base["Month_Num"].values
        rev_trend = np.polyfit(x, df_base["Revenue"], 1)
        cost_trend = np.polyfit(x, df_base["Total_Costs"], 1)
        
        df_projected = df_base.copy()
        #  Apply formula using actual Month_Num values
        df_projected["Projected_Revenue"] = [round(rev_trend[0] * month_num + rev_trend[1], 2) 
                                              for month_num in df_base["Month_Num"]]
        df_projected["Projected_Cost"] = [round(cost_trend[0] * month_num + cost_trend[1], 2) 
                                           for month_num in df_base["Month_Num"]]
    else:
        return None  # Only for Historical Average and Linear Trend
    
    # Calculate projected cash flow and balance
    df_projected["Projected_Net_Flow"] = df_projected["Projected_Revenue"] - df_projected["Projected_Cost"]
    
    # Get starting cash balance at beginning of lookback period
    lookback_start_idx = len(df_hist) - len(df_base)
    if lookback_start_idx > 0:
        start_cash = df_hist.iloc[lookback_start_idx - 1]["Cash_Balance"]
    else:
        start_cash = init_cash
    
    df_projected["Projected_Cash_Balance"] = df_projected["Projected_Net_Flow"].cumsum() + start_cash
    
    return df_projected[["Month", "Projected_Revenue", "Projected_Cost", "Projected_Net_Flow", "Projected_Cash_Balance"]]
    

def calculate_runway(df):
    """Calculate months until cash balance goes negative"""
    neg = df[df["Cash_Balance"] <= 0]
    return neg.index[0] + 1 if len(neg) > 0 else None

# --------------------------
# Session State Initialization
# --------------------------
if 'initialized' not in st.session_state:
    st.session_state.selected_demo = list(DEMO_BUSINESSES.keys())[0]
    st.session_state.initialized = True

for key in ['toggle_optimistic', 'toggle_conservative', 'toggle_custom', 'show_forecast', 
            'manual_edited_df', 'edit_mode_data', 'custom_edited_df', 'custom_mode']:
    if key not in st.session_state:
        if 'toggle' in key or key == 'show_forecast':
            st.session_state[key] = False
        elif key == 'custom_mode':
            st.session_state[key] = "adjustments"
        else:
            st.session_state[key] = None

if 'show_forecast' not in st.session_state:
    st.session_state.show_forecast = True

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Revenue, Cost and Cash Flow Planner",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# SEO Meta Tags Injection
components.html("""
<head>
    <meta name="description" content="Free revenue, cost and cash flow planner for small businesses. Visualize scenarios, forecast finances, and gain planning insights. Educational tool - verify all calculations before decisions.">
    <meta name="keywords" content="cash flow planner, revenue forecasting, business finance, scenario planning, startup runway">
    <meta name="author" content="Revenue, Cost and Cash Flow Planner">
    <meta property="og:title" content="Free Revenue, Cost and Cash Flow Planner - Business Scenario Planning Tool">
    <meta property="og:description" content="Educational planning tool to visualize cash flow and test business scenarios. Free forever - verify calculations before decisions.">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Revenue, Cost and Cash Flow Planner - Test Business Scenarios">
    <meta name="twitter:description" content="Free educational tool for exploring cash flow scenarios and forecasting. Always verify with professionals.">
</head>
""", height=0)

# Professional styling with minimized top space
st.markdown("""
<style>
    /* Minimize top spacing */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem;
    }

    /* Control sidebar width */
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 500px;
    }
    
    .header-container {
        margin-top: -50px !important;
        margin-bottom: 0px;
        padding: 0;
    }
    
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #1a5490;
        letter-spacing: -0.5px;
        margin: 0 0 2px 0;
        line-height: 1.1;
    }
    
    .tagline {
        font-size: 24px;
        color: #7f8c8d;
        margin: 0 0 10px 0;
        line-height: 1.3;
    }
    
    .intro-text {
        margin-top: -5px;
        margin-bottom: 6px;
        font-size: 20px;
    }
    
    .metric-container {background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #3498db;}
    .section-header {font-size: 16px; font-weight: 700; color: #34495e; text-transform: uppercase; 
                     letter-spacing: 0.5px; margin-top: 15px; margin-bottom: 4px;}
    
    .resource-link {
        background: #f8f9fa;
        border-left: 3px solid #3498db;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 3px;
        font-size: 14px;
    }
    
    /* Hide download button and sort options in dataframe */
    button[title="Download"] {display: none !important;}
    button[title="Download data as CSV"] {display: none !important;}
    .stDataFrame button[kind="header"] {display: none !important;}
    
    .based-on-label {
        color: #7f8c8d;
        font-weight: 400;
    }
    .based-on-value {
        color: #e67e22;
        font-weight: 600;
    }
    
    /* Instructions Box */
    .instructions-box {
        background: #e3f2fd;
        border: 2px solid #3498db;
        border-radius: 8px;
        padding: 12px 15px 12px 40px;  
        margin: 5px 0 5px 0;
        font-size: 14px;
        line-height: 1.5;
        color: #1565c0;
        position: relative;
    }
    
    .instructions-box::before {
        content: "👈";
        font-size: 20px;
        position: absolute;
        left: 8px;
        top: 50%;
        transform: translateY(-50%);
    }
    
    /* Warning Box */
    .warning-box {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 5px 0 20px 0;
        font-size: 14px;
        line-height: 1.5;
        color: #1565c0;
        font-weight: 600;
        text-align: center;
    }
    
    /* Debug formula box */
    .formula-box {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Consolidate header, intro, and warnings into single markdown block
header_html = """
<div class="header-container">
    <div class="main-title">💰 Revenue, Cost and Cash Flow Planner</div>
    <div class="tagline">Visualize scenarios and gain insights for your business planning</div>
</div>
<div class="intro-text">
Test different scenarios and explore projections for <strong>Retail, E-commerce, SaaS, Freelancers, Rentals, and more<strong>.
</div>
"""



# Add instructions box
header_html += """
<div class="instructions-box" style="margin-top: 8px; margin-bottom: 5px;">
    Choose a demo or input your data  →  Adjust scenarios  →  Add forecast  →  Analyze charts  →  Export results
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)

# Add Warning 2 if enabled
if SHOW_WARNING_INSTRUCTIONS:
    st.info(WARNING_INSTRUCTIONS)
# --------------------------
# Sidebar - Data Input & Controls
# --------------------------
with st.sidebar:
    st.markdown('<p class="section-header">Data Source</p>', unsafe_allow_html=True)
    
    source_tab = st.radio("Select data source", ["Demo", "Input Your Data"], label_visibility="collapsed", horizontal=True)
    
    df_in = None
    initial_cash = 20000
    
    if source_tab == "Demo":
        selected_demo = st.selectbox("Select Demo Business", list(DEMO_BUSINESSES.keys()), 
                                     index=list(DEMO_BUSINESSES.keys()).index(st.session_state.selected_demo),
                                     label_visibility="collapsed")
        st.session_state.selected_demo = selected_demo
        
        demo_data = DEMO_BUSINESSES[selected_demo]
        st.caption(demo_data['display_description'])
        df_in = demo_data['data'].copy()
        initial_cash = demo_data['initial_cash']
    
    else:  # Input Your Data


        input_method = st.radio("Input method", ["Paste Data from Spreadsheet"], 
                       horizontal=False, label_visibility="collapsed",
                       help="Choose how to enter your data")
        
        if input_method == "Paste Data from Spreadsheet":
            if st.session_state.edit_mode_data is None:
                st.session_state.edit_mode_data = pd.DataFrame({
                    'Month': list(range(1, 37)),
                    'Revenue': [50000] * 6 + [None] * 30,
                    'Total_Costs': [35000] * 6 + [None] * 30
                })
            
            col_tip, col_btn = st.columns([3, 1])
            with col_tip:
                st.caption("💡 **Tip:** Copy cells from spreadsheet and paste directly into the table below")
            with col_btn:
                if st.button("🗑️ Clear", key="clear_table", help="Clear all revenue and cost data"):
                    st.session_state.edit_mode_data = pd.DataFrame({
                        'Month': list(range(1, 37)),
                        'Revenue': [None] * 36,
                        'Total_Costs': [None] * 36
                    })
                    st.rerun()
            
            edited = st.data_editor(
                st.session_state.edit_mode_data, 
                num_rows="fixed", 
                width='stretch',
                hide_index=True, 
                height=250,
                column_config={
                    "Month": st.column_config.NumberColumn("Mo", disabled=True, format="%d"),
                    "Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                    "Total_Costs": st.column_config.NumberColumn("Costs", format="$%.2f")
                }
            )
            st.session_state.edit_mode_data = edited
            
            df_in = edited.dropna(subset=['Revenue', 'Total_Costs']).copy()
            if len(df_in) > 0:
                df_in['Month'] = df_in['Month'].apply(lambda x: f"M{int(x):02d}")
            else:
                df_in = None
        

    
    # Starting cash
    if df_in is not None:
        st.markdown('<p class="section-header" style="margin-top:15px;">Starting Cash</p>', unsafe_allow_html=True)
        initial_cash = st.number_input("Starting cash amount", value=initial_cash, step=5000, format="%d", label_visibility="collapsed")
    
    st.markdown("---")
    
    # Scenario controls
    if df_in is not None:
        st.markdown('<p class="section-header">Scenarios</p>', unsafe_allow_html=True)
        st.caption("💡 Modify current data to view alternative outcomes for the same time period")
        st.checkbox("Status Quo", value=True, disabled=True, key="sq", help="Original data without modifications")
        st.session_state.toggle_optimistic = st.checkbox("Optimistic", st.session_state.toggle_optimistic, key="opt",
                                                         help="Revenue +30%, Costs -10%")
        st.session_state.toggle_conservative = st.checkbox("Conservative", st.session_state.toggle_conservative, key="cons",
                                                           help="Revenue -10%, Costs +10%")
        st.session_state.toggle_custom = st.checkbox("Custom", st.session_state.toggle_custom, key="cust", 
                                                      help="Create custom scenario with adjustments or table editing")
        
        # Custom scenario settings
        if st.session_state.toggle_custom:
            with st.expander("⚙️ Custom Settings", expanded=False):
                st.session_state.custom_mode = st.radio(
                    "Custom Mode",
                    ["adjustments", "table_editor"],
                    format_func=lambda x: "% & $ Adjustments" if x == "adjustments" else "Table Editor",
                    help="Choose one method: Apply percentage and fixed adjustments, OR edit table directly"
                )
                
                if st.session_state.custom_mode == "adjustments":
                    st.info("ℹ️ Adjustments: **(Original × %) + Fixed $/month**")
                    
                    st.caption("**% Adjustment**")
                    rev_pct = st.number_input("Revenue %", -200.0, 200.0, 0.0, 5.0, key="custom_rev_pct",
                                              help="Percentage change (e.g., 10 = +10%, -10 = -10%)")
                    cost_pct = st.number_input("Cost %", -200.0, 200.0, 0.0, 5.0, key="custom_cost_pct",
                                               help="Percentage change (e.g., 10 = +10%, -10 = -10%)")
                    
                    st.caption("**Fixed $/month**")
                    rev_add = st.number_input("Revenue $", -500000, 500000, 0, 500, key="custom_rev_add",
                                              help="Fixed dollar amount added after % adjustment")
                    cost_add = st.number_input("Cost $", -500000, 500000, 0, 500, key="custom_cost_add",
                                               help="Fixed dollar amount added after % adjustment")
                
                else:  # table_editor mode
                    st.caption("**Edit values directly** (overrides adjustments)")
                    if st.session_state.custom_edited_df is None:
                        st.session_state.custom_edited_df = df_in.copy()
                    
                    custom_edited = st.data_editor(
                        st.session_state.custom_edited_df,
                        width='stretch',
                        hide_index=True,
                        height=250,
                        column_config={
                            "Month": st.column_config.TextColumn("Month", disabled=True),
                            "Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                            "Total_Costs": st.column_config.NumberColumn("Costs", format="$%.2f")
                        }
                    )
                    st.session_state.custom_edited_df = custom_edited
        
        st.markdown('<p class="section-header">Forecast</p>', unsafe_allow_html=True)
        st.caption("💡 Project future months using your selected method")
        st.session_state.show_forecast = st.checkbox("Show Forecast", st.session_state.show_forecast, key="fc_show")
        
        if st.session_state.show_forecast:
            forecast_months = st.number_input("Months to forecast", 1, 48, 6, 1)
            
            with st.expander("⚙️ Forecast Settings", expanded=False):
                st.caption("**Method**")
                forecast_method = st.selectbox("Forecast method", ["Linear Trend", "Historical Average", 
                                                    "Compound Growth", "Last Month Repeat"],
                                              label_visibility="collapsed")
                st.info(f"ℹ️ {FORECAST_METHODS[forecast_method]}")
                
                if forecast_method == "Compound Growth":
                    st.caption("**Growth %/month** *(from last month)*")
                    rev_growth = st.number_input("Revenue", -20.0, 30.0, 0.0, 0.5, format="%.1f") / 100.0
                    cost_growth = st.number_input("Cost", -20.0, 30.0, 0.0, 0.5, format="%.1f") / 100.0
                elif forecast_method in ["Historical Average", "Linear Trend"]:
                    st.info("💡 Tip: Lower lookback emphasizes recent trends")
                    hist_count_temp = len(df_in)
                    lookback = st.number_input(f"Lookback months (max {hist_count_temp})", 
                                              1, hist_count_temp, hist_count_temp, 1)
                    
                

                st.info("ℹ️  Forecast: **Method calculation, then Adjustment ($/month) applied**")

                flat_rev = st.number_input("Revenue Adjustment ($/month)", -500000, 500000, 0, 500, key="fc_flat_rev")
                flat_cost = st.number_input("Cost Adjustment ($/month)", -500000, 500000, 0, 500, key="fc_flat_cost")
        else:
            forecast_months = 0
            forecast_method = "Linear Trend"
            flat_rev = 0
            flat_cost = 0
            lookback = None
            rev_growth = 0
            cost_growth = 0
    
    st.markdown("---")
    
    # Resources
    st.markdown('<p class="section-header">Resources</p>', unsafe_allow_html=True)
    st.markdown("""
   
    <div class="resource-link">
        💼 <a href="https://www.sba.gov/business-guide/plan-your-business/calculate-your-startup-costs" target="_blank" rel="noopener">SBA: Calculate Startup Costs</a>
    </div>
                
    <div class="resource-link">
        🏫 <a href="https://www.score.org" target="_blank" rel="noopener">
            SCORE Mentorship
        </a>
    </div>
    <div class="resource-link">
        📘 <a href="https://www.the-founders-corner.com/p/a-guide-to-small-business-cash-flow" target="_blank" rel="noopener">
        The Founder's Corner: Guide to Cash Flow
        </a>
    </div>                              

   
                
    """, unsafe_allow_html=True)
    
    
    st.markdown("---")

# --------------------------
# Main Content - Calculations
# --------------------------
if df_in is None:
    st.info("👈 Choose a demo or input your data")
    st.stop()

try:
    df_in.columns = [c.strip() for c in df_in.columns]
    df0 = standardize_months(df_in)
    
    # Calculate all scenarios
    df_status_quo = compute_financials(df0.copy(), initial_cash)
    df_optimistic = compute_financials(df0.copy(), initial_cash, rev_mult=1.3, cost_mult=0.9)
    df_conservative = compute_financials(df0.copy(), initial_cash, rev_mult=0.9, cost_mult=1.1)
    
    # Custom scenario
    if st.session_state.toggle_custom:
        if st.session_state.custom_mode == "table_editor" and st.session_state.custom_edited_df is not None:
            df_custom_base = standardize_months(st.session_state.custom_edited_df)
            df_custom = compute_financials(df_custom_base, initial_cash)
        else:
            rev_mult = 1.0 + (st.session_state.get('custom_rev_pct', 0.0) / 100.0)
            cost_mult = 1.0 + (st.session_state.get('custom_cost_pct', 0.0) / 100.0)
            rev_add = st.session_state.get('custom_rev_add', 0)
            cost_add = st.session_state.get('custom_cost_add', 0)
            df_custom = compute_financials(df0.copy(), initial_cash, rev_mult, cost_mult, rev_add, cost_add)
    else:
        df_custom = df_status_quo.copy()
    
    SCENARIO_MAP = {
        "Status Quo": {"df": df_status_quo, "color": COLORS['neutral']},
        "Optimistic": {"df": df_optimistic, "color": COLORS['success']},
        "Conservative": {"df": df_conservative, "color": COLORS['warning']},
        "Custom": {"df": df_custom, "color": COLORS['primary']},
    }
    
    hist_count = len(df_status_quo)

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# Determine forecast base
forecast_base = "Custom" if st.session_state.toggle_custom else "Status Quo"
df_base = SCENARIO_MAP[forecast_base]["df"].copy()

# Generate forecast (updated to return formulas)
df_forecast = pd.DataFrame()
forecast_formulas = None
if st.session_state.show_forecast and forecast_months > 0:
    df_forecast, forecast_formulas = generate_forecast(
        df_base, forecast_months, forecast_method, flat_rev, flat_cost, 
        lookback if forecast_method in ["Historical Average", "Linear Trend"] else None, 
        rev_growth if forecast_method == "Compound Growth" else 0, 
        cost_growth if forecast_method == "Compound Growth" else 0
    )

df_combined = pd.concat([df_base, df_forecast], ignore_index=True) if not df_forecast.empty else df_base

# Generate projected historical values (for detailed mode)
df_projected_hist = None
if DETAILED_MODE and st.session_state.show_forecast and forecast_method in ["Historical Average", "Linear Trend"]:
    df_projected_hist = generate_projected_historical(
        df_base, forecast_method,
        lookback if 'lookback' in locals() else None,
        initial_cash
    )

# Historical analysis for trend overlay on Cash Balance chart
hist_analysis = {}
if st.session_state.show_forecast and forecast_method == "Linear Trend" and 'lookback' in locals():
    # ✓ CHANGED: Only for Linear Trend, not Historical Average
    df_lookback = df_base.tail(lookback).copy()
    if len(df_lookback) >= 2:
        # ✓ Use actual Month_Num values
        x = df_lookback["Month_Num"].values
        flow_trend = np.polyfit(x, df_lookback["Net_Cash_Flow"], 1)
        start_bal = df_base["Cash_Balance"].iloc[hist_count - lookback] - df_base["Net_Cash_Flow"].iloc[hist_count - lookback]
        # Apply fitted values to actual month numbers
        hist_analysis = {
            "name": f"{OVERLAY_LABELS['linear_fit_cash']} ({lookback}mo)",
            "y": start_bal + np.cumsum([flow_trend[0] * month + flow_trend[1] for month in x]),
            "x": df_lookback["Month_Num"].tolist(),
            "color": "teal", "dash": "dash"
        }

# Prepare overlay lines for Revenue & Costs chart
revenue_cost_overlays = {}
if st.session_state.show_forecast and forecast_method in ["Historical Average", "Linear Trend"] and 'lookback' in locals():
    df_lookback = df_base.tail(lookback).copy()
    
    if forecast_method == "Historical Average" and len(df_lookback) >= 1:
        avg_rev = df_lookback["Revenue"].mean()
        avg_cost = df_lookback["Total_Costs"].mean()
        
        revenue_cost_overlays = {
            "revenue": {
                "name": f"{OVERLAY_LABELS['avg_fit_rev']} ({lookback}mo)",
                "y": [avg_rev] * len(df_lookback),
                "x": df_lookback["Month_Num"].tolist(),
                "color": "orange",
                "dash": "dot"
            },
            "cost": {
                "name": f"{OVERLAY_LABELS['avg_fit_cost']} ({lookback}mo)",
                "y": [-avg_cost] * len(df_lookback),  # Negative for display
                "x": df_lookback["Month_Num"].tolist(),
                "color": "#d62728",  # Red
                "dash": "dot"
            }
        }
    
    elif forecast_method == "Linear Trend" and len(df_lookback) >= 2:
        # Use actual Month_Num values for fitting
        x = df_lookback["Month_Num"].values
        rev_trend = np.polyfit(x, df_lookback["Revenue"], 1)
        cost_trend = np.polyfit(x, df_lookback["Total_Costs"], 1)
        
        revenue_cost_overlays = {
            "revenue": {
                "name": f"{OVERLAY_LABELS['linear_fit_rev']} ({lookback}mo)",
                # Apply formula using actual month numbers
                "y": [rev_trend[0] * month + rev_trend[1] for month in x],
                "x": df_lookback["Month_Num"].tolist(),
                "color": "orange",
                "dash": "dash"
            },
            "cost": {
                "name": f"{OVERLAY_LABELS['linear_fit_cost']} ({lookback}mo)",
                # Apply formula using actual month numbers
                "y": [-(cost_trend[0] * month + cost_trend[1]) for month in x],  # Negative
                "x": df_lookback["Month_Num"].tolist(),
                "color": "#d62728",  # Red
                "dash": "dash"
            }
        }

# --------------------------
# Main Content - Metrics Display
# --------------------------
forecast_suffix = " + Forecast" if st.session_state.show_forecast and not df_forecast.empty else ""
st.markdown(f'<p class="section-header" style="margin-top: -10px;">Key Metrics</p>', unsafe_allow_html=True)
st.markdown(f'📊 <span class="based-on-label">Based on:</span> <span class="based-on-value">{forecast_base}{forecast_suffix}</span>', unsafe_allow_html=True)

runway = calculate_runway(df_combined)
final_bal = df_combined["Cash_Balance"].iloc[-1]
burn = df_base[df_base["Net_Cash_Flow"] < 0]["Net_Cash_Flow"].mean()

# Highlight negative cash flow months
neg_months = df_base[df_base["Net_Cash_Flow"] < 0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    if runway:
        st.metric("Runway", f"{runway}mo", help="Number of months until cash balance reaches zero or below")
    else:
        st.markdown('<div style="padding: 10px; background: #e8f5e9; border-radius: 5px; text-align: center;"><div style="color: #2e7d32; font-size: 12px; font-weight: 600;">RUNWAY</div><div style="color: #2e7d32; font-size: 24px; font-weight: 600;">Positive ✓</div></div>', unsafe_allow_html=True)
        st.caption("ℹ️ Cash balance stays positive throughout the period")
with c2:
    st.metric("Burn Rate", f"${-burn:,.0f}/mo" if not np.isnan(burn) and burn < 0 else "No Burn", 
              help="Average monthly cash spent when net cash flow is negative")
with c3:
    st.metric("End Balance", f"${final_bal:,.0f}", 
              help="Projected cash balance at the end of the period")
with c4:
    if len(neg_months) > 0:
        st.metric("⚠️ Neg Months", len(neg_months), 
                  help=f"Months with negative cash flow: {', '.join(neg_months['Month'].tolist())}")

# --------------------------
# Charts
# --------------------------

st.markdown(f'<p class="section-header" style="margin-top: -10px; margin-bottom: 5px;">Financial Analysis</p>', unsafe_allow_html=True)



st.markdown(f'<p style="margin-bottom: 5px;">📊 <span class="based-on-label">Based on:</span> <span class="based-on-value">{forecast_base}{forecast_suffix}</span></p>', unsafe_allow_html=True)

scenarios_to_plot = {k: v for k, v in SCENARIO_MAP.items() 
                     if k == "Status Quo" or 
                     (k == "Optimistic" and st.session_state.toggle_optimistic) or
                     (k == "Conservative" and st.session_state.toggle_conservative) or
                     (k == "Custom" and st.session_state.toggle_custom)}

fig_cash = go.Figure()

# Add forecast shaded region with annotation
if not df_forecast.empty:
    fig_cash.add_vrect(x0=hist_count + 0.5, x1=df_combined["Month_Num"].max() + 1,
                       fillcolor="lightgray", opacity=0.08, layer="below", line_width=0)
    
    # Add annotation for forecast region (top-right)
    fig_cash.add_annotation(
        text=f"Forecast: {forecast_method}",
        showarrow=False,
        font=dict(size=14, color="gray"),
        xref="paper", yref="paper",
        x=0.98, y=1.1,
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )

for name, data in scenarios_to_plot.items():
    df_plot = data["df"]
    color = data["color"]
    
    neg_mask = df_plot["Net_Cash_Flow"] < 0
    
    if name == forecast_base and not df_forecast.empty:
        fig_cash.add_trace(go.Scatter(
            x=df_base["Month_Num"], y=df_base["Cash_Balance"],
            mode='lines+markers', name=name,
            line=dict(color=color, width=2.5), marker=dict(size=6)
        ))
        df_fore = df_combined.iloc[hist_count-1:].copy()
        fig_cash.add_trace(go.Scatter(
            x=df_fore["Month_Num"], y=df_fore["Cash_Balance"],
            mode='lines+markers', name=f"{name} (Fcst)",
            line=dict(color=color, dash='dash', width=2), 
            marker=dict(size=5),
            showlegend=False
        ))        
    else:
        fig_cash.add_trace(go.Scatter(
            x=df_plot["Month_Num"], y=df_plot["Cash_Balance"],
            mode='lines+markers', name=name,
            line=dict(color=color, width=2), marker=dict(size=5)
        ))
    

if hist_analysis:
    fig_cash.add_trace(go.Scatter(
        x=hist_analysis["x"], y=hist_analysis["y"],
        mode="lines", name=hist_analysis["name"],
        line=dict(color=hist_analysis["color"], dash=hist_analysis["dash"], width=2),
        opacity=0.6
    ))

fig_cash.add_hline(y=0, line_dash="dash", line_color=COLORS['danger'], opacity=0.3)

all_months = df_combined[["Month", "Month_Num"]].drop_duplicates()
fig_cash.update_layout(
    font=dict(size=18, family="Arial, sans-serif"),
    title={
        'text': '<span style="font-size: 24px; font-weight: bold;">Cash Balance</span><span style="font-size: 24px; font-weight: normal;">: Cumulative cash position over time</span>',
        'x': 0,
        'xanchor': 'left'
    },
    xaxis_title='', yaxis_title='Cash Balance ($)',
    legend=dict(orientation="h", y=1.12, x=0),
    hovermode="x unified", 
    hoverlabel=dict(namelength=-1),
    height=300,
    yaxis=dict(tickformat="$,.0f"),
    xaxis=dict(tickvals=all_months["Month_Num"].tolist(), ticktext=all_months["Month"].tolist()),
    margin=dict(t=50, b=30, l=50, r=20),
    plot_bgcolor='rgba(248,249,250,0.5)'
)

st.plotly_chart(fig_cash, use_container_width=True)

# Revenue vs Costs
fig_bars = go.Figure()

fig_bars.add_trace(go.Bar(x=df_base["Month_Num"], y=df_base["Revenue"],
                          name='Revenue', marker_color=COLORS['success'], opacity=0.9, base=0))
fig_bars.add_trace(go.Bar(x=df_base["Month_Num"], y=-df_base["Total_Costs"],
                          name='Costs', marker_color=COLORS['danger'], opacity=0.9, base=0))

if not df_forecast.empty:
    fig_bars.add_trace(go.Bar(x=df_forecast["Month_Num"], y=df_forecast["Revenue"],
                              name='Rev (Fcst)', marker_color=COLORS['success'], opacity=0.5, base=0))
    fig_bars.add_trace(go.Bar(x=df_forecast["Month_Num"], y=-df_forecast["Total_Costs"],
                              name='Cost (Fcst)', marker_color=COLORS['danger'], opacity=0.5, base=0))

# Net Flow line with markers
fig_bars.add_trace(go.Scatter(x=df_combined["Month_Num"], y=df_combined["Net_Cash_Flow"],
                              mode='lines+markers', name='Net Flow',
                              line=dict(color=COLORS['primary'], width=2),
                              marker=dict(size=5)))

# Add overlay lines for Revenue & Costs (if available)
if revenue_cost_overlays:
    fig_bars.add_trace(go.Scatter(
        x=revenue_cost_overlays["revenue"]["x"],
        y=revenue_cost_overlays["revenue"]["y"],
        mode="lines",
        name=revenue_cost_overlays["revenue"]["name"],
        line=dict(color=revenue_cost_overlays["revenue"]["color"], 
                  dash=revenue_cost_overlays["revenue"]["dash"], width=2),
        opacity=0.7
    ))
    
    fig_bars.add_trace(go.Scatter(
        x=revenue_cost_overlays["cost"]["x"],
        y=revenue_cost_overlays["cost"]["y"],
        mode="lines",
        name=revenue_cost_overlays["cost"]["name"],
        line=dict(color=revenue_cost_overlays["cost"]["color"], 
                  dash=revenue_cost_overlays["cost"]["dash"], width=2),
        opacity=0.7
    ))

fig_bars.update_layout(
    font=dict(size=11, family="Arial, sans-serif"),
    title={
        'text': '<span style="font-size: 24px; font-weight: bold;">Revenue & Costs</span><span style="font-size: 24px; font-weight: normal;">:  Monthly revenue, costs, and net cash flow</span>',
        'x': 0,
        'xanchor': 'left'
    },
    
    legend=dict(orientation="h", y=1.12, x=0),
    hovermode="x unified", height=280, barmode='overlay',
    xaxis=dict(title='', tickvals=all_months["Month_Num"].tolist(), ticktext=all_months["Month"].tolist()),
    yaxis=dict(title='Amount ($)', tickformat="$,.0f"),
    margin=dict(t=50, b=30, l=50, r=20),
    plot_bgcolor='rgba(248,249,250,0.5)'
)


st.plotly_chart(fig_bars, use_container_width=True)
st.markdown('<p style="margin-top: -40px; font-size: 14px; color: gray;">⬆️ 💡 Chart note: Costs shown as negative for visual contrast with revenue</p>', unsafe_allow_html=True)

# --------------------------
# DETAILED MODE: Display Formulas
# --------------------------
if DETAILED_MODE and forecast_formulas:
    st.markdown("""
    <div class="formula-box">
        <strong>🔍 DETAILED MODE: Linear Trend Formulas</strong><br>
        <code>{}</code><br>
        <code>{}</code>
    </div>
    """.format(forecast_formulas["revenue"], forecast_formulas["cost"]), unsafe_allow_html=True)
elif DETAILED_MODE and st.session_state.show_forecast and forecast_method == "Historical Average" and 'lookback' in locals():
    df_lookback = df_base.tail(lookback).copy()
    avg_rev = df_lookback["Revenue"].mean()
    avg_cost = df_lookback["Total_Costs"].mean()
    st.markdown("""
    <div class="formula-box">
        <strong>🔍 DETAILED MODE: Historical Average Values</strong><br>
        <code>Average Revenue (last {} months) = ${:,.2f}</code><br>
        <code>Average Cost (last {} months) = ${:,.2f}</code>
    </div>
    """.format(lookback, avg_rev, lookback, avg_cost), unsafe_allow_html=True)

# --------------------------
# Inspiration Quote
# --------------------------
import random
selected_quote = random.choice(INSPIRATION_QUOTES)

st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 8px; margin: 20px 0; color: white; text-align: center;">
    <p style="font-size: 18px; font-style: italic; margin: 0 0 10px 0; line-height: 1.6;">"{selected_quote['quote']}"</p>
    <p style="margin: 0; font-size: 14px; opacity: 0.9;">— <strong>{selected_quote['author']}</strong>, <a href="{selected_quote['link']}" target="_blank" rel="noopener" style="color: white; text-decoration: underline;">{selected_quote['source']}</a></p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Helpful Reads & Resources
# --------------------------
st.markdown('<p class="section-header">🧭 EXPLORE MORE</p>', unsafe_allow_html=True)

st.markdown("""
**⚙️ Tools & Skills**
- [Strategies to improve cash flow for freelancers](https://ruul.io/blog/strategies-to-improve-cash-flow-for-freelancers) (Source: Blog from Ruul)
- [Budgeting for Freelancers: Effective Strategies for Variable Income](https://m1.com/knowledge-bank/budgeting-for-freelancers-effective-strategies-for-variable-income/) (Source: Knowledge Bank from M1 Finance)
- [How Finance Works: The HBR Guide to Thinking Smart About the Numbers](https://a.co/d/hQDOIl3) - Practical guide to finance, valuation, and risk by Mihir Desai
- [TinyWow](https://tinywow.com/) - Tools for PDFs, images, writing, and videos
- [LearningSEO.io](https://learningseo.io/) - SEO roadmap with free resources & tools

**🚀 Learn & Grow**
- [Founders Content](https://foundercontent.com/all/) - Curated list of resources from founders
- [The Side Hustle Show Podcast](https://www.sidehustlenation.com/side-hustle-show/) - Nick Loper's podcast on starting and growing side hustles
- [Guy Raz](https://www.guyraz.com/) - Host of “How I Built This” & other podcasts, sharing insights and lessons from entrepreneurs and innovators
- [Odd Lots](https://www.bloomberg.com/oddlots) - Bloomberg podcast exploring finance, markets, and economic stories with hosts Joe Weisenthal and Tracy Alloway

**🌍 Inspiration & Impact**
- [kottke.org](https://kottke.org/) - Long-running site (founded 1998) featuring Jason Kottke's curated posts on art, technology, science, design, and culture
- [Unbound](https://www.unbound.org/) - Nonprofit supporting families and communities on their self-directed paths out of poverty
- [Positive News](https://www.positive.news/) - Constructive journalism focused on positive global news
- [Our World in Data](https://ourworldindata.org/) - Data insights and visualizations on major global issues
                        
""")
# --------------------------
# Export Data (Copy/Paste) - ACTIVE SCENARIOS
# --------------------------
st.markdown('<p class="section-header">Export Data</p>', unsafe_allow_html=True)



st.info("💡 Download CSV or copy the table (Ctrl+C or Cmd+C) for use in spreadsheet applications")

# Prepare export dataframe with ACTIVE scenarios only
df_all_scenarios = pd.DataFrame()

# Always include Status Quo
df_scenario = SCENARIO_MAP["Status Quo"]["df"][["Month", "Revenue", "Total_Costs", "Net_Cash_Flow", "Cash_Balance"]].copy()
df_scenario["Scenario"] = "Status Quo"
df_all_scenarios = pd.concat([df_all_scenarios, df_scenario], ignore_index=True)

# Add Optimistic if toggled
if st.session_state.toggle_optimistic:
    df_scenario = SCENARIO_MAP["Optimistic"]["df"][["Month", "Revenue", "Total_Costs", "Net_Cash_Flow", "Cash_Balance"]].copy()
    df_scenario["Scenario"] = "Optimistic"
    df_all_scenarios = pd.concat([df_all_scenarios, df_scenario], ignore_index=True)

# Add Conservative if toggled
if st.session_state.toggle_conservative:
    df_scenario = SCENARIO_MAP["Conservative"]["df"][["Month", "Revenue", "Total_Costs", "Net_Cash_Flow", "Cash_Balance"]].copy()
    df_scenario["Scenario"] = "Conservative"
    df_all_scenarios = pd.concat([df_all_scenarios, df_scenario], ignore_index=True)

# Add Custom if toggled
if st.session_state.toggle_custom:
    df_scenario = SCENARIO_MAP["Custom"]["df"][["Month", "Revenue", "Total_Costs", "Net_Cash_Flow", "Cash_Balance"]].copy()
    df_scenario["Scenario"] = "Custom"
    df_all_scenarios = pd.concat([df_all_scenarios, df_scenario], ignore_index=True)

# Add forecast if enabled
if not df_forecast.empty:
    df_forecast_export = df_forecast[["Month", "Revenue", "Total_Costs", "Net_Cash_Flow", "Cash_Balance"]].copy()
    df_forecast_export["Scenario"] = f"{forecast_base} + Forecast"
    df_all_scenarios = pd.concat([df_all_scenarios, df_forecast_export], ignore_index=True)

# DETAILED MODE: Add projected historical values
if DETAILED_MODE and df_projected_hist is not None:
    df_proj_export = df_projected_hist.copy()
    df_proj_export["Scenario"] = f"DETAILED: {forecast_method} Projected"
    df_proj_export = df_proj_export.rename(columns={
        "Projected_Revenue": "Revenue",
        "Projected_Cost": "Total_Costs",
        "Projected_Net_Flow": "Net_Cash_Flow",
        "Projected_Cash_Balance": "Cash_Balance"
    })
    df_all_scenarios = pd.concat([df_all_scenarios, df_proj_export], ignore_index=True)

# Configure column display
col_config = {
    "Month": st.column_config.TextColumn("Month", width="small"),
    "Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
    "Total_Costs": st.column_config.NumberColumn("Costs", format="$%.2f"),
    "Net_Cash_Flow": st.column_config.NumberColumn("Net Cash Flow", format="$%.2f"),
    "Cash_Balance": st.column_config.NumberColumn("Cash Balance", format="$%.2f"),
    "Scenario": st.column_config.TextColumn("Scenario", width="medium")
}

if DETAILED_MODE:
    st.warning("🔍 **DETAILED MODE ENABLED**: Export includes projected historical values showing what the forecast method 'sees'")

st.dataframe(
    df_all_scenarios,
    width='stretch',
    hide_index=True,
    column_config=col_config
)

csv = df_all_scenarios.to_csv(index=False)
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="cashflow_export.csv",
    mime="text/csv"
)



# --------------------------
# Footer & Legal
# --------------------------
st.markdown("---")

if SHOW_WARNING_EXPORT:
    st.warning(WARNING_EXPORT)

# Feeedback
st.markdown("---")

st.markdown("""
<div style="background: #e3f2fd; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center;">
    <p style="color: #1565c0; font-size: 16px; font-weight: 600; margin: 0 0 12px 0;">✨ Share Feedback or Request Assistance</p>
    <a href="https://docs.google.com/forms/d/e/1FAIpQLSd7E0Vg3lD5SzrCbcJ7INpaMVX-Ad3WdSlgmiY-G8wXyBNymw/viewform?usp=header" target="_blank" rel="noopener" style="display: inline-block; background: #1565c0; color: white; padding: 10px 25px; border-radius: 5px; text-decoration: none; font-weight: 600; font-size: 13px;">
        Contact Us
    </a>
</div>
""", unsafe_allow_html=True)


# Buy Me a Coffee
st.markdown("""
<div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px; margin: 20px 0;">
    <p style="margin: 0 0 10px 0; font-size: 16px;">☕ <strong> Support this project and future work </strong></p>
    <a href="https://www.buymeacoffee.com/flourishingperspectivehub" target="_blank" rel="noopener">
        <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 40px;">
    </a>
</div>
""", unsafe_allow_html=True)

# Legal sections
with st.expander("⚖️ Legal & Privacy Information"):
    st.markdown("""
    **Educational Simulator**
    
    Best practice: **Assume all results may be inaccurate.** This tool demonstrates financial concepts through simplified models, not real-world predictions.
    
    **Your Responsibility:**
    - Verify all calculations independently.
    - Consult qualified professionals before financial decisions.
    - Use for learning and exploration only.
    
    ---
    
    ### Financial Disclaimer
    
    This tool is provided for informational and educational purposes only and does not constitute financial, investment, tax, or legal advice.
    
    ⚠️ **IMPORTANT**: This tool provides estimates only. Results may contain errors. Always verify calculations independently before making financial decisions.
    
    - **Not Professional Advice**: The calculations, projections, and scenarios are based on user-provided data and simplified models. They should not be relied upon as the sole basis for financial decisions.
    - **No Warranty**: This tool is provided "as is" without warranties of any kind, express or implied.
    - **Consult Professionals**: Always consult with qualified financial advisors, accountants, or legal professionals before making important financial decisions.
    - **Your Responsibility**: You are solely responsible for verifying the accuracy of inputs and interpreting results appropriately.
    
    ---
    
    ### Privacy & Data Handling
    
    **Important:** This tool is hosted on Streamlit Cloud. Your inputs are processed on their servers but are not stored by us after your session ends. For information on Streamlit's data handling, see their Privacy Policy.
    
    **Recommendations:**
    - Do not enter sensitive personal information.
    - Use representative numbers for planning purposes.
    
    **Forms & Feedback:** Submissions via Google Forms are stored by Google and are subject to Google's privacy policy.
    
    ---
    
    ### External Links, Affiliates & Advertising
    
    This site may contain:
    - Affiliate links to third-party services (we may earn commission on purchases).
    - Advertising or sponsored content.
    - Links to resources we recommend.
    
    **Disclosure**: We only recommend products or services we believe are valuable. Our recommendations are independent of any compensation received.
    
    **Third-Party Links**: Our site contains links to external resources. We are not responsible for the accuracy, availability, or practices of those sites.
    
    ---
    
    ### Terms of Use
    
    By using this tool, you agree to these terms:
    
    **Acceptable Use**:
    - Use this tool for lawful purposes only.
    - Do not attempt to reverse engineer, hack, or exploit the application.
    - Do not use automated tools to scrape or abuse the service.
    
    **Limitation of Liability**: To the maximum extent permitted by law, we are not liable for any damages arising from your use of this tool, including but not limited to financial losses, business interruptions, or data loss.
    
    **Modifications**: We reserve the right to modify, suspend, or discontinue the tool, or any portion of these terms, at any time without notice.
    
    **Governing Law & Disputes**: These terms are governed by the laws of the United States and the State of California. Any disputes arising from use of this tool should first be resolved through good faith negotiation. If unresolved, disputes will be settled through binding arbitration rather than in court. You agree to resolve disputes individually and waive the right to participate in class action lawsuits.

    ---

    ### Contact

    **Email:** FlourishingPerspectiveHub [AT] gmail [DOT] com

    For questions about this tool, please contact us at the email above or use the feedback form. 

    """)

st.markdown("---")
st.caption("💰 Revenue, Cost and Cash Flow Planner | Free educational tool — explore, plan, and learn. Always verify all calculations.")