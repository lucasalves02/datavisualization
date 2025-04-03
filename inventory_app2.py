# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:31:44 2025
Streamlit Inventory Policy Analysis Dashboard
@author: lucas
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io # Needed for text area parsing
import traceback # To display detailed errors if needed

# --- Configuration & Styling ---
st.set_page_config(layout="wide") # Use wider layout

# --- Helper Functions ---

def perform_baseline_calculations(df):
    """Takes the raw DataFrame and returns the merged_df with diff %."""
    try:
        # Ensure input is a DataFrame
        if not isinstance(df, pd.DataFrame):
             st.error("Input data is not a valid DataFrame.")
             return None

        # Check for required columns
        required_cols = {'POLICY', 'LT', 'SERVICE_LEVEL', 'STOCK'}
        if not required_cols.issubset(df.columns):
            st.error(f"Input data must contain columns: {required_cols}")
            return None

        # Convert relevant columns to numeric, coercing errors
        df['SERVICE_LEVEL'] = pd.to_numeric(df['SERVICE_LEVEL'], errors='coerce')
        df['STOCK'] = pd.to_numeric(df['STOCK'], errors='coerce')
        # Drop rows where essential numeric conversions failed
        df.dropna(subset=['SERVICE_LEVEL', 'STOCK'], inplace=True)

        # Pivot baseline (BSL) for calculations
        if 'BSL' not in df['POLICY'].unique():
            st.error("Baseline policy 'BSL' not found in the 'POLICY' column.")
            return None
        # Ensure baseline has unique LT index after potential NA drops
        baseline_df = df[df['POLICY'] == 'BSL'].drop_duplicates(subset=['LT'])
        baseline_df = baseline_df[['LT', 'SERVICE_LEVEL', 'STOCK']].set_index('LT')


        # Merge baseline data with other policies
        merged_df = df[df['POLICY'] != 'BSL'].merge(baseline_df, on='LT', suffixes=('', '_BSL'), how='left')

        # Handle cases where merge might fail or result in NAs
        if merged_df['SERVICE_LEVEL_BSL'].isnull().any() or merged_df['STOCK_BSL'].isnull().any():
             st.warning("Some non-BSL entries didn't find a matching 'LT' in the baseline 'BSL' data or baseline had missing values. Affected rows dropped.")
             merged_df.dropna(subset=['SERVICE_LEVEL_BSL', 'STOCK_BSL'], inplace=True)

        if merged_df.empty:
            st.warning("No data left after merging with baseline. Cannot calculate differences.")
            return None

        # Calculate percentage differences, avoiding division by zero
        merged_df['STOCK_DIFF%'] = 100 * (merged_df['STOCK'] - merged_df['STOCK_BSL']) / merged_df['STOCK_BSL'].replace({0: np.nan})
        merged_df['SERVICE_DIFF%'] = 100 * (merged_df['SERVICE_LEVEL'] - merged_df['SERVICE_LEVEL_BSL']) / merged_df['SERVICE_LEVEL_BSL'].replace({0: np.nan})
        # Drop rows where calculation itself failed (e.g., division by NaN)
        merged_df.dropna(subset=['STOCK_DIFF%', 'SERVICE_DIFF%'], inplace=True)

        # Rename Lead Times using mapping
        lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        merged_df['LT_Percent'] = merged_df['LT'].map(lt_mapping).fillna(merged_df['LT'].astype(str)) # Keep original LT (as string) if not in mapping

        return merged_df

    except Exception as e:
        st.error(f"Error during baseline calculations: {e}")
        # st.code(traceback.format_exc()) # Uncomment for detailed debugging
        return None

# --- Plotting Functions ---

# ++ Add figure_width and figure_height parameters ++
def plot_relative_performance(merged_df, custom_title=None,
                              xlim_min=-14, xlim_max=5, ylim_min=-14, ylim_max=5,
                              figure_width=12.0, figure_height=7.0):
    """Generates the Relative Performance plot with dynamic axis limits and figure size."""
    if merged_df is None or merged_df.empty:
        st.warning("Cannot generate relative performance plot: No processed data available.")
        return None

    # ++ Use parameters for figsize ++
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    try:
        # Color definition
        lt_labels_unique = sorted(merged_df['LT_Percent'].unique())
        num_colors_needed = len(lt_labels_unique)
        palette = sns.color_palette('tab10', n_colors=max(num_colors_needed, 10))
        distinct_colors = palette[:num_colors_needed]
        lt_color_map = dict(zip(lt_labels_unique, distinct_colors))

        # Heatmap using Slider Limits
        x = np.linspace(xlim_min, xlim_max, 200); y = np.linspace(ylim_min, ylim_max, 200)
        X, Y = np.meshgrid(x, y); Z = Y - X
        ax.contourf(X, Y, Z, levels=100, cmap='RdYlGn', alpha=0.6)

        # Scatter plot
        scatter = sns.scatterplot( ax=ax, data=merged_df, x='STOCK_DIFF%', y='SERVICE_DIFF%',
            hue='LT_Percent', hue_order=lt_labels_unique, style='POLICY', s=200, palette=lt_color_map, edgecolor='black' )

        # Reference lines, Labels, Title
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--'); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Relative Performance: Stock vs Service Level Difference (%)'
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel('Inventory Difference (%) (Negative is Better)', fontsize=12); ax.set_ylabel('Service Level Difference (%) (Positive is Better)', fontsize=12)

        # Custom Legends
        handles, labels = scatter.get_legend_handles_labels()
        lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in lt_labels_unique if lt in lt_color_map]
        unique_policies = merged_df['POLICY'].unique(); policy_handles = []; policy_labels_for_legend = []
        for policy in unique_policies:
             try:
                  handle_found = False
                  for h, l in zip(handles, labels):
                       label_str = str(l)
                       if policy == label_str or (isinstance(l, tuple) and policy in l):
                            if policy not in policy_labels_for_legend:
                                 policy_handles.append(h); policy_labels_for_legend.append(policy); handle_found = True; break
             except Exception: pass
        legend1 = ax.legend(handles=lt_patches, title='SL Target (%)', bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small') # Added fontsize
        ax.add_artist(legend1)
        if policy_handles: ax.legend(handles=policy_handles, labels=policy_labels_for_legend, title='Policy', bbox_to_anchor=(1.03, 0.55), loc='upper left', fontsize='small') # Added fontsize
        else: pass

        # Manual Axes Positioning
        ax.set_position([0.1, 0.1, 0.65, 0.8]) # Adjust width (0.65) if needed based on figsize

        # Set Axis Limits
        ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True, linestyle='--')

        return fig
    except Exception as e:
        st.error(f"Error generating relative performance plot: {e}"); return None

# ++ Add figure_width and figure_height parameters ++
def plot_absolute_performance(input_df, custom_title=None, figure_width=12.0, figure_height=7.0):
    """Generates the Absolute Performance plots (Inventory & Service Level) with dynamic figure size."""
    if input_df is None or input_df.empty:
        st.warning("Cannot generate absolute performance plots: No raw data available."); return None

    df = input_df.copy()
    # ++ Use parameters for figsize ++
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height))

    try:
        # Data processing...
        required_cols = {'POLICY', 'LT', 'SERVICE_LEVEL', 'STOCK'};
        if not required_cols.issubset(df.columns): st.error(f"Absolute plot requires columns: {required_cols}"); return None
        df['STOCK'] = pd.to_numeric(df['STOCK'], errors='coerce'); df['SERVICE_LEVEL'] = pd.to_numeric(df['SERVICE_LEVEL'], errors='coerce')
        df['LT'] = df['LT'].astype(str); df.dropna(subset=['POLICY', 'LT', 'STOCK', 'SERVICE_LEVEL'], inplace=True)
        if df.empty: st.warning("No valid data remaining after cleaning for absolute plots."); return None
        lt_full_order = ['LT1', 'LT2', 'LT3', 'LT4', 'LT5']; lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        existing_lts_in_order = [lt for lt in lt_full_order if lt in df['LT'].unique()]
        lt_categories = existing_lts_in_order if existing_lts_in_order else sorted(df['LT'].unique())
        df['LT'] = pd.Categorical(df['LT'], categories=lt_categories, ordered=True); df = df.sort_values('LT')
        df['LT_Label'] = df['LT'].map(lt_mapping).fillna(df['LT'].astype(str))
        label_order = [lt_mapping.get(cat, str(cat)) for cat in lt_categories]
        df['LT_Label'] = pd.Categorical(df['LT_Label'], categories=label_order, ordered=True)

        # Plotting...
        sns.lineplot(ax=axes[0], data=df, x='LT_Label', y='STOCK', hue='POLICY', marker='o', linewidth=2)
        axes[0].set_xlabel('Service Level Target (%)'); axes[0].set_ylabel('Inventory (Units)')
        axes[0].grid(True); axes[0].legend(title='Policy'); axes[0].set_ylim(bottom=0)
        sns.lineplot(ax=axes[1], data=df, x='LT_Label', y='SERVICE_LEVEL', hue='POLICY', marker='o', linewidth=2)
        axes[1].set_xlabel('Service Level Target (%)'); axes[1].set_ylabel('Service Level')
        axes[1].grid(True); axes[1].legend(title='Policy')
        min_sl = df['SERVICE_LEVEL'].min(); max_sl = df['SERVICE_LEVEL'].max()
        axes[1].set_ylim(bottom=max(0, min_sl - 0.05), top=min(1.0, max_sl + 0.05))

        # Figure Title & Layout
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Absolute Performance Levels by Policy'
        fig.suptitle(plot_title, fontsize=16, y=1.02)
        # Use standard tight_layout here
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        return fig
    except Exception as e:
        st.error(f"Error generating absolute performance plots: {e}"); return None

# ++ Add figure_width and figure_height parameters ++
def plot_quadrant_analysis(merged_df, custom_title=None,
                           xlim_min=-14, xlim_max=5, ylim_min=-14, ylim_max=5,
                           stock_target_thresh=-6.0, service_target_thresh=0.0, service_floor_thresh=-10.0,
                           figure_width=12.0, figure_height=7.0):
    """Generates the Quadrant Analysis plot with dynamic axis limits, desirability logic, and figure size."""
    if merged_df is None or merged_df.empty:
        st.warning("Cannot generate quadrant analysis plot: No processed data available."); return None
    if service_floor_thresh < ylim_min: st.warning(f"Service floor threshold ({service_floor_thresh}%) is below Y-axis minimum ({ylim_min}%).")
    if stock_target_thresh < xlim_min: st.warning(f"Stock target threshold ({stock_target_thresh}%) is below X-axis minimum ({xlim_min}%).")

    # ++ Use parameters for figsize ++
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    try:
        plot_data_df = merged_df # Use all data points

        # Color definition
        lt_labels_unique = sorted(plot_data_df['LT_Percent'].unique())
        num_colors_needed = len(lt_labels_unique)
        palette = sns.color_palette('tab10', n_colors=max(num_colors_needed, 10))
        distinct_colors = palette[:num_colors_needed]
        lt_color_map = dict(zip(lt_labels_unique, distinct_colors))

        # Heatmap Calculation using Desirability Params
        grid_size = 300; x_grid = np.linspace(xlim_min, xlim_max, grid_size); y_grid = np.linspace(ylim_min, ylim_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid); stock_score = np.zeros_like(X); service_score = np.zeros_like(Y)
        stock_score[X <= stock_target_thresh] = 1; mask_stock = (X > stock_target_thresh) & (X <= 0); stock_score_range = abs(stock_target_thresh)
        if stock_score_range > 1e-6: stock_score[mask_stock] = np.abs(X[mask_stock]) / stock_score_range
        stock_score = np.clip(stock_score, 0, 1); service_score[Y >= service_target_thresh] = 1; service_score_range = service_target_thresh - service_floor_thresh
        if service_score_range > 1e-6: mask_serv = (Y < service_target_thresh) & (Y >= service_floor_thresh); service_score[mask_serv] = (Y[mask_serv] - service_floor_thresh) / service_score_range
        service_score = np.clip(service_score, 0, 1); Z = stock_score * service_score
        # Plot heatmap
        img = ax.imshow(Z, extent=[xlim_min, xlim_max, ylim_min, ylim_max], origin='lower', cmap='RdYlGn', aspect='auto', alpha=0.6)

        # Overlay scatterplot
        if not plot_data_df.empty:
             scatter = sns.scatterplot( ax=ax, data=plot_data_df, x='STOCK_DIFF%', y='SERVICE_DIFF%', hue='LT_Percent', hue_order=lt_labels_unique,
                 style='POLICY', s=200, palette=lt_color_map, edgecolor='black' )
        else: scatter = None; st.info("No data points to plot.")

        # Axes, Title, Legend setup
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--'); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.add_patch(mpatches.Rectangle((xlim_min, ylim_min), xlim_max - xlim_min, ylim_max - ylim_min, hatch='///', fill=False, edgecolor='gray', linewidth=0, alpha=0.05))
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Quadrant Analysis: Stock vs Service Level Difference (%)'
        ax.set_title(plot_title, fontsize=14); ax.set_xlabel('Inventory Difference (%)', fontsize=12); ax.set_ylabel('Service Level Difference (%)', fontsize=12)

        # Legends
        if scatter:
             handles, labels = scatter.get_legend_handles_labels()
             lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in lt_labels_unique if lt in lt_color_map]
             unique_policies = plot_data_df['POLICY'].unique(); policy_handles = []; policy_labels_for_legend = []
             for policy in unique_policies:
                 try:
                      handle_found = False
                      for h, l in zip(handles, labels):
                           label_str = str(l)
                           if policy == label_str or (isinstance(l, tuple) and policy in l):
                                if policy not in policy_labels_for_legend:
                                     policy_handles.append(h); policy_labels_for_legend.append(policy); handle_found = True; break
                 except Exception: pass
             legend1 = ax.legend(handles=lt_patches, title='SL Target (%)', bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small') # Added fontsize
             ax.add_artist(legend1)
             if policy_handles: ax.legend(handles=policy_handles, labels=policy_labels_for_legend, title='Policy', bbox_to_anchor=(1.03, 0.55), loc='upper left', fontsize='small') # Added fontsize
             else: pass

        # Manual Axes Positioning
        ax.set_position([0.1, 0.1, 0.65, 0.8]) # Adjust width (0.65) if needed based on figsize

        # Set Axis Limits
        ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True, linestyle='--')

        return fig
    except Exception as e:
        st.error(f"Error generating quadrant analysis plot: {e}"); return None

# --- Streamlit App Layout ---

st.title("Inventory Policy Analysis Dashboard")

# --- Sidebar for Data Input ---
st.sidebar.header("Data Input")
# Method 1: Paste into Text Area
st.sidebar.subheader("Paste Data from Excel")
st.sidebar.info("Copy data range (incl. headers), paste below, then click 'Load'.")
if 'pasted_text_area' not in st.session_state: st.session_state.pasted_text_area = ""
pasted_text_widget_value = st.sidebar.text_area("Paste tab-separated data here:", key="pasted_text_area", height=150)
if st.sidebar.button("Load Data from Text Area", key="load_text_button"):
    current_pasted_text = st.session_state.pasted_text_area
    if current_pasted_text:
        try:
            string_io = io.StringIO(current_pasted_text)
            text_data = pd.read_csv(string_io, sep='\t', header=0, engine='python', on_bad_lines='warn')
            if text_data is not None and not text_data.empty:
                st.session_state['raw_df'] = text_data
                st.session_state['merged_df'] = perform_baseline_calculations(st.session_state['raw_df'])
                st.sidebar.success(f"Loaded {len(text_data)} rows from text.")
            else:
                 st.sidebar.error("Could not parse text or data is empty.")
                 if 'raw_df' in st.session_state: del st.session_state['raw_df'];
                 if 'merged_df' in st.session_state: del st.session_state['merged_df'];
        except Exception as e:
            st.sidebar.error(f"Error parsing text area: {e}")
            if 'raw_df' in st.session_state: del st.session_state['raw_df'];
            if 'merged_df' in st.session_state: del st.session_state['merged_df'];
    else: st.sidebar.warning("Text area is empty.")
st.sidebar.markdown("---")
# Method 2: File Uploader
st.sidebar.subheader("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx) or CSV (.csv):", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    try:
        file_data = None
        if uploaded_file.name.endswith('.csv'): file_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')): file_data = pd.read_excel(uploaded_file, engine='openpyxl')
        if file_data is not None and not file_data.empty:
            st.session_state['raw_df'] = file_data
            st.session_state['merged_df'] = perform_baseline_calculations(st.session_state['raw_df'])
            st.sidebar.success(f"Loaded {len(file_data)} rows from: {uploaded_file.name}")
        elif file_data is not None:
             st.sidebar.warning(f"File '{uploaded_file.name}' is empty.")
             if 'raw_df' in st.session_state: del st.session_state['raw_df'];
             if 'merged_df' in st.session_state: del st.session_state['merged_df'];
        else: st.sidebar.error("Could not read the uploaded file format.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        if 'raw_df' in st.session_state: del st.session_state['raw_df'];
        if 'merged_df' in st.session_state: del st.session_state['merged_df'];

# --- Main Display Area ---
if 'raw_df' in st.session_state:
    st.header("Loaded Data Preview")
    st.dataframe(st.session_state['raw_df'].head(), height=200)

    st.header("Analysis Plots")
    st.markdown("---")

    # --- Plot Controls ---
    control_col1, control_col2 = st.columns([0.6, 0.4])
    with control_col1: plot_choice = st.selectbox("Choose Plot:", ["Relative Performance (vs Baseline)", "Absolute Performance", "Quadrant Analysis"], key="plot_select")
    with control_col2: custom_title_input = st.text_input("Custom graph title (optional):", placeholder="Uses default if empty", key="custom_title_input")
    user_title = custom_title_input.strip() if custom_title_input else None

    # ++ Figure Size Controls ++
    st.subheader("Figure Size (inches)")
    size_col1, size_col2 = st.columns(2)
    with size_col1:
        fig_width = st.slider("Width:", min_value=6.0, max_value=20.0, value=12.0, step=0.5, key="fig_width_slider", format="%.1f")
    with size_col2:
        fig_height = st.slider("Height:", min_value=4.0, max_value=15.0, value=7.0, step=0.5, key="fig_height_slider", format="%.1f")
    # ++ End Figure Size Controls ++

    # --- Axis Limit Controls (Conditional) ---
    show_diff_plot_controls = plot_choice in ["Relative Performance (vs Baseline)", "Quadrant Analysis"]
    limits_valid = True
    x_min_limit, x_max_limit = -14, 5; y_min_limit, y_max_limit = -14, 5 # Defaults
    if show_diff_plot_controls:
        st.markdown("---")
        st.subheader("Axis Limits")
        limit_col1, limit_col2 = st.columns(2)
        with limit_col1: x_min_limit = st.slider("X-Axis Min (%)", -50, 50, -14, 1, key="x_min_slider"); y_min_limit = st.slider("Y-Axis Min (%)", -50, 50, -14, 1, key="y_min_slider")
        with limit_col2: x_max_limit = st.slider("X-Axis Max (%)", -50, 50, 5, 1, key="x_max_slider"); y_max_limit = st.slider("Y-Axis Max (%)", -50, 50, 5, 1, key="y_max_slider")
        if x_min_limit >= x_max_limit or y_min_limit >= y_max_limit: st.error("Axis Min limit must be strictly less than Max limit."); limits_valid = False

    # --- Desirability Controls (Conditional for Quadrant Plot) ---
    stock_target_thresh_param = -6.0; service_target_thresh_param = 0.0; service_floor_thresh_param = -10.0 # Defaults
    if plot_choice == "Quadrant Analysis":
        # st.markdown("---") # Separator might be redundant
        st.subheader("Desirability Logic Parameters")
        des_col1, des_col2, des_col3 = st.columns(3)
        with des_col1: stock_target_reduc_input = st.number_input("Stock Reduction for Max Score (%):", min_value=0.1, max_value=50.0, value=6.0, step=0.5, format="%.1f", help="e.g., 6 means -6%", key="stock_target_reduc"); stock_target_thresh_param = -abs(stock_target_reduc_input)
        with des_col2: service_target_thresh_param = st.number_input("Service Level for Max Score (%):", min_value=-20.0, max_value=20.0, value=0.0, step=0.5, format="%.1f", help="Service change at/above this gets full score.", key="service_target_level")
        with des_col3: service_floor_thresh_param = st.number_input("Service Level for Zero Score (%):", min_value=-50.0, max_value=-0.1, value=-10.0, step=0.5, format="%.1f", help="Service change at/below this gets zero score.", key="service_floor_level")
        if service_floor_thresh_param >= service_target_thresh_param: st.error("Desirability Error: Service 'Zero Score Level' must be less than 'Max Score Level'."); limits_valid = False
        st.markdown("---")


    # --- Display Chosen Plot ---
    fig_to_show = None; plot_error = False
    try:
        # ++ Pass fig_width and fig_height to ALL plot functions ++
        if plot_choice == "Relative Performance (vs Baseline)":
            if 'merged_df' in st.session_state and st.session_state['merged_df'] is not None:
                if limits_valid:
                    st.subheader("Relative Performance")
                    fig_to_show = plot_relative_performance(st.session_state['merged_df'], custom_title=user_title,
                        xlim_min=x_min_limit, xlim_max=x_max_limit, ylim_min=y_min_limit, ylim_max=y_max_limit,
                        figure_width=fig_width, figure_height=fig_height ) # Pass size
                else: plot_error = True
            else: st.warning("Data not processed for relative plot.")

        elif plot_choice == "Absolute Performance":
            st.subheader("Absolute Performance")
            fig_to_show = plot_absolute_performance(st.session_state['raw_df'], custom_title=user_title,
                        figure_width=fig_width, figure_height=fig_height ) # Pass size

        elif plot_choice == "Quadrant Analysis":
             if 'merged_df' in st.session_state and st.session_state['merged_df'] is not None:
                 if limits_valid:
                    st.subheader("Quadrant Analysis")
                    fig_to_show = plot_quadrant_analysis(st.session_state['merged_df'], custom_title=user_title,
                        xlim_min=x_min_limit, xlim_max=x_max_limit, ylim_min=y_min_limit, ylim_max=y_max_limit,
                        stock_target_thresh=stock_target_thresh_param, service_target_thresh=service_target_thresh_param, service_floor_thresh=service_floor_thresh_param,
                        figure_width=fig_width, figure_height=fig_height ) # Pass size
                 else: plot_error = True
             else: st.warning("Data not processed for quadrant plot.")

        # Display the figure
        if fig_to_show:
             st.pyplot(fig_to_show)
        elif not plot_error and 'raw_df' in st.session_state:
             st.warning("Could not generate the selected plot with the current data/settings.")

    except Exception as display_e:
         st.error(f"An unexpected error occurred trying to display plot: {display_e}")

# Initial State Message
else:
    st.info("Welcome! Please load data using the sidebar options (Paste or Upload).")
    st.markdown("---"); st.subheader("Expected Data Format")
    st.markdown("- `POLICY`: Text (e.g., 'BSL', 'IDL'). Include 'BSL'.\n- `LT`: Text (e.g., 'LT1', 'LT2').\n- `SERVICE_LEVEL`: Numeric (e.g., 0.876 or 87.6).\n- `STOCK`: Numeric (e.g., 2862.495).")
    st.subheader("Example:"); sample_data = { 'POLICY': ['BSL', 'IDL', 'BSL', 'IDL'], 'LT': ['LT1', 'LT1', 'LT2', 'LT2'], 'SERVICE_LEVEL': [0.88, 0.85, 0.90, 0.88], 'STOCK': [1000, 950, 1200, 1100] }; st.dataframe(pd.DataFrame(sample_data))
