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
        # ... (Data processing logic - unchanged) ...
        if not isinstance(df, pd.DataFrame): st.error("Input not DataFrame."); return None
        required_cols = {'POLICY', 'LT', 'SERVICE_LEVEL', 'STOCK'}
        if not required_cols.issubset(df.columns): st.error(f"Missing columns: {required_cols-set(df.columns)}"); return None
        df['SERVICE_LEVEL'] = pd.to_numeric(df['SERVICE_LEVEL'], errors='coerce'); df['STOCK'] = pd.to_numeric(df['STOCK'], errors='coerce')
        df.dropna(subset=['SERVICE_LEVEL', 'STOCK'], inplace=True)
        if 'BSL' not in df['POLICY'].unique(): st.error("Baseline 'BSL' not found."); return None
        baseline_df = df[df['POLICY'] == 'BSL'].drop_duplicates(subset=['LT'])
        baseline_df = baseline_df[['LT', 'SERVICE_LEVEL', 'STOCK']].set_index('LT')
        merged_df = df[df['POLICY'] != 'BSL'].merge(baseline_df, on='LT', suffixes=('', '_BSL'), how='left')
        if merged_df['SERVICE_LEVEL_BSL'].isnull().any() or merged_df['STOCK_BSL'].isnull().any():
             st.warning("Some rows dropped due to missing baseline match."); merged_df.dropna(subset=['SERVICE_LEVEL_BSL', 'STOCK_BSL'], inplace=True)
        if merged_df.empty: st.warning("No data after baseline merge."); return None
        merged_df['STOCK_DIFF%'] = 100 * (merged_df['STOCK'] - merged_df['STOCK_BSL']) / merged_df['STOCK_BSL'].replace({0: np.nan})
        merged_df['SERVICE_DIFF%'] = 100 * (merged_df['SERVICE_LEVEL'] - merged_df['SERVICE_LEVEL_BSL']) / merged_df['SERVICE_LEVEL_BSL'].replace({0: np.nan})
        merged_df.dropna(subset=['STOCK_DIFF%', 'SERVICE_DIFF%'], inplace=True)
        lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        merged_df['LT_Percent'] = merged_df['LT'].map(lt_mapping).fillna(merged_df['LT'].astype(str))
        return merged_df
    except Exception as e: st.error(f"Error during baseline calculations: {e}"); return None

# --- Plotting Functions ---

def plot_relative_performance(merged_df, custom_title=None,
                              xlim_min=-14, xlim_max=5, ylim_min=-14, ylim_max=5,
                              figure_width=10.0, figure_height=8.0, # Changed default figsize
                              title_axis_fontsize=12, legend_fontsize=9):
    """Generates the Relative Performance plot with bottom legend."""
    if merged_df is None or merged_df.empty: st.warning("No processed data for relative plot."); return None

    fig, ax = plt.subplots(figsize=(figure_width, figure_height)) # Use new default figsize
    try:
        # Color definition...
        lt_labels_unique = sorted(merged_df['LT_Percent'].unique())
        num_colors_needed = len(lt_labels_unique); palette = sns.color_palette('tab10', n_colors=max(num_colors_needed, 10))
        distinct_colors = palette[:num_colors_needed]; lt_color_map = dict(zip(lt_labels_unique, distinct_colors))

        # Heatmap...
        x = np.linspace(xlim_min, xlim_max, 200); y = np.linspace(ylim_min, ylim_max, 200); X, Y = np.meshgrid(x, y); Z = Y - X
        ax.contourf(X, Y, Z, levels=100, cmap='RdYlGn', alpha=0.6)

        # Scatter plot...
        scatter = sns.scatterplot( ax=ax, data=merged_df, x='STOCK_DIFF%', y='SERVICE_DIFF%', hue='LT_Percent', hue_order=lt_labels_unique, style='POLICY', s=200, palette=lt_color_map, edgecolor='black' )

        # Reference lines, Labels, Title...
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--'); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Relative Performance: Stock vs Service Level Difference (%)'
        ax.set_title(plot_title, fontsize=title_axis_fontsize + 2); ax.set_xlabel('Inventory Difference (%) (Negative is Better)', fontsize=title_axis_fontsize)
        ax.set_ylabel('Service Level Difference (%) (Positive is Better)', fontsize=title_axis_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2)

        # === Combine Legends for Bottom Placement ===
        handles, labels = scatter.get_legend_handles_labels()
        # Get SL Target handles (patches)
        lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in lt_labels_unique if lt in lt_color_map]
        # Get Policy handles (markers)
        unique_policies = merged_df['POLICY'].unique(); policy_handles = []; policy_labels_for_legend = []
        # This logic to extract policy handles might need refinement depending on seaborn version/label format
        temp_legend = ax.legend(handles=handles, labels=labels) # Create temporary legend to extract styled handles
        policy_handle_map = {l.get_label(): h for h, l in zip(temp_legend.legendHandles, temp_legend.get_texts()) if l.get_label() in unique_policies}
        temp_legend.remove() # Remove temporary legend
        for policy in unique_policies: # Ensure order
             if policy in policy_handle_map:
                  policy_handles.append(policy_handle_map[policy])
                  policy_labels_for_legend.append(policy)

        # Combine all handles and labels
        all_handles = lt_patches + policy_handles
        all_labels = [p.get_label() for p in lt_patches] + policy_labels_for_legend

        # Remove the old separate legend calls and ax.add_artist
        # legend1 = ax.legend(...)
        # ax.add_artist(legend1)
        # if policy_handles: ax.legend(...)

        # Create single legend at the bottom
        if all_handles:
            ax.legend(all_handles, all_labels,
                      loc='upper center', # Anchor point on the legend
                      bbox_to_anchor=(0.5, -0.15), # Position: 0.5=center horizontally, -0.15=below axes
                      ncol=max(len(lt_patches), len(policy_handles)), # Arrange in columns horizontally
                      fontsize=legend_fontsize, title_fontsize=legend_fontsize) # Apply legend size

        # --- Remove Manual Axes Positioning ---
        # ax.set_position([0.1, 0.1, 0.65, 0.8]) # REMOVED

        # --- Set Axis Limits & Grid ---
        ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True, linestyle='--')
        # -----------------------------

        # --- Use Figure-level tight_layout AFTER legend placement ---
        try:
             fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect bottom/top if needed for legend/title space
        except ValueError: # Handle potential tight_layout errors
             plt.subplots_adjust(bottom=0.2, top=0.9) # Manual adjustment as fallback

        return fig
    except Exception as e: st.error(f"Error generating relative plot: {e}"); return None


def plot_absolute_performance(input_df, custom_title=None,
                              figure_width=10.0, figure_height=8.0, # Changed default figsize
                              title_axis_fontsize=12, legend_fontsize=9):
    """Generates the Absolute Performance plots with dynamic sizes."""
    # This plot's legends are typically inside, no changes needed for bottom placement unless desired.
    # Keeping original legend logic for this one.
    if input_df is None or input_df.empty: st.warning("No raw data for absolute plot."); return None
    df = input_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height)) # Use new default figsize
    try:
        # Data processing... (remains the same)
        required_cols = {'POLICY', 'LT', 'SERVICE_LEVEL', 'STOCK'};
        if not required_cols.issubset(df.columns): st.error(f"Absolute plot requires columns: {required_cols}"); return None
        df['STOCK'] = pd.to_numeric(df['STOCK'], errors='coerce'); df['SERVICE_LEVEL'] = pd.to_numeric(df['SERVICE_LEVEL'], errors='coerce')
        df['LT'] = df['LT'].astype(str); df.dropna(subset=['POLICY', 'LT', 'STOCK', 'SERVICE_LEVEL'], inplace=True)
        if df.empty: st.warning("No valid data for absolute plots after cleaning."); return None
        lt_full_order = ['LT1', 'LT2', 'LT3', 'LT4', 'LT5']; lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        existing_lts_in_order = [lt for lt in lt_full_order if lt in df['LT'].unique()]
        lt_categories = existing_lts_in_order if existing_lts_in_order else sorted(df['LT'].unique())
        df['LT'] = pd.Categorical(df['LT'], categories=lt_categories, ordered=True); df = df.sort_values('LT')
        df['LT_Label'] = df['LT'].map(lt_mapping).fillna(df['LT'].astype(str))
        label_order = [lt_mapping.get(cat, str(cat)) for cat in lt_categories]
        df['LT_Label'] = pd.Categorical(df['LT_Label'], categories=label_order, ordered=True)

        # Plotting with fontsizes...
        sns.lineplot(ax=axes[0], data=df, x='LT_Label', y='STOCK', hue='POLICY', marker='o', linewidth=2)
        axes[0].set_xlabel('Service Level Target (%)', fontsize=title_axis_fontsize); axes[0].set_ylabel('Inventory (Units)', fontsize=title_axis_fontsize)
        axes[0].tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2); axes[0].grid(True); axes[0].legend(title='Policy', fontsize=legend_fontsize, title_fontsize=legend_fontsize); axes[0].set_ylim(bottom=0)
        sns.lineplot(ax=axes[1], data=df, x='LT_Label', y='SERVICE_LEVEL', hue='POLICY', marker='o', linewidth=2)
        axes[1].set_xlabel('Service Level Target (%)', fontsize=title_axis_fontsize); axes[1].set_ylabel('Service Level', fontsize=title_axis_fontsize)
        axes[1].tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2); axes[1].grid(True); axes[1].legend(title='Policy', fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        min_sl = df['SERVICE_LEVEL'].min(); max_sl = df['SERVICE_LEVEL'].max(); axes[1].set_ylim(bottom=max(0, min_sl - 0.05), top=min(1.0, max_sl + 0.05))

        # Figure Title & Layout...
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Absolute Performance Levels by Policy'
        fig.suptitle(plot_title, fontsize=title_axis_fontsize + 2, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Keep standard tight_layout

        return fig
    except Exception as e: st.error(f"Error generating absolute plots: {e}"); return None

# ++ Add fontsize parameters ++
def plot_quadrant_analysis(merged_df, custom_title=None,
                           xlim_min=-14, xlim_max=5, ylim_min=-14, ylim_max=5,
                           stock_target_thresh=-6.0, service_target_thresh=0.0, service_floor_thresh=-10.0,
                           figure_width=10.0, figure_height=8.0, # Changed default figsize
                           title_axis_fontsize=12, legend_fontsize=9):
    """Generates the Quadrant Analysis plot with bottom legend."""
    if merged_df is None or merged_df.empty: st.warning("No processed data for quadrant plot."); return None
    if service_floor_thresh < ylim_min: st.warning(f"Service floor ({service_floor_thresh}%) < Y-min ({ylim_min}%).")
    if stock_target_thresh < xlim_min: st.warning(f"Stock target ({stock_target_thresh}%) < X-min ({xlim_min}%).")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height)) # Use new default figsize
    try:
        plot_data_df = merged_df # Use all data points

        # Color definition...
        lt_labels_unique = sorted(plot_data_df['LT_Percent'].unique()); num_colors_needed = len(lt_labels_unique)
        palette = sns.color_palette('tab10', n_colors=max(num_colors_needed, 10)); distinct_colors = palette[:num_colors_needed]
        lt_color_map = dict(zip(lt_labels_unique, distinct_colors))

        # Heatmap Calculation...
        grid_size = 300; x_grid = np.linspace(xlim_min, xlim_max, grid_size); y_grid = np.linspace(ylim_min, ylim_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid); stock_score = np.zeros_like(X); service_score = np.zeros_like(Y)
        stock_score[X <= stock_target_thresh] = 1; mask_stock = (X > stock_target_thresh) & (X <= 0); stock_score_range = abs(stock_target_thresh)
        if stock_score_range > 1e-6: stock_score[mask_stock] = np.abs(X[mask_stock]) / stock_score_range
        stock_score = np.clip(stock_score, 0, 1); service_score[Y >= service_target_thresh] = 1; service_score_range = service_target_thresh - service_floor_thresh
        if service_score_range > 1e-6: mask_serv = (Y < service_target_thresh) & (Y >= service_floor_thresh); service_score[mask_serv] = (Y[mask_serv] - service_floor_thresh) / service_score_range
        service_score = np.clip(service_score, 0, 1); Z = stock_score * service_score
        img = ax.imshow(Z, extent=[xlim_min, xlim_max, ylim_min, ylim_max], origin='lower', cmap='RdYlGn', aspect='auto', alpha=0.6)

        # Overlay scatterplot...
        if not plot_data_df.empty: scatter = sns.scatterplot( ax=ax, data=plot_data_df, x='STOCK_DIFF%', y='SERVICE_DIFF%', hue='LT_Percent', hue_order=lt_labels_unique, style='POLICY', s=200, palette=lt_color_map, edgecolor='black' )
        else: scatter = None; st.info("No data points to plot.")

        # Axes, Title setup...
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--'); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.add_patch(mpatches.Rectangle((xlim_min, ylim_min), xlim_max - xlim_min, ylim_max - ylim_min, hatch='///', fill=False, edgecolor='gray', linewidth=0, alpha=0.05))
        # Apply font sizes
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Quadrant Analysis: Stock vs Service Level Difference (%)'
        ax.set_title(plot_title, fontsize=title_axis_fontsize + 2); ax.set_xlabel('Inventory Difference (%)', fontsize=title_axis_fontsize)
        ax.set_ylabel('Service Level Difference (%)', fontsize=title_axis_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2)

        # === Combine Legends for Bottom Placement ===
        if scatter:
             handles, labels = scatter.get_legend_handles_labels()
             lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in lt_labels_unique if lt in lt_color_map]
             unique_policies = plot_data_df['POLICY'].unique(); policy_handles = []; policy_labels_for_legend = []
             # Refined handle extraction
             temp_legend = ax.legend(handles=handles, labels=labels); policy_handle_map = {l.get_label(): h for h, l in zip(temp_legend.legendHandles, temp_legend.get_texts()) if l.get_label() in unique_policies} ; temp_legend.remove()
             for policy in unique_policies:
                  if policy in policy_handle_map: policy_handles.append(policy_handle_map[policy]); policy_labels_for_legend.append(policy)

             all_handles = lt_patches + policy_handles
             all_labels = [p.get_label() for p in lt_patches] + policy_labels_for_legend

             # Remove old legend calls if any were missed (like ax.add_artist)
             # ax.add_artist(...) # REMOVED

             # Create single legend at the bottom
             if all_handles:
                ax.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), # Adjust y (-0.15) if needed
                          ncol=max(len(lt_patches), len(policy_handles)), fontsize=legend_fontsize, title_fontsize=legend_fontsize)

        # --- Remove Manual Axes Positioning ---
        # ax.set_position([0.1, 0.1, 0.65, 0.8]) # REMOVED

        # --- Set Axis Limits & Grid ---
        ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True, linestyle='--')
        # -----------------------------

        # --- Use Figure-level tight_layout AFTER legend placement ---
        try:
             fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect bottom/top for legend/title space
        except ValueError:
             plt.subplots_adjust(bottom=0.25, top=0.9) # Manual adjustment fallback

        return fig
    except Exception as e: st.error(f"Error generating quadrant plot: {e}"); return None


# --- Streamlit App Layout ---

st.title("Inventory Policy Analysis Dashboard")

# --- Sidebar for Data Input ---
st.sidebar.header("Data Input")
# (Sidebar code remains unchanged - Text Area & File Upload)
# ... [Same sidebar code as before] ...
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

    st.header("Plot Configuration")
    st.markdown("---")

    # --- Plot Controls ---
    # Row 1: Plot Choice and Title Input
    control_col1, control_col2 = st.columns([0.6, 0.4])
    with control_col1: plot_choice = st.selectbox("Choose Plot:", ["Relative Performance (vs Baseline)", "Absolute Performance", "Quadrant Analysis"], key="plot_select")
    with control_col2: custom_title_input = st.text_input("Custom graph title (optional):", placeholder="Uses default if empty", key="custom_title_input")
    user_title = custom_title_input.strip() if custom_title_input else None

    # Row 2: Figure Size Controls
    st.subheader("Figure Size (inches)")
    size_col1, size_col2 = st.columns(2)
    # ++ Update default figure size ++
    with size_col1: fig_width = st.slider("Width:", min_value=6.0, max_value=20.0, value=10.0, step=0.5, key="fig_width_slider", format="%.1f")
    with size_col2: fig_height = st.slider("Height:", min_value=4.0, max_value=15.0, value=8.0, step=0.5, key="fig_height_slider", format="%.1f")

    # Row 3: Font Size Controls
    st.subheader("Font Sizes")
    font_col1, font_col2 = st.columns(2)
    with font_col1: title_axis_font_size = st.slider("Title & Axis Label Size:", min_value=6, max_value=20, value=12, step=1, key="title_axis_font_slider")
    with font_col2: legend_font_size = st.slider("Legend Size:", min_value=5, max_value=18, value=9, step=1, key="legend_font_slider")

    # --- Conditional Controls: Axis Limits & Desirability ---
    show_diff_plot_controls = plot_choice in ["Relative Performance (vs Baseline)", "Quadrant Analysis"]
    limits_valid = True # Flag for validation checks
    x_min_limit, x_max_limit = -14, 5; y_min_limit, y_max_limit = -14, 5 # Defaults
    stock_target_thresh_param = -6.0; service_target_thresh_param = 0.0; service_floor_thresh_param = -10.0 # Defaults

    if show_diff_plot_controls:
        st.markdown("---")
        st.subheader("Axis Limits")
        limit_col1, limit_col2 = st.columns(2)
        with limit_col1: x_min_limit = st.slider("X-Axis Min (%)", -50, 50, -14, 1, key="x_min_slider"); y_min_limit = st.slider("Y-Axis Min (%)", -50, 50, -14, 1, key="y_min_slider")
        with limit_col2: x_max_limit = st.slider("X-Axis Max (%)", -50, 50, 5, 1, key="x_max_slider"); y_max_limit = st.slider("Y-Axis Max (%)", -50, 50, 5, 1, key="y_max_slider")
        if x_min_limit >= x_max_limit or y_min_limit >= y_max_limit: st.error("Axis Min limit must be strictly less than Max limit."); limits_valid = False

    if plot_choice == "Quadrant Analysis":
        st.subheader("Desirability Logic Parameters")
        des_col1, des_col2, des_col3 = st.columns(3)
        with des_col1: stock_target_reduc_input = st.number_input("Stock Reduction for Max Score (%):", min_value=0.1, max_value=50.0, value=6.0, step=0.5, format="%.1f", help="e.g., 6 means -6%", key="stock_target_reduc"); stock_target_thresh_param = -abs(stock_target_reduc_input)
        with des_col2: service_target_thresh_param = st.number_input("Service Level for Max Score (%):", min_value=-20.0, max_value=20.0, value=0.0, step=0.5, format="%.1f", help="Service change at/above this gets full score.", key="service_target_level")
        with des_col3: service_floor_thresh_param = st.number_input("Service Level for Zero Score (%):", min_value=-50.0, max_value=-0.1, value=-10.0, step=0.5, format="%.1f", help="Service change at/below this gets zero score.", key="service_floor_level")
        if service_floor_thresh_param >= service_target_thresh_param: st.error("Desirability Error: Service 'Zero Score Level' must be less than 'Max Score Level'."); limits_valid = False
    st.markdown("---") # Separator after all controls

    # --- Display Chosen Plot ---
    st.header("Generated Plot") # Add header for the plot itself
    fig_to_show = None; plot_error = False
    try:
        # Pass figure size and font sizes to ALL plot functions
        if plot_choice == "Relative Performance (vs Baseline)":
            if 'merged_df' in st.session_state and st.session_state['merged_df'] is not None:
                if limits_valid:
                    fig_to_show = plot_relative_performance(st.session_state['merged_df'], custom_title=user_title,
                        xlim_min=x_min_limit, xlim_max=x_max_limit, ylim_min=y_min_limit, ylim_max=y_max_limit,
                        figure_width=fig_width, figure_height=fig_height,
                        title_axis_fontsize=title_axis_font_size, legend_fontsize=legend_font_size)
                else: plot_error = True
            else: st.warning("Data not processed for relative plot.")

        elif plot_choice == "Absolute Performance":
            fig_to_show = plot_absolute_performance(st.session_state['raw_df'], custom_title=user_title,
                        figure_width=fig_width, figure_height=fig_height,
                        title_axis_fontsize=title_axis_font_size, legend_fontsize=legend_font_size)

        elif plot_choice == "Quadrant Analysis":
             if 'merged_df' in st.session_state and st.session_state['merged_df'] is not None:
                 if limits_valid:
                    fig_to_show = plot_quadrant_analysis(st.session_state['merged_df'], custom_title=user_title,
                        xlim_min=x_min_limit, xlim_max=x_max_limit, ylim_min=y_min_limit, ylim_max=y_max_limit,
                        stock_target_thresh=stock_target_thresh_param, service_target_thresh=service_target_thresh_param, service_floor_thresh=service_floor_thresh_param,
                        figure_width=fig_width, figure_height=fig_height,
                        title_axis_fontsize=title_axis_font_size, legend_fontsize=legend_font_size)
                 else: plot_error = True
             else: st.warning("Data not processed for quadrant plot.")

        # Display the figure
        if fig_to_show: st.pyplot(fig_to_show)
        elif not plot_error and 'raw_df' in st.session_state: st.warning("Could not generate plot with current settings.")

    except Exception as display_e: st.error(f"An unexpected error occurred trying to display plot: {display_e}")

# Initial State Message
else:
    st.info("Welcome! Please load data using the sidebar options (Paste or Upload).")
    # ... (rest of initial message remains same) ...
    st.markdown("---"); st.subheader("Expected Data Format")
    st.markdown("- `POLICY`: Text (e.g., 'BSL', 'IDL'). Include 'BSL'.\n- `LT`: Text (e.g., 'LT1', 'LT2').\n- `SERVICE_LEVEL`: Numeric (e.g., 0.876 or 87.6).\n- `STOCK`: Numeric (e.g., 2862.495).")
    st.subheader("Example:"); sample_data = { 'POLICY': ['BSL', 'IDL', 'BSL', 'IDL'], 'LT': ['LT1', 'LT1', 'LT2', 'LT2'], 'SERVICE_LEVEL': [0.88, 0.85, 0.90, 0.88], 'STOCK': [1000, 950, 1200, 1100] }; st.dataframe(pd.DataFrame(sample_data))
