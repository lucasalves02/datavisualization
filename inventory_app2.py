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
        # ... (Data processing logic remains the same) ...
        # Ensure input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            st.error("Input data is not a valid DataFrame."); return None
        # Check for required columns
        required_cols = {'POLICY', 'LT', 'SERVICE_LEVEL', 'STOCK'}
        if not required_cols.issubset(df.columns):
            st.error(f"Input data must contain columns: {required_cols}"); return None
        # Convert relevant columns to numeric, coercing errors
        df['SERVICE_LEVEL'] = pd.to_numeric(df['SERVICE_LEVEL'], errors='coerce')
        df['STOCK'] = pd.to_numeric(df['STOCK'], errors='coerce')
        df.dropna(subset=['SERVICE_LEVEL', 'STOCK'], inplace=True) # Drop rows where essential numeric conversions failed
        # Pivot baseline (BSL) for calculations
        if 'BSL' not in df['POLICY'].unique(): st.error("Baseline policy 'BSL' not found."); return None
        baseline_df = df[df['POLICY'] == 'BSL'].drop_duplicates(subset=['LT'])
        baseline_df = baseline_df[['LT', 'SERVICE_LEVEL', 'STOCK']].set_index('LT')
        # Merge baseline data with other policies
        merged_df = df[df['POLICY'] != 'BSL'].merge(baseline_df, on='LT', suffixes=('', '_BSL'), how='left')
        # Handle cases where merge might fail or result in NAs
        if merged_df['SERVICE_LEVEL_BSL'].isnull().any() or merged_df['STOCK_BSL'].isnull().any():
            st.warning("Some non-BSL entries dropped due to missing baseline match/value.")
            merged_df.dropna(subset=['SERVICE_LEVEL_BSL', 'STOCK_BSL'], inplace=True)
        if merged_df.empty: st.warning("No data left after merging with baseline."); return None
        # Calculate percentage differences
        merged_df['STOCK_DIFF%'] = 100 * (merged_df['STOCK'] - merged_df['STOCK_BSL']) / merged_df['STOCK_BSL'].replace({0: np.nan})
        merged_df['SERVICE_DIFF%'] = 100 * (merged_df['SERVICE_LEVEL'] - merged_df['SERVICE_LEVEL_BSL']) / merged_df['SERVICE_LEVEL_BSL'].replace({0: np.nan})
        merged_df.dropna(subset=['STOCK_DIFF%', 'SERVICE_DIFF%'], inplace=True) # Drop rows where calculation itself failed
        # Rename Lead Times using mapping
        lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        merged_df['LT_Percent'] = merged_df['LT'].map(lt_mapping).fillna(merged_df['LT'].astype(str))

        return merged_df
    except Exception as e:
        st.error(f"Error during baseline calculations: {e}"); return None

# --- Plotting Functions ---

# ++ Add fontsize parameters ++
def plot_relative_performance(merged_df, custom_title=None,
                              xlim_min=-14, xlim_max=5, ylim_min=-14, ylim_max=5,
                              figure_width=12.0, figure_height=7.0,
                              title_axis_fontsize=12, legend_fontsize=9): # New fontsize params
    """Generates the Relative Performance plot with dynamic sizes."""
    if merged_df is None or merged_df.empty: st.warning("No processed data for relative plot."); return None

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
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

        # Reference lines...
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--'); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

        # ++ Use fontsize parameters for Labels and Title ++
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Relative Performance: Stock vs Service Level Difference (%)'
        ax.set_title(plot_title, fontsize=title_axis_fontsize + 2) # Make title slightly larger than axes
        ax.set_xlabel('Inventory Difference (%) (Negative is Better)', fontsize=title_axis_fontsize)
        ax.set_ylabel('Service Level Difference (%) (Positive is Better)', fontsize=title_axis_fontsize)
        # Update tick label sizes too (optional)
        ax.tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2)

        # ++ Use fontsize parameters for Legends ++
        handles, labels = scatter.get_legend_handles_labels()
        # --- Legend Modification START ---
        # Remove original legend created by scatterplot
        scatter.legend_.remove()

        # Create SL Target Legend (Colors)
        lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in lt_labels_unique if lt in lt_color_map]
        legend1 = ax.legend(handles=lt_patches, title='SL Target (%)',
                            # Position: Centered below axes, slightly offset down
                            bbox_to_anchor=(0.5, -0.15), # X=center, Y=below plot
                            loc='upper center',         # Anchor point on legend box
                            ncol=1,                     # Display vertically
                            fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        ax.add_artist(legend1) # Add the first legend manually

        # Create Policy Legend (Markers)
        unique_policies = merged_df['POLICY'].unique(); policy_handles = []; policy_labels_for_legend = []
        # Extract unique policy handles/labels (simplified finding might be needed if complex)
        temp_handles, temp_labels = ax.get_legend_handles_labels() # Get handles created by scatterplot (before removal)
        seen_policies = set()
        for h, l in zip(handles, labels):
            # Try to determine if the label corresponds to a Policy
            # This heuristic checks if the label is one of the unique policies
            if l in unique_policies and l not in seen_policies:
                 policy_handles.append(h)
                 policy_labels_for_legend.append(l)
                 seen_policies.add(l)
            # Add more sophisticated checks if labels are complex (tuples etc.)

        if policy_handles:
            # Calculate offset for the second legend based on the first one's estimated height
            # This is an approximation and might need tuning
            # A simpler way is to just give it a fixed offset further down.
            legend2_y_offset = -0.15 - (len(lt_patches) * 0.03 + 0.06) # Heuristic offset

            ax.legend(handles=policy_handles, labels=policy_labels_for_legend, title='Policy',
                      # Position: Centered below axes, offset below the first legend
                      # bbox_to_anchor=(0.5, legend2_y_offset), # Dynamic offset (can be complex)
                      bbox_to_anchor=(0.5, -0.25), # Fixed offset below the first - adjust Y as needed
                      loc='upper center',         # Anchor point on legend box
                      ncol=1,                     # Display vertically
                      fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        # --- Legend Modification END ---

        # Manual Axes Positioning (REMOVED - Let tight_layout handle it)
        # ax.set_position([0.1, 0.1, 0.65, 0.8]) # Adjust width (0.65) if needed

        # Axis Limits & Grid...
        ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True, linestyle='--')

        # Adjust layout to prevent overlap, especially with legends at bottom
        fig.subplots_adjust(bottom=0.3) # Increase bottom margin significantly

        return fig
    except Exception as e: st.error(f"Error generating relative plot: {e}"); traceback.print_exc(); return None # Added traceback

# ++ Add fontsize parameters ++
def plot_absolute_performance(input_df, custom_title=None,
                              figure_width=12.0, figure_height=7.0,
                              title_axis_fontsize=12, legend_fontsize=9): # New fontsize params
    """Generates the Absolute Performance plots with dynamic sizes."""
    if input_df is None or input_df.empty: st.warning("No raw data for absolute plot."); return None
    df = input_df.copy()
    # Create figure with 2 subplots, maybe slightly taller to accommodate legends if needed
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height + 1)) # Increased height slightly
    try:
        # Data processing...
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

        # Plotting...
        # Plot 1: Stock
        line1 = sns.lineplot(ax=axes[0], data=df, x='LT_Label', y='STOCK', hue='POLICY', marker='o', linewidth=2)
        # ++ Apply fontsizes ++
        axes[0].set_xlabel('Service Level Target (%)', fontsize=title_axis_fontsize)
        axes[0].set_ylabel('Inventory (Units)', fontsize=title_axis_fontsize)
        axes[0].tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2)
        axes[0].grid(True);
        axes[0].set_ylim(bottom=0)
        # Remove default legend
        if axes[0].get_legend(): axes[0].get_legend().remove()

        # Plot 2: Service Level
        line2 = sns.lineplot(ax=axes[1], data=df, x='LT_Label', y='SERVICE_LEVEL', hue='POLICY', marker='o', linewidth=2)
        # ++ Apply fontsizes ++
        axes[1].set_xlabel('Service Level Target (%)', fontsize=title_axis_fontsize)
        axes[1].set_ylabel('Service Level', fontsize=title_axis_fontsize)
        axes[1].tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2)
        axes[1].grid(True);
        min_sl = df['SERVICE_LEVEL'].min(); max_sl = df['SERVICE_LEVEL'].max()
        axes[1].set_ylim(bottom=max(0, min_sl - 0.05), top=min(1.0, max_sl + 0.05))
        # Remove default legend
        if axes[1].get_legend(): axes[1].get_legend().remove()

        # --- Common Legend START ---
        # Get handles and labels from one of the plots (they should be the same)
        handles, labels = line1.get_legend_handles_labels()

        # Create a single legend for the whole figure, placed at the bottom
        fig.legend(handles=handles, labels=labels, title='Policy',
                   # Position: Centered below the subplots
                   bbox_to_anchor=(0.5, 0.05), # X=center, Y=near bottom of FIGURE
                   loc='upper center',         # Anchor point on legend box
                   ncol=1,                     # Display vertically
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        # --- Common Legend END ---


        # Figure Title & Layout
        # ++ Apply fontsize ++
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Absolute Performance Levels by Policy'
        fig.suptitle(plot_title, fontsize=title_axis_fontsize + 2, y=0.98) # Adjust title position slightly if needed

        # Adjust layout to prevent overlap and make space for the common legend
        # rect=[left, bottom, right, top] - increase bottom margin
        plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust bottom (0.1) and top (0.95)

        return fig
    except Exception as e: st.error(f"Error generating absolute plots: {e}"); traceback.print_exc(); return None # Added traceback


# ++ Add fontsize parameters ++
def plot_quadrant_analysis(merged_df, custom_title=None,
                           xlim_min=-14, xlim_max=5, ylim_min=-14, ylim_max=5,
                           stock_target_thresh=-6.0, service_target_thresh=0.0, service_floor_thresh=-10.0,
                           figure_width=12.0, figure_height=7.0,
                           title_axis_fontsize=12, legend_fontsize=9): # New fontsize params
    """Generates the Quadrant Analysis plot with dynamic sizes and desirability logic."""
    if merged_df is None or merged_df.empty: st.warning("No processed data for quadrant plot."); return None
    if service_floor_thresh < ylim_min: st.warning(f"Service floor ({service_floor_thresh}%) < Y-min ({ylim_min}%).")
    if stock_target_thresh < xlim_min: st.warning(f"Stock target ({stock_target_thresh}%) < X-min ({xlim_min}%).")

    # ++ Use parameters for figsize ++
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
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
        scatter = None # Initialize scatter
        if not plot_data_df.empty:
             scatter = sns.scatterplot( ax=ax, data=plot_data_df, x='STOCK_DIFF%', y='SERVICE_DIFF%', hue='LT_Percent', hue_order=lt_labels_unique, style='POLICY', s=200, palette=lt_color_map, edgecolor='black' )
             # --- Legend Modification START (Quadrant) ---
             if scatter and hasattr(scatter, 'legend_') and scatter.legend_:
                 scatter.legend_.remove() # Remove default combined legend
             # --- Legend Modification END (Quadrant) ---
        else: st.info("No data points to plot.")


        # Axes, Title, Legend setup...
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--'); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.add_patch(mpatches.Rectangle((xlim_min, ylim_min), xlim_max - xlim_min, ylim_max - ylim_min, hatch='///', fill=False, edgecolor='gray', linewidth=0, alpha=0.05))
        # ++ Use fontsize parameters ++
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Quadrant Analysis: Stock vs Service Level Difference (%)'
        ax.set_title(plot_title, fontsize=title_axis_fontsize + 2) # Make title slightly larger
        ax.set_xlabel('Inventory Difference (%)', fontsize=title_axis_fontsize)
        ax.set_ylabel('Service Level Difference (%)', fontsize=title_axis_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=title_axis_fontsize - 2) # Adjust tick labels too

        # ++ Use fontsize parameters for Legends ++
        if scatter: # Only create legends if scatter plot was created
             # --- Legend Modification START (Quadrant) ---
            # Create SL Target Legend (Colors)
            lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in lt_labels_unique if lt in lt_color_map]
            legend1 = ax.legend(handles=lt_patches, title='SL Target (%)',
                                # Position: Centered below axes, slightly offset down
                                bbox_to_anchor=(0.5, -0.15), # X=center, Y=below plot
                                loc='upper center',         # Anchor point on legend box
                                ncol=1,                     # Display vertically
                                fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            ax.add_artist(legend1) # Add the first legend manually

            # Create Policy Legend (Markers)
            unique_policies = plot_data_df['POLICY'].unique()
            policy_handles = []
            policy_labels_for_legend = []
            # Need to get handles/labels *after* plotting but *before* removing legend
            # Re-create dummy handles based on unique styles/markers if needed, or retrieve from scatter
            # Let's try getting from scatter *before* removal (this is tricky)
            # A more robust way is to create Line2D objects for the markers manually
            all_handles, all_labels = scatter.get_legend_handles_labels()
            seen_policies = set()
            for h, l in zip(all_handles, all_labels):
                 if l in unique_policies and l not in seen_policies:
                      policy_handles.append(h)
                      policy_labels_for_legend.append(l)
                      seen_policies.add(l)

            if policy_handles:
                # Calculate offset for the second legend
                legend2_y_offset = -0.15 - (len(lt_patches) * 0.03 + 0.06) # Heuristic offset

                ax.legend(handles=policy_handles, labels=policy_labels_for_legend, title='Policy',
                          # Position: Centered below axes, offset below the first legend
                          # bbox_to_anchor=(0.5, legend2_y_offset), # Dynamic offset
                          bbox_to_anchor=(0.5, -0.25), # Fixed offset below the first - adjust Y as needed
                          loc='upper center',         # Anchor point on legend box
                          ncol=1,                     # Display vertically
                          fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            # --- Legend Modification END (Quadrant) ---

        # Manual Axes Positioning (REMOVED - Let tight_layout handle it)
        # ax.set_position([0.1, 0.1, 0.65, 0.8]) # Adjust width (0.65) if needed

        # Axis Limits & Grid...
        ax.set_xlim(xlim_min, xlim_max); ax.set_ylim(ylim_min, ylim_max)
        ax.grid(True, linestyle='--')

        # Adjust layout to prevent overlap, especially with legends at bottom
        fig.subplots_adjust(bottom=0.3) # Increase bottom margin significantly

        return fig
    except Exception as e: st.error(f"Error generating quadrant plot: {e}"); traceback.print_exc(); return None # Added traceback


# --- Streamlit App Layout ---
# [ Rest of your Streamlit app code remains the same ]
# ... (Keep the Streamlit UI code as it was) ...

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
                st.session_state['merged_df'] = perform_baseline_calculations(st.session_state['raw_df'].copy()) # Pass copy
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
            st.session_state['merged_df'] = perform_baseline_calculations(st.session_state['raw_df'].copy()) # Pass copy
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
    # Ensure raw_df exists and is a DataFrame before displaying
    if isinstance(st.session_state.get('raw_df'), pd.DataFrame):
         st.dataframe(st.session_state['raw_df'].head(), height=200)
    else:
         st.warning("Raw data is not available or invalid.")


    st.header("Analysis Plots")
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
    with size_col1: fig_width = st.slider("Width:", min_value=6.0, max_value=20.0, value=12.0, step=0.5, key="fig_width_slider", format="%.1f")
    # Adjusted default height to better accommodate bottom legends
    with size_col2: fig_height = st.slider("Height:", min_value=5.0, max_value=15.0, value=8.0, step=0.5, key="fig_height_slider", format="%.1f")

    # ++ Row 3: Font Size Controls ++
    st.subheader("Font Sizes")
    font_col1, font_col2 = st.columns(2)
    with font_col1: title_axis_font_size = st.slider("Title & Axis Label Size:", min_value=6, max_value=20, value=12, step=1, key="title_axis_font_slider")
    with font_col2: legend_font_size = st.slider("Legend Size:", min_value=5, max_value=18, value=9, step=1, key="legend_font_slider")
    # ++ End Font Size Controls ++

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
        # ++ Pass Font Size parameters to ALL plot functions ++
        if plot_choice == "Relative Performance (vs Baseline)":
            # Ensure merged_df exists and is DataFrame
            merged_df_state = st.session_state.get('merged_df')
            if isinstance(merged_df_state, pd.DataFrame) and not merged_df_state.empty:
                if limits_valid:
                    st.subheader("Relative Performance")
                    fig_to_show = plot_relative_performance(merged_df_state, custom_title=user_title,
                                xlim_min=x_min_limit, xlim_max=x_max_limit, ylim_min=y_min_limit, ylim_max=y_max_limit,
                                figure_width=fig_width, figure_height=fig_height,
                                title_axis_fontsize=title_axis_font_size, legend_fontsize=legend_font_size) # Pass fonts
                else: plot_error = True
            elif merged_df_state is None:
                st.warning("Data not yet processed for relative plot. Load data first.")
            else: # It exists but might be empty after processing
                 st.warning("No data available for relative plot after processing.")


        elif plot_choice == "Absolute Performance":
             # Ensure raw_df exists and is DataFrame
            raw_df_state = st.session_state.get('raw_df')
            if isinstance(raw_df_state, pd.DataFrame) and not raw_df_state.empty:
                st.subheader("Absolute Performance")
                fig_to_show = plot_absolute_performance(raw_df_state, custom_title=user_title,
                                figure_width=fig_width, figure_height=fig_height,
                                title_axis_fontsize=title_axis_font_size, legend_fontsize=legend_font_size) # Pass fonts
            else:
                 st.warning("Raw data not loaded or is empty.")


        elif plot_choice == "Quadrant Analysis":
            # Ensure merged_df exists and is DataFrame
            merged_df_state = st.session_state.get('merged_df')
            if isinstance(merged_df_state, pd.DataFrame) and not merged_df_state.empty:
                if limits_valid:
                    st.subheader("Quadrant Analysis")
                    fig_to_show = plot_quadrant_analysis(merged_df_state, custom_title=user_title,
                                xlim_min=x_min_limit, xlim_max=x_max_limit, ylim_min=y_min_limit, ylim_max=y_max_limit,
                                stock_target_thresh=stock_target_thresh_param, service_target_thresh=service_target_thresh_param, service_floor_thresh=service_floor_thresh_param,
                                figure_width=fig_width, figure_height=fig_height,
                                title_axis_fontsize=title_axis_font_size, legend_fontsize=legend_font_size) # Pass fonts
                else: plot_error = True
            elif merged_df_state is None:
                 st.warning("Data not yet processed for quadrant plot. Load data first.")
            else: # It exists but might be empty after processing
                 st.warning("No data available for quadrant plot after processing.")


        # Display the figure
        if fig_to_show: st.pyplot(fig_to_show)
        elif not plot_error and 'raw_df' in st.session_state: st.warning("Could not generate plot with current settings or data.")

    except Exception as display_e:
         st.error(f"An unexpected error occurred trying to display plot: {display_e}")
         st.exception(display_e) # Show full traceback in Streamlit


# Initial State Message
else:
    st.info("Welcome! Please load data using the sidebar options (Paste or Upload).")
    st.markdown("""
        This dashboard helps analyze different inventory policies against a baseline ('BSL').
        - **Relative Performance:** Shows % difference in Stock and Service Level vs BSL. Lower-left is generally better (less stock, similar/better service).
        - **Absolute Performance:** Shows raw Stock and Service Level values per policy across different Service Level Targets (derived from 'LT' column).
        - **Quadrant Analysis:** Similar to Relative, but with a background heatmap indicating 'desirability' based on achieving stock reduction targets while maintaining service levels.
    """)
    st.markdown("---"); st.subheader("Expected Data Format")
    st.markdown("- `POLICY`: Text (e.g., 'BSL', 'IDL'). **Must include 'BSL'** for baseline calculations.\n- `LT`: Text mapping to Service Level Targets (e.g., 'LT1', 'LT2').\n- `SERVICE_LEVEL`: Numeric (e.g., 0.876 or 87.6).\n- `STOCK`: Numeric (e.g., 2862.495).")
    st.subheader("Example:"); sample_data = { 'POLICY': ['BSL', 'IDL', 'BSL', 'IDL'], 'LT': ['LT1', 'LT1', 'LT2', 'LT2'], 'SERVICE_LEVEL': [0.88, 0.85, 0.90, 0.88], 'STOCK': [1000, 950, 1200, 1100] }; st.dataframe(pd.DataFrame(sample_data))
