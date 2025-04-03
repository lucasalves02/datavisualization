# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:31:44 2025

@author: lucas
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io # Potentially needed for error handling or alternative input
import traceback

# --- Configuration & Styling (Optional) ---
st.set_page_config(layout="wide") # Use wider layout
# You can add custom CSS here if needed

# --- Helper Functions (Refactored from your notebook cells) ---

def perform_baseline_calculations(df):
    """Takes the raw DataFrame and returns the merged_df with diff %."""
    try:
        # Check for required columns
        required_cols = {'POLICY', 'LT', 'SERVICE_LEVEL', 'STOCK'}
        if not required_cols.issubset(df.columns):
            st.error(f"Input data must contain columns: {required_cols}")
            return None

        # Pivot baseline (BSL) for calculations
        if 'BSL' not in df['POLICY'].unique():
            st.error("Baseline policy 'BSL' not found in the 'POLICY' column.")
            return None
        baseline_df = df[df['POLICY'] == 'BSL'][['LT', 'SERVICE_LEVEL', 'STOCK']].set_index('LT')

        # Merge baseline data with other policies
        merged_df = df[df['POLICY'] != 'BSL'].merge(baseline_df, on='LT', suffixes=('', '_BSL'), how='left') # Use left merge to keep all non-BSL rows

        # Handle cases where merge might fail for some LT if BSL doesn't have all LTs
        if merged_df['SERVICE_LEVEL_BSL'].isnull().any() or merged_df['STOCK_BSL'].isnull().any():
             st.warning("Some non-BSL entries didn't find a matching 'LT' in the baseline 'BSL' data. Calculations might be incomplete.")
             # Optionally drop rows with missing baseline data or handle differently
             merged_df.dropna(subset=['SERVICE_LEVEL_BSL', 'STOCK_BSL'], inplace=True)

        # Calculate percentage differences
        # Avoid division by zero if baseline stock is 0
        merged_df['STOCK_DIFF%'] = 100 * (merged_df['STOCK'] - merged_df['STOCK_BSL']) / merged_df['STOCK_BSL'].replace({0: np.nan})
        merged_df['SERVICE_DIFF%'] = 100 * (merged_df['SERVICE_LEVEL'] - merged_df['SERVICE_LEVEL_BSL']) / merged_df['SERVICE_LEVEL_BSL'].replace({0: np.nan})
        merged_df.dropna(subset=['STOCK_DIFF%', 'SERVICE_DIFF%'], inplace=True) # Drop rows where calculation failed

        # Rename Lead Times
        lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        # Handle potential missing LT values gracefully
        merged_df['LT_Percent'] = merged_df['LT'].map(lt_mapping).fillna(merged_df['LT']) # Keep original LT if not in mapping


        return merged_df

    except Exception as e:
        st.error(f"Error during calculations: {e}")
        return None


def plot_relative_performance(merged_df,custom_title=None):
    """Generates the plot from Cell 1."""
    if merged_df is None or merged_df.empty:
        st.warning("Cannot generate relative performance plot: No processed data available.")
        return None

    fig, ax = plt.subplots(figsize=(10, 7)) # Create fig and ax

    try:
        # Define distinct colors for SL Targets (ensure enough colors if more than 5 LTs)
        lt_labels_unique = merged_df['LT_Percent'].unique()
        # Use a standard colormap if too many LTs, or define more colors
        num_colors_needed = len(lt_labels_unique)
        if num_colors_needed <= 5:
             distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        else:
             # Generate colors from a colormap if more are needed
             cmap = plt.get_cmap('tab10') # Or another suitable colormap like 'viridis'
             distinct_colors = [cmap(i) for i in np.linspace(0, 1, num_colors_needed)]

        lt_color_map = dict(zip(lt_labels_unique, distinct_colors[:num_colors_needed]))


        # Heatmap range - adjust dynamically or keep fixed
        x_min, x_max = merged_df['STOCK_DIFF%'].min() - 1, merged_df['STOCK_DIFF%'].max() + 1
        y_min, y_max = merged_df['SERVICE_DIFF%'].min() - 1, merged_df['SERVICE_DIFF%'].max() + 1
        x_min = min(x_min, -14) # Ensure minimum range if desired
        x_max = max(x_max, 1)
        y_min = min(y_min, -14)
        y_max = max(y_max, 1)

        x = np.linspace(x_min, x_max, 200)
        y = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x, y)
        Z = Y - X # Desirability heatmap

        ax.contourf(X, Y, Z, levels=100, cmap='RdYlGn', alpha=0.6)

        # Scatter plot
        scatter = sns.scatterplot(
            ax=ax, # Pass the ax object
            data=merged_df,
            x='STOCK_DIFF%', y='SERVICE_DIFF%',
            hue='LT_Percent', style='POLICY',
            s=200,
            palette=lt_color_map, # Use the defined color map
            edgecolor='black'
        )

        # Reference lines
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

        # Labels
        # OLD: ax.set_title('Service Level and Inventory (%) Difference', fontsize=14)
        # NEW: Determine title to use
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Service Level and Inventory (%) Difference'
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel('Inventory Difference (%) (negative is better)', fontsize=12)
        ax.set_ylabel('Service Level Difference (%) (positive is better)', fontsize=12)

        # === Custom Legends ===
        handles, labels = scatter.get_legend_handles_labels()
        policy_labels = merged_df['POLICY'].unique()

        # Separate handles/labels for color (LT) and style (POLICY)
        # Legend needs careful reconstruction because seaborn combines hue and style
        unique_lts = merged_df['LT_Percent'].unique()
        unique_policies = merged_df['POLICY'].unique()

        lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in unique_lts]

        # Recreate policy handles (find representative marker for each policy)
        policy_handles_dict = {}
        for h, l in zip(handles, labels):
             # Seaborn labels might be tuples (LT, POLICY), need to extract POLICY
             current_policy = l # Assuming label directly corresponds to POLICY based on 'style' usage
             if isinstance(l, tuple): # If seaborn creates tuple labels
                 current_policy = l[1] # Adjust index if needed

             if current_policy in unique_policies and current_policy not in policy_handles_dict:
                  policy_handles_dict[current_policy] = h

        policy_handles_list = [policy_handles_dict[p] for p in unique_policies if p in policy_handles_dict]
        policy_labels_list = [p for p in unique_policies if p in policy_handles_dict]


        # Add custom legends outside the plot
        legend1 = ax.legend(handles=lt_patches, title='SL Target', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.add_artist(legend1)
        ax.legend(handles=policy_handles_list, labels=policy_labels_list, title='Policy', bbox_to_anchor=(1.05, 0.55), loc='upper left')


        ax.grid(True, linestyle='--')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legends

        return fig # Return the figure object

    except Exception as e:
        st.error(f"Error generating relative performance plot: {e}")
        return None

def plot_absolute_performance(input_df,custom_title=None): # Renamed parameter for clarity
    """Generates the two plots from Cell 2. Works on a copy of the input DataFrame."""
    if input_df is None or input_df.empty:
        st.warning("Cannot generate absolute performance plots: No raw data available.")
        return None

    # +++ Create a copy to work with +++
    df = input_df.copy()
    # +++++++++++++++++++++++++++++++++++

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    try:
        # Ensure correct data types for plotting
        df['STOCK'] = pd.to_numeric(df['STOCK'], errors='coerce')
        df['SERVICE_LEVEL'] = pd.to_numeric(df['SERVICE_LEVEL'], errors='coerce')
        # Drop rows where conversion failed for essential columns
        df.dropna(subset=['POLICY', 'LT', 'STOCK', 'SERVICE_LEVEL'], inplace=True) # inplace=True modifies the copy 'df'

        if df.empty:
             st.warning("No valid data remaining after cleaning numeric columns.")
             return None

        # --- Refined Category Handling ---
        # Define the desired order
        lt_full_order = ['LT1', 'LT2', 'LT3', 'LT4', 'LT5']
        # Find which of the desired categories actually exist in the current data
        existing_lts_in_order = [lt for lt in lt_full_order if lt in df['LT'].unique()]

        if not existing_lts_in_order:
             st.warning("No standard LT values (LT1-LT5) found for sorting. Using unique values found.")
             # Use unique values found, trying to sort them alphanumerically as a fallback
             lt_categories = sorted(df['LT'].unique())
        else:
             lt_categories = existing_lts_in_order # Use the subset in the predefined order

        # Convert 'LT' column in the *copy* to Categorical
        df['LT'] = pd.Categorical(df['LT'], categories=lt_categories, ordered=True)

        # Sort the DataFrame based on the new categorical 'LT' column
        df = df.sort_values('LT') # This also modifies the copy 'df'

        # Map LT to Labels for the plot's x-axis
        lt_mapping = {'LT1': '88%', 'LT2': '90%', 'LT3': '92%', 'LT4': '94%', 'LT5': '96%'}
        # Create the LT_Label column. Map known LTs, keep others as is.
        df['LT_Label'] = df['LT'].map(lt_mapping)
        df['LT_Label'].fillna(df['LT'].astype(str), inplace=True) # Fill missing mappings with original LT value (as string)

        # Ensure the LT_Label is also categorical for correct plot axis order
        label_order = [lt_mapping.get(cat, str(cat)) for cat in lt_categories]
        df['LT_Label'] = pd.Categorical(df['LT_Label'], categories=label_order, ordered=True)
        # --- End Refined Category Handling ---


        # Check if data remains after processing
        if df.empty:
            st.warning("Data became empty after processing categories. Cannot plot.")
            return None

        # Plotting
        sns.lineplot(ax=axes[0], data=df, x='LT_Label', y='STOCK', hue='POLICY', marker='o', linewidth=2)
        #Remove axes[0].set_title('Absolute Inventory Levels by Policy')
        axes[0].set_xlabel('Service Level Target (%)')
        axes[0].set_ylabel('Inventory (Units)')
        axes[0].grid(True)
        axes[0].legend(title='Policy')

        sns.lineplot(ax=axes[1], data=df, x='LT_Label', y='SERVICE_LEVEL', hue='POLICY', marker='o', linewidth=2)
        #Remove axes[1].set_title('Absolute Service Levels by Policy')
        axes[1].set_xlabel('Service Level Target (%)')
        axes[1].set_ylabel('Service Level')
        axes[1].grid(True)
        axes[1].legend(title='Policy')
        # ADD before plt.tight_layout()
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Absolute Performance Levels by Policy'
        fig.suptitle(plot_title, fontsize=16, y=1.02) # y adjusts vertical position if needed

        # Adjust layout slightly to accommodate suptitle if necessary
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # May need slight adjustment [left, bottom, right, top]
        # OLD: plt.tight_layout()
        return fig # Return the figure object

    except Exception as e:
        st.error(f"Error generating absolute performance plots: {e}")
        # For more detailed debugging, you might want to print the traceback in your console
        import traceback
        print(traceback.format_exc()) # Print detailed error to console where streamlit runs
        return None


def plot_quadrant_analysis(merged_df, custom_title=None):
    """Generates the plot from Cell 3."""
    if merged_df is None or merged_df.empty:
        st.warning("Cannot generate quadrant analysis plot: No processed data available.")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    try:
        # Filter data for the scatter plot (negative quadrant)
        filtered_df = merged_df[(merged_df['SERVICE_DIFF%'] < 0) & (merged_df['STOCK_DIFF%'] < 0)].copy() # Use .copy()

        if filtered_df.empty:
            st.info("No data points fall in the negative quadrant (Stock Diff % < 0 and Service Diff % < 0).")
            # Optionally still show the heatmap without scatter points
        else:
             st.info(f"Plotting {len(filtered_df)} points in the negative quadrant.")


        # Define distinct colors (same logic as relative plot)
        lt_labels_unique = merged_df['LT_Percent'].unique()
        num_colors_needed = len(lt_labels_unique)
        if num_colors_needed <= 5:
             distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        else:
             cmap = plt.get_cmap('tab10')
             distinct_colors = [cmap(i) for i in np.linspace(0, 1, num_colors_needed)]
        lt_color_map = dict(zip(lt_labels_unique, distinct_colors[:num_colors_needed]))

        # --- Heatmap Calculation ---
        # Use data ranges from the full merged_df for consistency or from filtered_df
        x_min, x_max = merged_df['STOCK_DIFF%'].min() -1, merged_df['STOCK_DIFF%'].max() + 1
        y_min, y_max = merged_df['SERVICE_DIFF%'].min() - 1, merged_df['SERVICE_DIFF%'].max() + 1
        # Keep desired negative focus if needed, but allow seeing all data context
        x_min = min(x_min, -14); x_max = max(x_max, 1)
        y_min = min(y_min, -14); y_max = max(y_max, 1)

        grid_size = 300
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Desirability logic
        stock_score = np.zeros_like(X)
        stock_score[X <= -6] = 1
        mask = (X > -6) & (X <= 0)
        stock_score[mask] = (np.abs(X[mask]) / 6)
        stock_score = np.clip(stock_score, 0, 1)

        service_score = np.zeros_like(Y)
        # Adjust service score: better if Y is close to 0 or positive
        service_score[Y >= 0] = 1 # Full score if service level improved or stayed same
        mask_neg = (Y < 0) & (Y >= y_min)
        # Score decreases as Y becomes more negative. Scale from 0 at y_min to 1 at 0.
        service_score[mask_neg] = 1 - (np.abs(Y[mask_neg]) / np.abs(y_min)) # Assumes y_min is the worst case
        service_score = np.clip(service_score, 0, 1)


        Z = stock_score * service_score # Combine scores

        # Plot heatmap
        img = ax.imshow(Z, extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap='RdYlGn', aspect='auto', alpha=0.6)

        # --- Overlay scatterplot (only if filtered_df is not empty) ---
        if not filtered_df.empty:
             scatter = sns.scatterplot(
                 ax=ax,
                 data=filtered_df, # Use filtered data
                 x='STOCK_DIFF%', y='SERVICE_DIFF%',
                 hue='LT_Percent', style='POLICY', s=200,
                 palette=lt_color_map, # Use same colors
                 edgecolor='black'
             )
        else:
             scatter = None # No scatter object if no points


        # Quadrant axes
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

        # Hatched pattern (optional styling)
        ax.add_patch(
            mpatches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                hatch='///', fill=False, edgecolor='black', linewidth=0, alpha=0.05
            )
        )

        # Axes and legend
        # OLD: ax.set_title('Quadrant Analysis: Focus on Negative Differences', fontsize=14)
        # NEW: Determine title to use
        plot_title = custom_title if (custom_title and custom_title.strip()) else 'Quadrant Analysis: Focus on Negative Differences'
        ax.set_title(plot_title, fontsize=14)
        ax.set_xlabel('Inventory Difference (%)', fontsize=12)
        ax.set_ylabel('Service Level Difference (%)', fontsize=12)

        # === Custom Legends (only if scatter plot was created) ===
        if scatter:
             handles, labels = scatter.get_legend_handles_labels()
             unique_lts = filtered_df['LT_Percent'].unique() # Use LTs present in filtered data
             unique_policies = filtered_df['POLICY'].unique() # Use Policies present in filtered data

             lt_patches = [mpatches.Patch(color=lt_color_map[lt], label=lt) for lt in unique_lts if lt in lt_color_map]

             policy_handles_dict = {}
             for h, l in zip(handles, labels):
                 current_policy = l
                 if isinstance(l, tuple):
                     current_policy = l[1]
                 if current_policy in unique_policies and current_policy not in policy_handles_dict:
                     policy_handles_dict[current_policy] = h

             policy_handles_list = [policy_handles_dict[p] for p in unique_policies if p in policy_handles_dict]
             policy_labels_list = [p for p in unique_policies if p in policy_handles_dict]

             legend1 = ax.legend(handles=lt_patches, title='SL Target', bbox_to_anchor=(1.05, 1), loc='upper left')
             ax.add_artist(legend1)
             ax.legend(handles=policy_handles_list, labels=policy_labels_list, title='Policy', bbox_to_anchor=(1.05, 0.55), loc='upper left')


        ax.grid(True, linestyle='--')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

        return fig

    except Exception as e:
        st.error(f"Error generating quadrant analysis plot: {e}")
        return None


# --- Streamlit App Layout ---

st.title("Inventory Policy Analysis Dashboard")

st.sidebar.header("Data Input")

# --- Method 1: Paste into Text Area (Works when Deployed) ---
st.sidebar.subheader("Paste Data from Excel")
st.sidebar.info("Copy your data range from Excel (including headers), then paste into the text area below and click 'Load'.")
pasted_text = st.sidebar.text_area("Paste tab-separated data here:", height=150, key="pasted_text_area")
if st.sidebar.button("Load Data from Text Area", key="load_text_button"):
    if pasted_text:
        try:
            string_io = io.StringIO(pasted_text)
            # Use read_csv, assuming tab separation from Excel copy/paste
            text_data = pd.read_csv(string_io, sep='\t', header=0, engine='python', on_bad_lines='warn')
            if text_data is not None and not text_data.empty:
                st.session_state['raw_df'] = text_data
                # Perform calculations immediately after loading if desired
                st.session_state['merged_df'] = perform_baseline_calculations(st.session_state['raw_df'])
                st.sidebar.success(f"Successfully loaded {len(text_data)} rows from text area.")
                # Clear the text area after successful load (optional)
                st.session_state.pasted_text_area = ""
            else:
                 st.sidebar.error("Could not parse data from text area.")
                 # Clear potentially outdated data if load fails
                 if 'raw_df' in st.session_state: del st.session_state['raw_df']
                 if 'merged_df' in st.session_state: del st.session_state['merged_df']
        except Exception as e:
            st.sidebar.error(f"Error parsing text area data: {e}")
            if 'raw_df' in st.session_state: del st.session_state['raw_df']
            if 'merged_df' in st.session_state: del st.session_state['merged_df']
    else:
        st.sidebar.warning("Text area is empty.")

# --- REMOVED Direct Clipboard Button ---
# The following button and logic relying on pd.read_clipboard()
# will NOT work in deployed environments and should be removed.
#
# st.sidebar.markdown("---")
# st.sidebar.info("Copy your data range from Excel (including headers), then click the button below.")
# if st.sidebar.button("Load Data from Clipboard"):
#     try:
#         clipboard_data = pd.read_clipboard(header=0, sep='\t', engine='python', on_bad_lines='warn')
#         # ... (rest of clipboard logic) ...
#     except Exception as e:
#         # This is where the Pyperclip error occurs when deployed
#         st.sidebar.error(f"Error reading from clipboard: {e}. Is data copied correctly (tab-separated)?")
#         # ...

st.sidebar.markdown("---") # Separator

# --- Method 2: File Uploader (Optional but Recommended) ---
st.sidebar.subheader("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Upload an Excel (.xlsx) or CSV (.csv) file:",
    type=["csv", "xlsx", "xls"] # Specify allowed file types
)
if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            file_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Might need to install openpyxl: pip install openpyxl
            # Add 'openpyxl' to requirements.txt if using Excel upload
            file_data = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            # Should not happen due to 'type' restriction, but good practice
             st.sidebar.error("Unsupported file type.")
             file_data = None

        if file_data is not None and not file_data.empty:
            st.session_state['raw_df'] = file_data
            # Perform calculations immediately after loading if desired
            st.session_state['merged_df'] = perform_baseline_calculations(st.session_state['raw_df'])
            st.sidebar.success(f"Successfully loaded {len(file_data)} rows from file: {uploaded_file.name}")
        elif file_data is not None: # Handle case where file is empty but read ok
             st.sidebar.warning(f"Uploaded file '{uploaded_file.name}' appears to be empty.")
             if 'raw_df' in st.session_state: del st.session_state['raw_df']
             if 'merged_df' in st.session_state: del st.session_state['merged_df']

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        if 'raw_df' in st.session_state: del st.session_state['raw_df']
        if 'merged_df' in st.session_state: del st.session_state['merged_df']


# --- Main Display Area ---
if 'raw_df' in st.session_state:
    st.header("Loaded Data Preview")
    # Limit preview height if desired
    st.dataframe(st.session_state['raw_df'].head(), height=200)

    st.header("Analysis Plots")
    st.markdown("---") # Add a horizontal rule

    # --- Plot Selection ---
    plot_choice = st.selectbox(
        "Choose Plot:",
        ["Relative Performance (vs Baseline)", "Absolute Performance", "Quadrant Analysis"],
        key="plot_select" # Using a key is good practice
    )

    # +++ Add Text Input for Custom Title +++
    custom_title_input = st.text_input(
        "Enter custom graph title (optional):",
        placeholder="Leave empty to use default title", # Placeholder text
        key="custom_title_input"
    )
    # Get the title from the input, treat empty or whitespace-only as None
    user_title = custom_title_input.strip() if custom_title_input else None
    # ++++++++++++++++++++++++++++++++++++++++++


    # --- Display Chosen Plot ---
    if plot_choice == "Relative Performance (vs Baseline)":
        if 'merged_df' in st.session_state:
            st.subheader("Relative Performance")
            # Pass the user_title to the plotting function
            fig1 = plot_relative_performance(st.session_state['merged_df'], custom_title=user_title)
            if fig1:
                st.pyplot(fig1)
            else:
                 st.warning("Could not generate relative performance plot.") # Handle function returning None
        else:
            st.warning("Baseline calculations failed or data not loaded. Cannot show relative plot.")

    elif plot_choice == "Absolute Performance":
        st.subheader("Absolute Performance")
        # Pass the user_title to the plotting function
        fig2 = plot_absolute_performance(st.session_state['raw_df'], custom_title=user_title)
        if fig2:
            st.pyplot(fig2)
        else:
            st.warning("Could not generate absolute performance plot.") # Handle function returning None


    elif plot_choice == "Quadrant Analysis":
         if 'merged_df' in st.session_state:
            st.subheader("Quadrant Analysis")
            # Pass the user_title to the plotting function
            fig3 = plot_quadrant_analysis(st.session_state['merged_df'], custom_title=user_title)
            if fig3:
                st.pyplot(fig3)
            else:
                st.warning("Could not generate quadrant analysis plot.") # Handle function returning None
         else:
            st.warning("Baseline calculations failed or data not loaded. Cannot show quadrant plot.")

else:
    st.info("Please load data using the sidebar options.")
    st.markdown("---")
    st.subheader("Expected Data Format")
    st.markdown("""
    The application expects data with at least the following columns:
    - `POLICY`: Text identifier for the policy (e.g., 'BSL', 'IDL'). Must include 'BSL' for baseline calculations.
    - `LT`: Text identifier for the lead time scenario (e.g., 'LT1', 'LT2').
    - `SERVICE_LEVEL`: Numeric service level achieved (e.g., 0.876).
    - `STOCK`: Numeric stock level (e.g., 2862.495).

    Copy the range directly from Excel, including the header row.
    """)
    # Display the sample data structure as a guide
    st.subheader("Example Data Structure:")
    sample_data = {
         'POLICY': ['BSL', 'IDL', 'BSL', 'IDL'],
         'LT': ['LT1', 'LT1', 'LT2', 'LT2'],
         'SERVICE_LEVEL': [0.88, 0.85, 0.90, 0.88],
         'STOCK': [1000, 950, 1200, 1100]
    }
    st.dataframe(pd.DataFrame(sample_data))
