import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Stop and Search Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Police Stop and Search Analysis by Force Area")
st.markdown("Interactive analysis of stop and search rates by ethnicity and police force area")

# Load master dataframes from CSV files
@st.cache_data
def load_data():
    """Load the pre-generated master dataframes from CSV files"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        # Load the master dataframes
        sns_path = os.path.join(data_dir, 'master_sns_df.csv')
        census_path = os.path.join(data_dir, 'master_census_df.csv')
        
        if not os.path.exists(sns_path) or not os.path.exists(census_path):
            st.error("Data files not found. Please run the data generation script first.")
            st.info("Run: python generate_data.py")
            st.stop()
        
        # Load dataframes
        master_sns_df = pd.read_csv(sns_path)
        master_census_df = pd.read_csv(census_path)
        
        # Convert year_month back to datetime
        master_sns_df['year_month'] = pd.to_datetime(master_sns_df['year_month'])
        
        return master_sns_df, master_census_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the data generation script has been run successfully.")
        st.stop()

# Helper function to filter data by ethnicity
def filter_data_by_ethnicity(data, ethnicity_selection):
    """Filter dataframe by ethnicity selection"""
    if ethnicity_selection == "All":
        return data
    elif ethnicity_selection in ["Asian", "Black", "Mixed", "White", "Other"]:
        return data[data['ethnicity_simple'] == ethnicity_selection]
    else:
        return data[data['ethnicity_full'] == ethnicity_selection]

# Function to calculate rates by police force area
def calculate_pfa_ethnicity_rates(sns_df, census_df, start_date=None, end_date=None, ethnicity_selection="All"):
    """Calculate stop and search rates per 100,000 people by police force area and ethnicity"""
    
    # Filter by date range
    filtered_sns = sns_df.copy()
    if start_date is not None:
        filtered_sns = filtered_sns[filtered_sns['year_month'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        filtered_sns = filtered_sns[filtered_sns['year_month'] <= pd.to_datetime(end_date)]
    
    # Filter by ethnicity
    sns_filtered = filter_data_by_ethnicity(filtered_sns, ethnicity_selection)
    census_filtered = filter_data_by_ethnicity(census_df, ethnicity_selection)
    
    # Aggregate data by police force
    sns_counts = sns_filtered.groupby(['police_force_code', 'police_force_name'])['sns_count'].sum().reset_index()
    pop_counts = census_filtered.groupby(['police_force_code', 'police_force_name'])['census_pop'].sum().reset_index()
    
    # Get all police force areas and merge data
    all_pfas = census_df[['police_force_code', 'police_force_name']].drop_duplicates()
    result_df = all_pfas.merge(sns_counts, on=['police_force_code', 'police_force_name'], how='left')
    result_df = result_df.merge(pop_counts, on=['police_force_code', 'police_force_name'], how='left')
    
    # Fill missing values and calculate rate per 100k
    result_df[['sns_count', 'census_pop']] = result_df[['sns_count', 'census_pop']].fillna(0)
    result_df['rate_per_100k'] = result_df['sns_count'] / result_df['census_pop'] * 100000
    result_df['rate_per_100k'] = result_df['rate_per_100k'].fillna(0)
    
    return result_df

# Function to add labels to bars
def add_bar_labels(ax, bars, values):
    """Add value labels on top of bars"""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label_y = height + max(abs(height) * 0.02, 0.5) if height >= 0 else height - max(abs(height) * 0.02, 0.5)
        ax.text(bar.get_x() + bar.get_width()/2., label_y, f'{value:.0f}', 
               ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

# Function to create comparison plot (two ethnicities)
def create_comparison_plot(ax, primary_stats, comp_stats, primary_ethnicity, comparator_ethnicity):
    """Create plot comparing rates between two ethnicities"""
    
    # Merge data and calculate differences
    merged_stats = primary_stats[['police_force_name', 'rate_per_100k']].merge(
        comp_stats[['police_force_name', 'rate_per_100k']], 
        on='police_force_name', suffixes=('_primary', '_comparator'), how='outer'
    ).fillna(0)
    
    merged_stats['delta_rate'] = merged_stats['rate_per_100k_primary'] - merged_stats['rate_per_100k_comparator']
    merged_stats = merged_stats.sort_values('delta_rate', ascending=False)
    
    # Create bars with colors based on positive/negative differences
    colors = ['darkred' if delta > 0 else 'darkblue' for delta in merged_stats['delta_rate']]
    bars = ax.bar(range(len(merged_stats)), merged_stats['delta_rate'], color=colors)
    
    # Add horizontal line at zero and labels
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    add_bar_labels(ax, bars, merged_stats['delta_rate'])
    
    # Set up axis labels and formatting
    ax.set_xticks(range(len(merged_stats)))
    ax.set_xticklabels(merged_stats['police_force_name'], rotation=45, ha='right')
    ax.set_ylabel(f'Rate Difference per 100,000 people\n(Positive = Higher {primary_ethnicity} rate)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', label=f'Higher {primary_ethnicity} rate'),
        Patch(facecolor='darkblue', label=f'Higher {comparator_ethnicity} rate')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return f"Difference in stop and search rates per 100,000 people by Police Force Area\n({primary_ethnicity} minus {comparator_ethnicity})"

# Function to create single ethnicity plot
def create_single_plot(ax, primary_stats, primary_ethnicity):
    """Create plot for single ethnicity rates"""
    
    primary_stats = primary_stats.sort_values('rate_per_100k', ascending=False)
    
    # Create bars
    bars = ax.bar(range(len(primary_stats)), primary_stats['rate_per_100k'], color='navy')
    add_bar_labels(ax, bars, primary_stats['rate_per_100k'])
    
    # Set up axis labels and formatting
    ax.set_xticks(range(len(primary_stats)))
    ax.set_xticklabels(primary_stats['police_force_name'], rotation=45, ha='right')
    ax.set_ylabel('Rate per 100,000 people')
    
    # Set y-axis limits
    if len(primary_stats) > 0:
        ax.set_ylim(0, primary_stats['rate_per_100k'].max() * 1.15)
    
    return f"Stop and search rate per 100,000 people by Police Force Area ({primary_ethnicity})"

# Function to apply common plot styling
def style_plot(ax):
    """Apply consistent styling to plots"""
    # Remove spines and add grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis='y', linestyle='-', alpha=0.3, color='lightgray')
    ax.grid(False, axis='x')
    ax.set_xlabel('')

# Main app logic
def main():
    # Load data
    try:
        sns_df, census_df = load_data()
    except:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Get ethnicity options for dropdowns
    simple_ethnicities = ['All'] + sorted([e for e in census_df['ethnicity_simple'].unique() if e != 'Unknown'])
    detailed_ethnicities = sorted([e for e in census_df['ethnicity_full'].unique() if e != 'Unknown'])
    ethnicity_options = simple_ethnicities + detailed_ethnicities
    
    # Create sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    # Ethnicity selection
    primary_ethnicity = st.sidebar.selectbox(
        "Primary Ethnicity:",
        options=ethnicity_options,
        index=ethnicity_options.index('Black') if 'Black' in ethnicity_options else 0
    )
    
    comparator_ethnicity = st.sidebar.selectbox(
        "Comparator Ethnicity:",
        options=['None'] + ethnicity_options,
        index=ethnicity_options.index('White') + 1 if 'White' in ethnicity_options else 0
    )
    
    # Date selection
    start_date = st.sidebar.date_input(
        "Start Date:",
        value=pd.to_datetime('2022-04-01').date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date:",
        value=pd.to_datetime('2023-03-31').date()
    )
    
    # Generate plot when button is clicked or automatically
    if st.sidebar.button("Update Plot") or True:  # Auto-update
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                # Calculate rates for primary ethnicity
                primary_stats = calculate_pfa_ethnicity_rates(
                    sns_df, census_df, start_date, end_date, primary_ethnicity
                )
                
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(16, 10))
                
                # Create appropriate plot type
                if comparator_ethnicity != 'None':
                    comp_stats = calculate_pfa_ethnicity_rates(
                        sns_df, census_df, start_date, end_date, comparator_ethnicity
                    )
                    title_text = create_comparison_plot(ax, primary_stats, comp_stats, primary_ethnicity, comparator_ethnicity)
                else:
                    title_text = create_single_plot(ax, primary_stats, primary_ethnicity)
                
                # Apply styling and finalize plot
                style_plot(ax)
                title_text += f"\n{pd.to_datetime(start_date).strftime('%b %Y')} to {pd.to_datetime(end_date).strftime('%b %Y')}"
                ax.set_title(title_text, fontsize=14, pad=20)
                
                plt.rcParams.update({'font.size': 10})
                plt.tight_layout()
                
                # Display plot in Streamlit
                st.pyplot(fig)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{primary_ethnicity} Statistics:**")
                    st.write(f"- Highest rate: {primary_stats['rate_per_100k'].max():.1f} per 100k")
                    st.write(f"- Lowest rate: {primary_stats['rate_per_100k'].min():.1f} per 100k")
                    st.write(f"- Average rate: {primary_stats['rate_per_100k'].mean():.1f} per 100k")
                
                if comparator_ethnicity != 'None':
                    with col2:
                        comp_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, comparator_ethnicity
                        )
                        st.write(f"**{comparator_ethnicity} Statistics:**")
                        st.write(f"- Highest rate: {comp_stats['rate_per_100k'].max():.1f} per 100k")
                        st.write(f"- Lowest rate: {comp_stats['rate_per_100k'].min():.1f} per 100k")
                        st.write(f"- Average rate: {comp_stats['rate_per_100k'].mean():.1f} per 100k")
                
            except Exception as e:
                st.error(f"Error creating plot: {e}")
                st.error("Please check that your data contains the required columns and date ranges.")

if __name__ == "__main__":
    main()