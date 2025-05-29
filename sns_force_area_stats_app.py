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

# Function to calculate rates by police force area with exclusion capability
def calculate_pfa_ethnicity_rates(sns_df, census_df, start_date=None, end_date=None, ethnicity_selection="All", exclude_ethnicity=None):
    """
    Calculate stop and search rates per 100,000 people by police force area and ethnicity
    Can exclude another ethnicity category to create meaningful non-overlapping comparisons
    """
    
    # Filter by date range
    filtered_sns = sns_df.copy()
    if start_date is not None:
        filtered_sns = filtered_sns[filtered_sns['year_month'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        filtered_sns = filtered_sns[filtered_sns['year_month'] <= pd.to_datetime(end_date)]
    
    # Filter by ethnicity
    sns_filtered = filter_data_by_ethnicity(filtered_sns, ethnicity_selection)
    census_filtered = filter_data_by_ethnicity(census_df, ethnicity_selection)
    
    # Apply exclusion if specified (to avoid overlapping categories)
    if exclude_ethnicity and exclude_ethnicity != "All":
        if exclude_ethnicity in ["Asian", "Black", "Mixed", "White", "Other"]:
            # Exclude from simple ethnicity
            sns_filtered = sns_filtered[sns_filtered['ethnicity_simple'] != exclude_ethnicity]
            census_filtered = census_filtered[census_filtered['ethnicity_simple'] != exclude_ethnicity]
        else:
            # Exclude from detailed ethnicity
            sns_filtered = sns_filtered[sns_filtered['ethnicity_full'] != exclude_ethnicity]
            census_filtered = census_filtered[census_filtered['ethnicity_full'] != exclude_ethnicity]
    
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

# Function to determine overlap relationship between ethnicities
def get_overlap_relationship(ethnicity1, ethnicity2, census_df):
    """Determine if ethnicities overlap and which should be excluded from which"""
    if ethnicity1 == ethnicity2:
        return "identical"
    
    # Handle "All" category overlaps
    if ethnicity1 == "All" and ethnicity2 != "All":
        return "exclude_from_all"  # exclude ethnicity2 from ethnicity1
    elif ethnicity2 == "All" and ethnicity1 != "All":
        return "exclude_from_all"  # exclude ethnicity1 from ethnicity2
    
    simple_ethnicities = ["Asian", "Black", "Mixed", "White", "Other"]
    
    # Get mapping of simple to detailed ethnicities
    simple_detailed_map = census_df.groupby('ethnicity_simple')['ethnicity_full'].unique()
    
    # Check if one is simple and other is its detailed subset
    if ethnicity1 in simple_ethnicities and ethnicity2 in simple_detailed_map.get(ethnicity1, []):
        return "exclude_detailed_from_simple"  # exclude ethnicity2 from ethnicity1
    elif ethnicity2 in simple_ethnicities and ethnicity1 in simple_detailed_map.get(ethnicity2, []):
        return "exclude_detailed_from_simple"  # exclude ethnicity1 from ethnicity2
    
    return "no_overlap"

# Function to add labels to bars
def add_bar_labels(ax, bars, values):
    """Add value labels on top of bars"""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label_y = height + max(abs(height) * 0.02, 0.5) if height >= 0 else height - max(abs(height) * 0.02, 0.5)
        ax.text(bar.get_x() + bar.get_width()/2., label_y, f'{value:.0f}', 
               ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

# Function to create comparison plot (two ethnicities)
def create_comparison_plot(ax, primary_stats, comp_stats, primary_ethnicity, comparator_ethnicity, census_df):
    """Create plot comparing rates between two ethnicities"""
    
    # Check relationship between categories
    relationship = get_overlap_relationship(primary_ethnicity, comparator_ethnicity, census_df)
    
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
    
    # Add warning message for identical categories
    if relationship == "identical":
        ax.text(0.5, 0.95, f'ℹ️ Note: Comparing identical categories\nShowing raw rates for {primary_ethnicity}', 
               transform=ax.transAxes, ha='center', va='top', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
               fontsize=10, fontweight='bold')
    
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
    
    # Show overlap relationship info
    if comparator_ethnicity != 'None':
        relationship = get_overlap_relationship(primary_ethnicity, comparator_ethnicity, census_df)
        if relationship == "identical":
            st.sidebar.warning("⚠️ You've selected identical categories for comparison")
        elif relationship == "exclude_from_all":
            st.sidebar.info("ℹ️ Overlapping categories detected - exclusions will be applied automatically")
        elif relationship == "exclude_detailed_from_simple":
            st.sidebar.info("ℹ️ Overlapping categories detected - exclusions will be applied automatically")
    
    # Generate plot
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
                # Determine overlap relationship and apply appropriate exclusions
                relationship = get_overlap_relationship(primary_ethnicity, comparator_ethnicity, census_df)
                
                if relationship == "exclude_detailed_from_simple":
                    # Determine which is simple and which is detailed, then exclude detailed from simple
                    simple_ethnicities = ["Asian", "Black", "Mixed", "White", "Other"]
                    
                    if primary_ethnicity in simple_ethnicities:
                        # Primary is simple, exclude comparator from primary
                        primary_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, primary_ethnicity, exclude_ethnicity=comparator_ethnicity
                        )
                        comp_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, comparator_ethnicity
                        )
                    else:
                        # Comparator is simple, exclude primary from comparator
                        comp_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, comparator_ethnicity, exclude_ethnicity=primary_ethnicity
                        )
                        
                elif relationship == "exclude_from_all":
                    # One category is "All", exclude the other from it
                    if primary_ethnicity == "All":
                        # Primary is "All", exclude comparator from primary
                        primary_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, primary_ethnicity, exclude_ethnicity=comparator_ethnicity
                        )
                        comp_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, comparator_ethnicity
                        )
                    else:
                        # Comparator is "All", exclude primary from comparator
                        comp_stats = calculate_pfa_ethnicity_rates(
                            sns_df, census_df, start_date, end_date, comparator_ethnicity, exclude_ethnicity=primary_ethnicity
                        )
                else:
                    # No overlap or identical - no exclusions needed
                    comp_stats = calculate_pfa_ethnicity_rates(
                        sns_df, census_df, start_date, end_date, comparator_ethnicity
                    )
                
                title_text = create_comparison_plot(ax, primary_stats, comp_stats, primary_ethnicity, comparator_ethnicity, census_df)
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
                st.write(f"- Total searches: {primary_stats['sns_count'].sum():,}")
                st.write(f"- Total population: {primary_stats['census_pop'].sum():,}")
            
            if comparator_ethnicity != 'None':
                with col2:
                    st.write(f"**{comparator_ethnicity} Statistics:**")
                    st.write(f"- Highest rate: {comp_stats['rate_per_100k'].max():.1f} per 100k")
                    st.write(f"- Lowest rate: {comp_stats['rate_per_100k'].min():.1f} per 100k")
                    st.write(f"- Average rate: {comp_stats['rate_per_100k'].mean():.1f} per 100k")
                    st.write(f"- Total searches: {comp_stats['sns_count'].sum():,}")
                    st.write(f"- Total population: {comp_stats['census_pop'].sum():,}")
            
            # Show relationship info
            if comparator_ethnicity != 'None':
                st.subheader("Category Relationship")
                relationship = get_overlap_relationship(primary_ethnicity, comparator_ethnicity, census_df)
                
                if relationship == "identical":
                    st.info("ℹ️ **Identical categories:** You're comparing the same ethnicity group")
                elif relationship == "exclude_from_all":
                    st.info("ℹ️ **Overlapping categories:** One category contains the other - exclusions applied automatically")
                elif relationship == "exclude_detailed_from_simple":
                    st.info("ℹ️ **Overlapping categories:** Detailed ethnicity is subset of simple ethnicity - exclusions applied automatically")
                else:
                    st.success("✅ **Non-overlapping categories:** Clean comparison between distinct groups")
            
        except Exception as e:
            st.error(f"Error creating plot: {e}")
            st.error("Please check that your data contains the required columns and date ranges.")

if __name__ == "__main__":
    main()