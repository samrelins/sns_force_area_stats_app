#!/usr/bin/env python3
"""
Generate master dataframes for the Streamlit stop and search analysis app.
This script processes the raw data and saves the master dataframes as CSV files
in a 'data' subdirectory relative to this script.
"""

import os
import pandas as pd

def generate_master_dataframes():
    """Generate and save master dataframes for the Streamlit app"""
    
    print("Starting data generation process...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create data directory for output files
    output_dir = os.path.join(script_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load datasets from your original data directory
    data_dir = os.path.expanduser('~/Documents/VPFRC/stop_and_search/data/')
    print(f"Loading data from: {data_dir}")
    
    # Load stop and search data
    print("Loading stop and search data...")
    sns_path = os.path.join(data_dir, "cleaned_data/sns_cleaned.csv")
    sns_df = pd.read_csv(sns_path)
    sns_df['year_month'] = pd.to_datetime(sns_df['year_month'], format='%Y-%m')
    print(f"Loaded {len(sns_df)} stop and search records")
    
    # Load census data
    print("Loading census data...")
    census_path = os.path.join(data_dir, "cleaned_data/census_21_msoa_cleaned.csv")
    census_df = pd.read_csv(census_path)
    print(f"Loaded {len(census_df)} census records")
    
    # Load geography join tables
    print("Loading geography mapping tables...")
    lad_join_path = os.path.join(data_dir, "cleaned_data/msoa_to_ward_to_lad.csv")
    lad_join_df = pd.read_csv(lad_join_path)
    
    pfa_join_path = os.path.join(data_dir, "cleaned_data/lad_to_pfa_23.csv")
    pfa_join_df = pd.read_csv(pfa_join_path)
    
    # Drop City of London as specified
    pfa_join_df = pfa_join_df[pfa_join_df.police_force_name != "London, City of"]
    print(f"Geography tables loaded, {len(pfa_join_df)} police force areas")
    
    # Create the geography mapping from MSOA to Police Force Area
    print("Creating geography mappings...")
    msoa_to_la = lad_join_df[['msoa_code', 'local_auth_code', 'local_auth_name']].drop_duplicates()
    la_to_pfa = pfa_join_df[['local_auth_code', 'police_force_code', 'police_force_name']].drop_duplicates()
    msoa_to_pfa = pd.merge(msoa_to_la, la_to_pfa, on='local_auth_code', how='left')
    
    # Create master SNS dataframe with police force area information
    print("Processing stop and search data...")
    
    # Filter out unknown MSOAs first
    sns_clean = sns_df[sns_df['msoa_code'] != 'Unknown'].copy()
    print(f"Filtered to {len(sns_clean)} records with known MSOA codes")
    
    # Join SNS data with geography mapping
    master_sns_df = pd.merge(sns_clean, msoa_to_pfa, on='msoa_code', how='left')
    
    # Remove rows where we couldn't map to a police force area
    before_filter = len(master_sns_df)
    master_sns_df = master_sns_df.dropna(subset=['police_force_name'])
    print(f"Mapped {len(master_sns_df)} of {before_filter} records to police force areas")
    
    # Aggregate SNS data to police force area level
    master_sns_df = master_sns_df.groupby([
        'police_force_code', 'police_force_name', 'year_month', 
        'ethnicity_simple', 'ethnicity_full'
    ])['sns_count'].sum().reset_index()
    
    print(f"Aggregated to {len(master_sns_df)} stop and search records by police force area")
    
    # Create master census dataframe with police force area information
    print("Processing census data...")
    master_census_df = pd.merge(census_df, msoa_to_pfa, on='msoa_code', how='left')
    
    # Remove rows where we couldn't map to a police force area
    before_filter = len(master_census_df)
    master_census_df = master_census_df.dropna(subset=['police_force_name'])
    print(f"Mapped {len(master_census_df)} of {before_filter} census records to police force areas")
    
    # Aggregate census data to police force area level
    master_census_df = master_census_df.groupby([
        'police_force_code', 'police_force_name', 
        'ethnicity_simple', 'ethnicity_full'
    ])['census_pop'].sum().reset_index()
    
    print(f"Aggregated to {len(master_census_df)} census records by police force area")
    
    # Save the master dataframes
    print("Saving master dataframes...")
    
    sns_output_path = os.path.join(output_dir, 'master_sns_df.csv')
    master_sns_df.to_csv(sns_output_path, index=False)
    print(f"Saved master SNS dataframe to: {sns_output_path}")
    
    census_output_path = os.path.join(output_dir, 'master_census_df.csv')
    master_census_df.to_csv(census_output_path, index=False)
    print(f"Saved master census dataframe to: {census_output_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    
    print(f"Stop and Search Data:")
    print(f"  - Total records: {len(master_sns_df):,}")
    print(f"  - Date range: {master_sns_df['year_month'].min()} to {master_sns_df['year_month'].max()}")
    print(f"  - Police force areas: {master_sns_df['police_force_name'].nunique()}")
    print(f"  - Ethnicities (simple): {master_sns_df['ethnicity_simple'].nunique()}")
    print(f"  - Ethnicities (detailed): {master_sns_df['ethnicity_full'].nunique()}")
    
    print(f"\nCensus Data:")
    print(f"  - Total records: {len(master_census_df):,}")
    print(f"  - Police force areas: {master_census_df['police_force_name'].nunique()}")
    print(f"  - Ethnicities (simple): {master_census_df['ethnicity_simple'].nunique()}")
    print(f"  - Ethnicities (detailed): {master_census_df['ethnicity_full'].nunique()}")
    print(f"  - Total population: {master_census_df['census_pop'].sum():,}")
    
    print(f"\nFiles saved to: {output_dir}")
    print("Data generation complete!")
    
    return master_sns_df, master_census_df

if __name__ == "__main__":
    try:
        generate_master_dataframes()
    except Exception as e:
        print(f"Error during data generation: {e}")
        print("Please check that your data files exist and are accessible.")
        raise