import sys
import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv

# Add Ithemal paths
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

# Import Ithemal modules
import data.data_cost as dt
from ithemal_utils import BaseParameters, load_data

def analyze_high_error_samples(predictions_file, data_file, output_file, error_threshold=15.0, direct=True):
    """
    Identify samples with high prediction error and output their code_xml
    
    Args:
        predictions_file: Path to the CSV file with predictions (from evaluate_model.py)
        data_file: Path to the original data file
        output_file: Path to save the high error samples with their code_xml
        error_threshold: Percentage error threshold (default: 15.0%)
        direct: Whether to load data directly from CSV
    """
    print(f"Analyzing samples with error > {error_threshold}%...")
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}...")
    predictions_df = pd.read_csv(predictions_file)
    
    # Filter high error samples
    high_error_df = predictions_df[abs(predictions_df['percent_error']) > error_threshold]
    print(f"Found {len(high_error_df)} samples with error > {error_threshold}%")
    
    # Create base parameters for data loading
    base_params = BaseParameters(
        data=data_file,
        embed_mode='none',
        embed_file=None,
    )
    
    # Load data
    print(f"Loading dataset from {data_file}...")
    data = load_data(base_params, direct=direct)
    
    # Create a dictionary to map actual throughput values to data samples
    # This is a heuristic approach since we don't have direct IDs
    print("Building throughput-to-sample mapping...")
    throughput_to_samples = {}
    for i, item in enumerate(tqdm(data.test)):
        if item.y not in throughput_to_samples:
            throughput_to_samples[item.y] = []
        throughput_to_samples[item.y].append(i)
    
    # Open output file
    print(f"Writing high error samples to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual', 'predicted', 'percent_error', 'code_intel', 'code_xml'])
        
        # Process each high error sample
        for _, row in tqdm(high_error_df.iterrows(), total=len(high_error_df)):
            actual = row['actual']
            predicted = row['predicted']
            percent_error = row['percent_error']
            
            # Find matching samples
            if actual in throughput_to_samples:
                # Get all potential matches
                potential_matches = throughput_to_samples[actual]
                
                # For each potential match, write its information
                for idx in potential_matches:
                    item = data.test[idx]
                    
                    # Get code_intel and code_xml
                    code_intel = item.code_intel if hasattr(item, 'code_intel') else "N/A"
                    code_xml = item.code_xml if hasattr(item, 'code_xml') else "N/A"
                    
                    # Write to CSV
                    writer.writerow([actual, predicted, percent_error, code_intel, code_xml])
    
    print(f"Analysis complete. Results saved to {output_file}")

def analyze_high_error_samples_with_ids(predictions_file, data_file, output_file, error_threshold=15.0, direct=True):
    """
    Alternative version that works if your dataset has code_id that can be matched directly
    
    Args:
        predictions_file: Path to the CSV file with predictions (from evaluate_model.py)
        data_file: Path to the original data file
        output_file: Path to save the high error samples with their code_xml
        error_threshold: Percentage error threshold (default: 15.0%)
        direct: Whether to load data directly from CSV
    """
    print(f"Analyzing samples with error > {error_threshold}%...")
    
    # Load predictions with sample indices
    print(f"Loading predictions from {predictions_file}...")
    predictions_df = pd.read_csv(predictions_file)
    
    # Filter high error samples
    high_error_df = predictions_df[abs(predictions_df['percent_error']) > error_threshold]
    print(f"Found {len(high_error_df)} samples with error > {error_threshold}%")
    
    # Create base parameters for data loading
    base_params = BaseParameters(
        data=data_file,
        embed_mode='none',
        embed_file=None,
    )
    
    # Load data
    print(f"Loading dataset from {data_file}...")
    data = load_data(base_params, direct=direct)
    
    # Create a dictionary to map code_ids to data samples
    print("Building code_id-to-sample mapping...")
    id_to_sample = {}
    for i, item in enumerate(tqdm(data.test)):
        if hasattr(item, 'code_id'):
            id_to_sample[item.code_id] = i
    
    # Open output file
    print(f"Writing high error samples to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code_id', 'actual', 'predicted', 'percent_error', 'code_intel', 'code_xml'])
        
        # Process each high error sample
        for i, row in tqdm(high_error_df.iterrows(), total=len(high_error_df)):
            actual = row['actual']
            predicted = row['predicted']
            percent_error = row['percent_error']
            
            # Get the sample index
            sample_idx = i % len(data.test)  # This assumes predictions are in the same order as data.test
            item = data.test[sample_idx]
            
            # Get code_id, code_intel and code_xml
            code_id = item.code_id if hasattr(item, 'code_id') else str(sample_idx)
            code_intel = item.code_intel if hasattr(item, 'code_intel') else "N/A"
            code_xml = item.code_xml if hasattr(item, 'code_xml') else "N/A"
            
            # Write to CSV
            writer.writerow([code_id, actual, predicted, percent_error, code_intel, code_xml])
    
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze samples with high prediction error")
    parser.add_argument("--predictions", required=True, help="Path to the predictions CSV file")
    parser.add_argument("--data", required=True, help="Path to the original data file")
    parser.add_argument("--output", required=True, help="Path to save the high error samples")
    parser.add_argument("--threshold", type=float, default=15.0, help="Error threshold percentage")
    parser.add_argument("--direct", action="store_true", help="Load data directly from CSV")
    parser.add_argument("--use-ids", action="store_true", help="Use code_ids for matching (if available)")
    
    args = parser.parse_args()
    
    # Make sure ITHEMAL_HOME is set
    if 'ITHEMAL_HOME' not in os.environ:
        print("Error: ITHEMAL_HOME environment variable not set")
        sys.exit(1)
    
    # Analyze high error samples
    if args.use_ids:
        analyze_high_error_samples_with_ids(
            args.predictions, args.data, args.output, args.threshold, args.direct
        )
    else:
        analyze_high_error_samples(
            args.predictions, args.data, args.output, args.threshold, args.direct
        )