import sys
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

# Import Ithemal modules
import models.graph_models as md
import data.data_cost as dt
from ithemal_utils import BaseParameters, load_data, load_model

def evaluate_model(model_file, data_file, output_dir="./results", direct=True):
    """
    Evaluate a trained Ithemal model on a test dataset
    
    Args:
        model_file: Path to the trained model (.pt file)
        data_file: Path to the test data
        output_dir: Directory to save results
        direct: Whether to load data directly from CSV
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base parameters (same as used for training)
    base_params = BaseParameters(
        data=data_file,
        embed_mode='none',
        embed_file=os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch', 
                               'inputs', 'embeddings', 'code_delim.emb'),
        random_edge_freq=0.0,
        predict_log=False,
        no_residual=True,
        no_dag_rnn=True,
        dag_reduction=md.ReductionType.MAX,
        edge_ablation_types=[],
        embed_size=256,
        hidden_size=256,
        linear_embeddings=False,
        use_rnn=True,
        rnn_type=md.RnnType.LSTM,
        rnn_hierarchy_type=md.RnnHierarchyType.MULTISCALE,
        rnn_connect_tokens=False,
        rnn_skip_connections=False,
        rnn_learn_init=False,
        no_mem=False,
        linear_dependencies=False,
        flat_dependencies=False,
        dag_nonlinearity=None,
        dag_nonlinearity_width=128,
        dag_nonlinear_before_max=False,
    )
    
    # Load data
    print("Loading data...")
    data = load_data(base_params, direct=direct)
    print(f"Test set size: {len(data.test)} samples")
    
    # Load model
    print(f"Loading model from {model_file}...")
    model = load_model(base_params, data)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for item in tqdm(data.test):
            # Get prediction
            prediction = model(item).item()
            
            # Get actual throughput
            actual = item.y
            
            predictions.append(prediction)
            actuals.append(actual)
            
            # Free memory
            model.remove_refs(item)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Calculate distribution of errors
    errors = np.abs(actuals - predictions)
    error_percentiles = np.percentile(errors, [25, 50, 75, 90, 95, 99])
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f} cycles")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} cycles")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print("\nError Distribution:")
    print(f"25th percentile: {error_percentiles[0]:.4f} cycles")
    print(f"Median error: {error_percentiles[1]:.4f} cycles")
    print(f"75th percentile: {error_percentiles[2]:.4f} cycles")
    print(f"90th percentile: {error_percentiles[3]:.4f} cycles")
    print(f"95th percentile: {error_percentiles[4]:.4f} cycles")
    print(f"99th percentile: {error_percentiles[5]:.4f} cycles")
    
    # Save results to file
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f} cycles\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f} cycles\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
        f.write(f"R² Score: {r2:.4f}\n\n")
        f.write("Error Distribution:\n")
        f.write(f"25th percentile: {error_percentiles[0]:.4f} cycles\n")
        f.write(f"Median error: {error_percentiles[1]:.4f} cycles\n")
        f.write(f"75th percentile: {error_percentiles[2]:.4f} cycles\n")
        f.write(f"90th percentile: {error_percentiles[3]:.4f} cycles\n")
        f.write(f"95th percentile: {error_percentiles[4]:.4f} cycles\n")
        f.write(f"99th percentile: {error_percentiles[5]:.4f} cycles\n")
    
    print(f"Results saved to {results_file}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Throughput (cycles)')
    plt.ylabel('Predicted Throughput (cycles)')
    plt.title('Actual vs. Predicted Throughput')
    plt.grid(True)
    
    # Add metrics to plot
    plt.text(0.05, 0.95, f'MAE: {mae:.4f} cycles\nMAPE: {mape:.2f}%\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plot_file = os.path.join(output_dir, "prediction_scatter.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {plot_file}")
    
    # Create error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.xlabel('Absolute Error (cycles)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    
    # Save histogram
    hist_file = os.path.join(output_dir, "error_histogram.png")
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    print(f"Error histogram saved to {hist_file}")
    
    # Save raw predictions for further analysis
    predictions_file = os.path.join(output_dir, "predictions.csv")
    with open(predictions_file, "w") as f:
        f.write("actual,predicted,error,percent_error\n")
        for i in range(len(actuals)):
            error = predictions[i] - actuals[i]
            percent_error = (error / actuals[i]) * 100 if actuals[i] != 0 else float('inf')
            f.write(f"{actuals[i]},{predictions[i]},{error},{percent_error}\n")
    
    print(f"Raw predictions saved to {predictions_file}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'error_percentiles': error_percentiles,
        'predictions': predictions,
        'actuals': actuals
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Ithemal model")
    parser.add_argument("--model", required=True, help="Path to the trained model (.pt file)")
    parser.add_argument("--data", required=True, help="Path to the test data")
    parser.add_argument("--output", default="./results", help="Directory to save results")
    parser.add_argument("--direct", action="store_true", help="Load data directly from CSV")
    
    args = parser.parse_args()
    
    # Make sure ITHEMAL_HOME is set
    if 'ITHEMAL_HOME' not in os.environ:
        print("Error: ITHEMAL_HOME environment variable not set")
        sys.exit(1)
    
    # Evaluate model
    evaluate_model(args.model, args.data, args.output, args.direct)