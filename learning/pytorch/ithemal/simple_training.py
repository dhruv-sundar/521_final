import sys
import os
import torch
import argparse
import time
from tqdm import tqdm
import random

# Add Ithemal paths
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

# Import Ithemal modules
import models.graph_models as md
import models.losses as ls
import models.train as tr
import data.data_cost as dt
from ithemal_utils import BaseParameters, load_data, load_model

def train_model(data_file, model_file=None, epochs=3, batch_size=32, 
                learning_rate=0.001, hidden_size=256, embed_size=256):
    """Train the RNN model directly without the distributed setup"""
    
    # Create base parameters
    base_params = BaseParameters(
        data=data_file,
        embed_mode='none',
        embed_file=os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch', 
                               'inputs', 'embeddings', 'code_delim.emb'),
        random_edge_freq=0.0,
        predict_log=False,
        no_residual=True,  # Use only RNN, not the residual model
        no_dag_rnn=True,   # Don't use the DAG-RNN model
        dag_reduction=md.ReductionType.MAX,
        edge_ablation_types=[],
        embed_size=embed_size,
        hidden_size=hidden_size,
        linear_embeddings=False,
        use_rnn=True,      # Use the RNN model
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
    
    # Load data and model
    print("Loading data...")
    data = load_data(base_params, direct=True)
    print(f"Data loaded: {len(data.train)} training samples, {len(data.test)} test samples")
    
    print("Creating model...")
    model = load_model(base_params, data)
    
    # Load pre-trained model if specified
    if model_file:
        print(f"Loading pre-trained model from {model_file}")
        model.load_state_dict(torch.load(model_file))
    
    # Create trainer
    trainer = tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=batch_size, clip=None,
        lr=learning_rate, predict_log=base_params.predict_log
    )
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle training data
        random.shuffle(data.train)
        
        # Train for one epoch
        epoch_loss = 0
        total_items = 0
        
        for i in tqdm(range(0, len(data.train), batch_size)):
            batch = data.train[i:i+batch_size]
            loss, n_items = trainer.train_batch(batch)
            epoch_loss += loss * n_items
            total_items += n_items
        
        avg_epoch_loss = epoch_loss / total_items
        print(f"Training loss: {avg_epoch_loss:.6f}")
        
        # Validate
        val_loss = 0
        val_items = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(data.test), batch_size)):
                batch = data.test[i:i+batch_size]
                loss, n_items = trainer.validate_batch(batch)
                val_loss += loss * n_items
                val_items += n_items
        
        avg_val_loss = val_loss / val_items
        print(f"Validation loss: {avg_val_loss:.6f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pt")
            print(f"Saved new best model with validation loss: {best_loss:.6f}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ithemal RNN model with simplified setup")
    parser.add_argument("--data", required=True, help="Path to the data file")
    parser.add_argument("--model", help="Path to pre-trained model to continue training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--embed-size", type=int, default=256, help="Embedding size")
    
    args = parser.parse_args()
    
    # Make sure ITHEMAL_HOME is set
    if 'ITHEMAL_HOME' not in os.environ:
        print("Error: ITHEMAL_HOME environment variable not set")
        sys.exit(1)
    
    # Train the model
    train_model(
        data_file=args.data,
        model_file=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        embed_size=args.embed_size
    )