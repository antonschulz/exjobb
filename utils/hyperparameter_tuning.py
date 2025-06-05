# hyperparameter_tuning.py

import random
import numpy as np
from config import hyperparameter_spaces  # Import the mapping from model type to sampling functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from .evaluation import evaluate_model
import torch

def tune_hyperparameters(ModelClass, train_data, val_data, tuning_iterations, model_type, logger=None):
    """
    Performs random search hyperparameter tuning.

    Args:
        ModelClass: The class or factory function to create the model.
        train_data: Training data (e.g., DataLoader, dataset, or NumPy arrays).
        val_data: Validation data for evaluating performance.
        tuning_iterations (int): Number of random configurations to try.
        model_type (str): Model type key (e.g., 'lstm', 'cnn', 'rf') to select the corresponding hyperparameter space.

    Returns:
        best_config (dict): The hyperparameter configuration that achieved the highest validation metric.
    """
    best_config = None
    best_metrics = {'balanced_accuracy': -float('inf')}  # Assuming higher is better (e.g., accuracy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    for i in range(tuning_iterations):
        # Sample a configuration for the given model type using the function defined in your config module.
        config = hyperparameter_spaces[model_type]()  # e.g., sample_lstm_config() if model_type == 'lstm'
        print(f"Iteration {i+1}/{tuning_iterations}: Testing config: {config}")
        hp_config = config

        # Instantiate the model using the sampled configuration.
        model = ModelClass(**config, logger=logger, device=device)

        # Train the model on the training data.
        model = train_model(model, train_data, val_data)  # You need to implement or import this function.

        epoch_hist = model.training_history
        # Evaluate the model on the validation set to get a performance metric.
        metrics = evaluate_model(model, val_data)  # You need to implement or import this function.
        final_val = metrics
        logger.log_hp_tuning(hp_config, epoch_hist, final_val)
        print(f"Iteration {i+1}: Config: {config}, Validation Metric: {metrics}")

        # Keep track of the best configuration based on the performance metric.
        if metrics['balanced_accuracy'] > best_metrics['balanced_accuracy']:
            best_metrics = metrics
            best_config = config
            if hasattr(model, "early_stop_epochs") and model.early_stop_epochs:
                best_config['early_stop_epochs'] = model.early_stop_epochs

    print("Best hyperparameter configuration found:", best_config)
    print("Best validation metric:", best_metrics)
    logger.set_best_hyperparameters(best_config)
    logger.set_best_validation_metric(best_metrics)
    return best_config


# --- Example implementations for train_model and evaluate_model ---
# Note: In your actual project, these functions should include your full training loop
# and evaluation procedure, respectively.

def train_model(model, train_data, val_data):
    """
    A placeholder training function.
    Replace this with your actual training routine that trains 'model' on 'train_data'.
    """
    # For example, if using PyTorch, iterate over the train_data DataLoader,
    # perform forward and backward passes, update weights, etc.
    # Here, we'll assume the model has a method 'fit' that handles training.
    model.fit(train_data, val_data)
    return model


