# config.py

import random

# ----- Model-specific hyperparameter sampling functions -----

def sample_lstm_config():
    """
    Sample a random hyperparameter configuration for an LSTM model.
    Returns a dictionary with keys corresponding to hyperparameter names.
    """
    config = {
        "input_size": 2,
        "num_classes": 4,
        "hidden_size": random.choice([64, 32, 16]),
        "num_layers": random.choice([2,4,8]),
        "fc_units": random.choice([[32], [32, 16]]),
        "bi_lstm": False,
        "dropout_lstm": random.choice([0.0, 0.1]),
        "dropout_fc": random.choice([0.0,0.1, 0.2]),
        "num_epochs": random.choice([50, 100]),
        "learning_rate": random.choice([1e-3]),
        "weight_decay": random.choice([0.0, 1e-5]),
    }
    return config

def sample_rf_config():
    """
    Sample a random hyperparameter configuration for a Random Forest model.
    Returns a dictionary with keys corresponding to hyperparameter names.
    """
    config = {
        "n_estimators": random.choice([50, 100, 200]),
        "max_depth": random.choice([None, 10, 20, 30]),
        "min_samples_split": random.choice([2, 5, 10])
    }
    return config

def sample_tcn_config():
    """
    Sample a random hyperparameter configuration for an LSTM model.
    Returns a dictionary with keys corresponding to hyperparameter names.
    """
    config = {
        "input_channels": 2,
        "num_classes": 4,
        "num_epochs": random.choice([50, 100]),
        "learning_rate": random.choice([0.001, 0.005, 5e-4]),
        "num_levels": random.choice([3, 6, 9, 12, 16]),
        "kernel_size": random.choice([1,3,5,10]),
        "dropout": random.choice([0, 0.2]),
        "num_filters": random.choice([8,16,32]),
        "weight_decay": random.choice([0,1e-5]),
    }
    return config

def sample_rocket_config():
    config = {
        "num_kernels": 10000, 
        "padding": 800,
    }
    return config


hyperparameter_spaces = {
    "lstm": sample_lstm_config,
    "tcn": sample_tcn_config,
    "rf": sample_rf_config,
    "rocket": sample_rocket_config,
}

# ----- Global configuration settings -----

GLOBAL_CONFIG = {
    "tuning_iterations": 50,    # Total iterations for hyperparameter tuning
    "num_runs": 30,             # Number of final evaluation runs
    "batch_size": 16,           # Batch size for deep learning experiments
    "data_dir": "../datasets/it-trajs",      # Default path to the dataset directory
    "random_seed": 42           # Global random seed for reproducibility
}
