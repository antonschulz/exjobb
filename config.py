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
        "early_stopping": True,
        "batch_size": random.choice([16, 32]),
        "num_epochs": random.choice([200]),
        "learning_rate": random.choice([1e-3, 5e-3, 5e-4]),
        "hidden_size": random.choice([64, 32, 16]),
        "num_layers": random.choice([1,2,4]),
        "fc_units": random.choice([[32], [32, 16]]),
        "bi_lstm": False,
        "dropout_lstm": random.choice([0.0, 0.1]),
        "dropout_fc": random.choice([0.0,0.1]),
        "weight_decay": random.choice([0.0, 1e-5]),
    }
    return config

def sample_rocket_config():
    config = {
        "num_kernels": 10000, 
        "padding": 3000,
    }
    return config

def default_lstm_config():
    """
    Sample a random hyperparameter configuration for an LSTM model.
    Returns a dictionary with keys corresponding to hyperparameter names.
    """
    config = {
        "input_size": 2,
        "num_classes": 4,
        "hidden_size": 64,
        "num_layers": 3,
        "fc_units": [32],
        "bi_lstm": False,
        "dropout_lstm": 0,
        "dropout_fc": 0,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 0,
    }
    return config



def default_cnn_config():
    config = {
        "input_channels": 2,
        "num_classes": 4,
        "batch_size": 16,
        "early_stopping": True,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "num_levels": 6,
        "kernel_size": 3,
        "dropout": 0,
        "num_filters": 16,
        "weight_decay": 0,
    }

    return config

def sample_cnn_config():
    """
    Sample a random hyperparameter configuration for an LSTM model.
    Returns a dictionary with keys corresponding to hyperparameter names.
    """
    config = {
        "input_channels": 2,
        "num_classes": 4,
        "early_stopping": True,
        "batch_size": random.choice([16, 32]),
        "num_epochs": random.choice([200]),
        "learning_rate": random.choice([1e-3, 5e-3, 5e-4]),
        "num_levels": random.choice([3, 6, 9, 12]),
        "kernel_size": random.choice([1,3,5,9]),
        "dropout": 0,
        "num_filters": random.choice([4, 8,16,32]),
        "weight_decay": random.choice([0,1e-5]),
    }
    return config



hyperparameter_spaces = {
    "lstm": sample_lstm_config,
    "cnn": sample_cnn_config,
    "rocket": sample_rocket_config,
    # defaults
    "lstm-default": default_lstm_config,
    "cnn-default": default_cnn_config
}

# ----- Global configuration settings -----

GLOBAL_CONFIG = {
    "data_dir": "../datasets/it-trajs",      # Default path to the dataset directory
    "random_seed": 42           # Global random seed for reproducibility
}
