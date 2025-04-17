# logger.py
import json
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="logs", run_id=None):
        """
        Initializes the logger and sets up the structure for logging experiment data.
        
        Args:
            log_dir (str): Directory where log files will be stored.
            run_id (str, optional): Identifier for this run. If None, a timestamp is used.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.log_file = os.path.join(log_dir, f"run_{self.run_id}.json")
        
        # Structure for storing logs.
        self.log_data = {
            "global_config": {},
            "hp_tuning": [],          # List of dicts: hyperparameter set, per-epoch history and final validation metrics
            "best_hyperparameters": {}, 
            "full_training_history": [],  # Epoch-level metrics when training on full training+validation set with best hyperparams
            "test_results": [],       # List of dicts for each test run results
            "final_avg_res": {},
            "final_std_res": {},
        }

    def set_global_config(self, config):
        """
        Store global configuration settings.
        
        Args:
            config (dict): Global settings (e.g., number of epochs, batch size, etc.).
        """
        self.log_data["global_config"] = config

    def log_hp_tuning(self, hyperparams, epoch_history, final_val_metrics):
        """
        Log the tuning results for one hyperparameter configuration.
        
        Args:
            hyperparams (dict): The hyperparameters tried.
            epoch_history (list[dict]): A list of dictionaries for each epoch with metrics,
                e.g., [{"epoch": 1, "train_loss": 0.5, "train_acc": 80, "val_loss": 0.6, "val_acc": 78}, ...].
            final_val_metrics (dict): Final validation metrics (e.g., F1-score, precision, accuracy, recall).
        """
        self.log_data["hp_tuning"].append({
            "hyperparameters": hyperparams,
            "epoch_history": epoch_history,
            "final_val_metrics": final_val_metrics
        })

    def set_best_hyperparameters(self, hyperparams):
        """
        Store the best hyperparameters found during tuning.
        
        Args:
            hyperparams (dict): The best hyperparameter set.
        """
        self.log_data["best_hyperparameters"] = hyperparams

    def log_full_training_history(self, epoch_history):
        """
        Log the training history (e.g., loss and accuracy per epoch) for training on the full
        training+validation set using the best hyperparameters.
        
        Args:
            epoch_history (list[dict]): A list of epoch metrics for the full training run.
        """
        self.log_data["full_training_history"].append(epoch_history)

    def log_test_result(self, test_result):
        """
        Log the final test result for one run.
        
        Args:
            test_result (dict): A dictionary with test metrics (e.g., {"test_loss": 0.3, "test_accuracy": 85,
                "test_f1": 0.78, "test_precision": 0.8, ...}).
        """
        self.log_data["test_results"].append(test_result)

    def log_final_test_results(self, mean_results, std_results):
        self.log_data["final_avg_res"] = mean_results
        self.log_data["final_std_res"] = std_results

    def save(self):
        """
        Save the logged experiment data to a JSON file.
        """
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=4)

# Example usage in isolation:
if __name__ == "__main__":
    logger = ExperimentLogger(run_id="example_run")
    logger.set_global_config({
        "num_epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    })

    # Simulate logging one hyperparameter set tuning:
    hp_config = {"learning_rate": 0.001, "hidden_size": 64, "num_layers": 2, "dropout": 0.3}
    epoch_hist = [
        {"epoch": 1, "train_loss": 0.5, "train_acc": 80, "val_loss": 0.6, "val_acc": 78},
        {"epoch": 2, "train_loss": 0.4, "train_acc": 82, "val_loss": 0.55, "val_acc": 80}
    ]
    final_val = {"val_accuracy": 80, "val_f1": 0.78, "val_precision": 0.79}
    logger.log_hp_tuning(hp_config, epoch_hist, final_val)
    
    # Set best hyperparameters
    logger.set_best_hyperparameters(hp_config)
    
    # Log full training history for the best hyperparams
    full_train_hist = [
        {"epoch": 1, "train_loss": 0.45, "train_acc": 81},
        {"epoch": 2, "train_loss": 0.35, "train_acc": 83}
    ]
    logger.log_full_training_history(full_train_hist)
    
    # Log multiple test results.
    logger.log_test_result({"test_loss": 0.3, "test_accuracy": 85, "test_f1": 0.80, "test_precision": 0.82})
    logger.log_test_result({"test_loss": 0.28, "test_accuracy": 86, "test_f1": 0.81, "test_precision": 0.83})
    
    logger.save()
