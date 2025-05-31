# experiments/run_experiment.py
import argparse
import numpy as np
import sys
sys.path.append('../')
from utils.data_loading import load_dataset  # Function to load train/val/test sets
from utils.hyperparameter_tuning import tune_hyperparameters, train_model  # Function to perform hyperparameter tuning
from utils.evaluation import evaluate_model
#from utils.evaluation import evaluate_model  # Evaluation function for your models
from utils.logger import ExperimentLogger
from config import GLOBAL_CONFIG
import torch

# Import model modules
from models import lstm_model, tcn_model, rocket_model, cnn_model

def main(args):
    # Load dataset (should return train, validation, and test sets)
    train_data, val_data, test_data, full_training_data = load_dataset(args.data_dir, diff=True, scaler=args.scaler, augment=args.augment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select the model based on the command-line argument
    if args.model == 'lstm':
        ModelClass = lstm_model.LSTM_model
    elif args.model == 'rocket':
        ModelClass = rocket_model.Rocket_model
    elif args.model == 'cnn':
        ModelClass = cnn_model.CNN_model
    else:
        raise ValueError("Unsupported model type")
    
    logger = ExperimentLogger(run_id=args.run_id)
    logger.set_run_args(args)
    logger.set_global_config(GLOBAL_CONFIG)
    # Hyperparameter tuning (using train and validation sets)
    model_parameter_space_name = f"{args.model}-default" if args.default_model else args.model

    best_config = tune_hyperparameters(ModelClass, train_data, val_data, args.tuning_iterations, model_parameter_space_name, logger)
    print("Best hyperparameters found:", best_config)
    
    import numpy as np

    if args.evaluate:
        test_metrics_list = []
        for run in range(args.num_runs):
            # Initialize, train, and evaluate
            model = ModelClass(**best_config, device=device)
            train_model(model, full_training_data, None)
            logger.log_full_training_history(model.training_history)

            run_metrics = evaluate_model(model, test_data)
            test_metrics_list.append(run_metrics)
            logger.log_test_result(run_metrics)
            print(f"Run {run+1}/{args.num_runs}: Test Metrics: {run_metrics}")

        # Aggregate
        metric_keys = test_metrics_list[0].keys()
        aggregated_metrics = {}
        aggregated_std     = {}

        for key in metric_keys:
            if key == "confusion_matrix":
                # sum confusion‐matrix counts across runs
                first_cm = test_metrics_list[0][key]
                agg_cm = {
                    true_label: {pred_label: 0
                                for pred_label in first_cm[true_label]}
                    for true_label in first_cm
                }
                for run_metrics in test_metrics_list:
                    cm = run_metrics["confusion_matrix"]
                    for t_lbl, row in cm.items():
                        for p_lbl, cnt in row.items():
                            agg_cm[t_lbl][p_lbl] += cnt
                aggregated_metrics[key] = agg_cm
                aggregated_std[key]     = None
            elif key == "classification_report":
                # test_metrics_list[0][key] is a dict whose values are either dicts or a float for 'accuracy'
                avg_report = {}
                for lbl, entry in test_metrics_list[0][key].items():
                    if isinstance(entry, dict):
                        # average each sub‐metric
                        avg_report[lbl] = {}
                        for m_name, _ in entry.items():
                            vals = [run[key][lbl][m_name] for run in test_metrics_list]
                            avg_report[lbl][m_name] = float(np.mean(vals))
                    else:
                        # scalar accuracy
                        vals = [run[key][lbl] for run in test_metrics_list]
                        avg_report[lbl] = float(np.mean(vals))
                aggregated_metrics[key] = avg_report
                aggregated_std[key]     = None
            else:
                # scalar metric → mean ± std
                vals = [run[key] for run in test_metrics_list]
                aggregated_metrics[key] = float(np.mean(vals))
                aggregated_std[key]     = float(np.std(vals))

        # Log & print
        logger.log_final_test_results(aggregated_metrics, aggregated_std)

        print(f"Final Test Performance over {args.num_runs} runs:")
        for key in metric_keys:
            if key in ("confusion_matrix","classification_report"):
                print(f"{key}:")
                print(aggregated_metrics[key])
            else:
                mean = aggregated_metrics[key]
                std  = aggregated_std[key]
                print(f"{key}:  Mean = {mean:.4f}, Std = {std:.4f}")
    else:
        print("Evaluation on the test set is disabled. Only hyperparameter tuning was performed.")

    logger.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run time series model experiments")
    parser.add_argument('--model', type=str, default='lstm', help='Model type (lstm, tcn, rf)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--tuning_iterations', type=int, default=50, help='Number of hyperparameter tuning iterations')
    parser.add_argument('--default_model', action=argparse.BooleanOptionalAction, help="If set, use only default hyperparameters for model")
    parser.add_argument('--num_runs', type=int, default=30, help='Number of final evaluation runs on test set')
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, help="If set, perform evaluation on the test set")
    parser.add_argument('--run_id', type=str, required=False, default=None, help='run_id for logger')
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, help="If set, use data augmentation in pytorch dataloaders")
    parser.add_argument('--scaler', type=str, required=False, default=None, help='scaler string when diff is True. See util.py')
    """
        scalers = {
        "standard": StandardScaler(),
        "minmax":    MinMaxScaler(),
        "robust":    RobustScaler(),
        "maxabs":    MaxAbsScaler(),
        "quantile":  QuantileTransformer(output_distribution="uniform"),
        "power":     PowerTransformer(method="yeo-johnson"),
    }
    """

    parser.set_defaults(evaluate=False, default_model=False, augment=True)
    
    
    args = parser.parse_args()
    main(args)
