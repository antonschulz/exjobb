import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sktime.transformations.panel.rocket import Rocket


def dataset_to_numpy(dataset, max_length=1000):
    """
    Converts a PyTorch dataset (where __getitem__ returns (x, y) and x is a tensor of shape (seq_len, 2))
    into NumPy arrays with shape (n_samples, 2, max_length), where sequences longer than max_length are truncated
    and those shorter are padded with zeros.
    """
    X_list = []
    y_list = []
    
    # Collect all sequences and convert each to shape (2, seq_len) via transposition.
    for i in range(len(dataset)):
        x, y = dataset[i]
        # x: (seq_len, 2) --> transpose to (2, seq_len)
        x_np = x.numpy().transpose(1, 0)
        X_list.append(x_np)
        y_list.append(y.item())
    
    X_fixed = []
    for x_np in X_list:
        channels, L = x_np.shape
        if L > max_length:
            # Truncate to first max_length timesteps.
            x_fixed = x_np[:, :max_length]
        elif L < max_length:
            # Pad along the time axis (axis=1)
            pad_width = ((0, 0), (0, max_length - L))
            x_fixed = np.pad(x_np, pad_width, mode='constant', constant_values=0)
        else:
            x_fixed = x_np
        X_fixed.append(x_fixed)
    
    return np.array(X_fixed), np.array(y_list)

class ROCKET_wrapper():
    def __init__(self, num_kernels: int=10000, padding: int=800, logger=None, device=None):
        self.model = Rocket(num_kernels=num_kernels, random_state=42)
        self.clf = None
        self.fitted = False
        self.padding=padding
        self.logger=None
        self.training_history = []
        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')

    def fit(self, train_dataset, val_dataset=None):
        """
        Fit the Rocket transformer and the downstream classifier.

        Args:
            train_dataset: tuple (X_train, y_train), where
                X_train.shape == (n_train, n_channels, series_length)
                y_train.shape == (n_train,)
            val_dataset: optional tuple (X_val, y_val)
        Returns:
            self
        """
        X_train, y_train = dataset_to_numpy(train_dataset, self.padding)
        if val_dataset is not None:
            X_val, y_val = dataset_to_numpy(val_dataset, self.padding)

        # 1) Fit the ROCKET transformer on train
                # after computing your class frequencies
        counts = np.bincount(y_train, minlength=4)
        inv_freq = 1.0 / counts
        sample_weights = inv_freq[y_train]  # shape (n_train,)

        # then
        self.model.fit(X_train)

        X_train_t = self.model.transform(X_train)
        self.clf.fit(X_train_t, y_train)

        # 4) Compute training accuracy
        y_pred_train = self.clf.predict(X_train_t)
        train_acc = accuracy_score(y_train, y_pred_train)

        # Prepare a dict to hold metrics
        run_metrics = {"train_accuracy": train_acc}

        if val_dataset is not None:
            X_val_t     = self.model.transform(X_val)
            y_pred_val  = self.clf.predict(X_val_t)

            # basic accuracy
            val_acc     = accuracy_score(y_val, y_pred_val)
            run_metrics["val_accuracy"] = val_acc

            # macro-F1
            val_macro_f1 = f1_score(y_val, y_pred_val, average='macro')
            run_metrics["val_macro_f1"] = val_macro_f1

            # balanced accuracy
            val_bal_acc  = balanced_accuracy_score(y_val, y_pred_val)
            run_metrics["val_balanced_accuracy"] = val_bal_acc


        # 6) Log or store the metrics
        self.training_history.append(run_metrics)

        print(
        "ROCKET fit complete — "
        "Train Acc: {:.2f}%{}"
        .format(
            train_acc*100,
            (
                f", Val Acc: {val_acc*100:.2f}%"
                f", Val Macro-F1: {val_macro_f1:.3f}"
                f", Val Bal Acc: {val_bal_acc:.3f}"
            ) if val_dataset is not None else ""
            )
        )

        return self
    
    def predict(self, dataset):
        """
        Predict class labels for every sample in dataset.

        Args:
            dataset: either
                - an array X of shape (n_samples, n_channels, series_length), or
                - a tuple (X, y) where only X is used.
        Returns:
            np.ndarray of shape (n_samples,) with predicted class indices.
        """
        X_test, y_test = dataset_to_numpy(dataset, self.padding)

        # Transform & predict
        X_t = self.model.transform(X_test)
        preds = self.clf.predict(X_t)
        return preds

