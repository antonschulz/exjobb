import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_loading import collate_fn
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(ResidualBlock, self).__init__()
        padding = "same"#(kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # x shape: (batch, channels, seq_length)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        # Trim padded extra timesteps to ensure causal convolution output matches input length.
        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_levels=3, kernel_size=3, dropout=0.2, num_filters=32):
        """
        Args:
            input_channels (int): Number of features per point (here 2 for x and y).
            num_classes (int): Number of output classes.
            num_levels (int): Number of TemporalBlocks.
            kernel_size (int): Convolution kernel size.
            dropout (float): Dropout rate.
            num_filters (int): Number of filters in conv layers.
        """
        super(CNN, self).__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation
            in_channels = input_channels if i == 0 else num_filters
            layers += [ResidualBlock(in_channels, num_filters, kernel_size, dilation, dropout)]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, num_classes)
        
    def forward(self, x, lengths):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, channels)
            lengths (torch.Tensor): Actual lengths of each sequence in the batch.
        Returns:
            torch.Tensor: Class logits of shape (batch, num_classes)
        """
        # Transpose to (batch, channels, seq_length) for Conv1d layers.
        x = x.transpose(1, 2)
        y = self.tcn(x)  # (batch, num_filters, seq_length)
        # Transpose back to (batch, seq_length, num_filters)
        y = y.transpose(1, 2)
        # Create a mask for valid timesteps
        batch_size, max_seq, num_filters = y.shape
        mask = torch.arange(max_seq, device=y.device).expand(batch_size, max_seq) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).float()  # shape (batch, seq_length, 1)
        # Mask the output and compute the sum per sample, then divide by the sequence lengths.
        y = (y * mask).sum(dim=1) / lengths.unsqueeze(1).float()
        out = self.fc(y)
        return out
    
class CNN_model:
    def __init__(self, input_channels, num_classes, early_stopping=True, num_levels=3, kernel_size=3, dropout=0.3, num_filters=16, num_epochs=10, learning_rate=0.001, device=None, logger=None, weight_decay=None):
        """
        A wrapper that instantiates a TCN model and provides fit and predict methods.
        
        Args:
            input_channels (int): Number of features per time step.
            num_classes (int): Number of output classes.
            num_levels (int): Number of temporal blocks.
            kernel_size (int): Convolution kernel size.
            dropout (float): Dropout rate.
            num_filters (int): Number of filters in the TCN layers.
            num_epochs (int): Default number of epochs for training (used in fit).
            lr (float): Learning rate.
            device (torch.device): Device on which to run the model.
        """
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.early_stopping=early_stopping
        self.device = device if device is not None else torch.device("cpu")
        self.model = CNN(input_channels, num_classes, num_levels, kernel_size, dropout, num_filters)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logger
        self.training_history = [] # {"epoch": 1, "train_loss": 0.5, "train_acc": 80, "val_loss": 0.6, "val_acc": 78},



    def fit(self, train_dataset, val_dataset=None):
        """
        Trains the TCN model for a preset number of epochs.
        
        Args:
            train_data: A PyTorch DataLoader that yields (inputs, labels, lengths)
        Returns:
            self: So the method can be chained.
        """
        patience = 35
        epochs_no_improve = 0
        min_delta = 1e-4
        best_val_loss = float("inf")
        # define loaders
        #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        # 1) extract all labels from the dataset
        # all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        # num_classes = self.model.fc.out_features
        # import numpy as np
        # counts = np.bincount(all_labels, minlength=num_classes)

        # # 2) build class_weights = 1/counts (then normalize so mean=1)
        # class_weights = torch.tensor(
        #     counts.sum() / (counts * num_classes),
        #     dtype=torch.float,
        #     device=self.device
        # )



        # # inject into your loss
        # #self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # # 3) build per-sample weights for the sampler
        # sample_weights = class_weights[all_labels]  # tensor index by label
        # from torch.utils.data import WeightedRandomSampler
        # sampler = WeightedRandomSampler(
        #     weights=sample_weights,
        #     num_samples=len(sample_weights),
        #     replacement=True
        # )

        # 4) now create a loader that uses the sampler (no shuffle!)
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,#sampler=sampler,
            collate_fn=collate_fn   
        )
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

        # Refactored Training Loop with Logger Integration

        for epoch in range(self.num_epochs):
            # --- Training Phase ---
            self.model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for sequences, labels, lengths in train_loader:
                # Move the data to device
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(sequences, lengths)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_size = sequences.size(0)
                running_train_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                total_train += batch_size
                correct_train += (predicted == labels).sum().item()

            train_loss_epoch = running_train_loss / total_train
            train_accuracy_epoch = (correct_train / total_train) * 100

                    # --- Validation Phase ---
            val_loss_epoch = None
            val_acc_epoch  = None
            val_macro_f1   = None
            val_bal_acc    = None
            val_recall_cls = None

            if val_dataset is not None:
                self.model.eval()
                running_val_loss = 0.0
                correct_val = 0
                total_val = 0

                all_val_preds = []
                all_val_labels = []

                with torch.no_grad():
                    for sequences, labels, lengths in val_loader:
                        sequences, labels, lengths = (
                            sequences.to(self.device),
                            labels.to(self.device),
                            lengths.to(self.device)
                        )
                        outputs = self.model(sequences, lengths)
                        loss = self.criterion(outputs, labels)

                        bs = sequences.size(0)
                        running_val_loss += loss.item() * bs
                        _, preds = torch.max(outputs, 1)

                        total_val += bs
                        correct_val += (preds == labels).sum().item()

                        all_val_preds.append(preds.cpu())
                        all_val_labels.append(labels.cpu())

                # aggregate
                val_loss_epoch = running_val_loss / total_val
                val_acc_epoch  = correct_val / total_val * 100

                y_true = torch.cat(all_val_labels).numpy()
                y_pred = torch.cat(all_val_preds).numpy()

                val_macro_f1   = f1_score(y_true, y_pred, average='macro')
                val_bal_acc    = balanced_accuracy_score(y_true, y_pred)
                val_recall_cls = recall_score(y_true, y_pred, average=None)

                # early stopping on val loss (or swap in macro-F1 if desired)
                if self.early_stopping:
                    if best_val_loss - val_loss_epoch > min_delta:
                        best_val_loss = val_loss_epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"No improvement for {patience} epochs. Stopping early.")
                        break

            # --- Logging ---
            entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss_epoch,
                "train_acc":  train_accuracy_epoch
            }
            if val_dataset is not None:
                entry.update({
                    "val_loss":       val_loss_epoch,
                    "val_acc":        val_acc_epoch,
                    "val_macro_f1":   val_macro_f1,
                    "val_bal_acc":    val_bal_acc,
                    #"val_recall_cls": val_recall_cls,  # array of per-class recalls
                })
            self.training_history.append(entry)

            # Print summary
            msg = (f"Epoch {epoch+1}/{self.num_epochs}  "
                f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_accuracy_epoch:.2f}%")
            if val_dataset is not None:
                msg += (f"  |  Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%  "
                        f"Macro-F1: {val_macro_f1:.3f}, BalAcc: {val_bal_acc:.3f}")
            print(msg)

        return self

    def predict(self, input_dataset):
        """
        Makes predictions for given inputs.
        
        Args:
            input_dataset: A PyTorch Dataset where each sample is a tuple (input, label, length).
                The input is a tensor of shape (seq_length, channels).
                If lengths are not provided, assumes full-length sequences.
                
        Returns:
            numpy.ndarray: Predicted class indices for all input samples.
        """
        data_loader = DataLoader(input_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for sequences, labels, lengths in data_loader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                outputs = self.model(sequences, lengths)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu())
                
        # Concatenate all predictions into a single tensor, then convert to numpy array.
        all_preds = torch.cat(all_preds, dim=0)
        return all_preds.numpy()
    

