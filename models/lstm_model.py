import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from utils.data_loading import collate_fn
from .helpers import EarlyStopping
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score




class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        bi_lstm: bool = False,
        dropout_lstm: float = 0.1,
        fc_units: list[int] = [32],
        dropout_fc: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
            bidirectional=bi_lstm,
            batch_first=True,
        )
        lstm_out_dim = hidden_size * (2 if bi_lstm else 1)

        # build FC head
        layers = []
        in_dim = lstm_out_dim
        for dim in fc_units:
            layers += [nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout_fc)]
            in_dim = dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        # pack
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        # if you want mask‑based pooling instead:
        # out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # mask = ...
        # feat = (out * mask).sum(1)/lengths.unsqueeze(1)
        # else use h_n:
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            feat = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            feat = h_n[-1]
        return self.fc(feat)

    
class LSTM_model:
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size=None,
        early_stopping=True,
        num_layers: int = 2,
        fc_units=None,
        bi_lstm: bool = False,
        dropout_lstm: float = 0.0,
        dropout_fc: float = 0.0,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size=16,
        early_stop_epochs=None,
        device=None,
        logger=None,
    ):
        """
        A wrapper that instantiates your LSTMModel and provides fit & predict.
        """
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.early_stopping=early_stopping
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.logger = logger
        self.training_history = []
        self.early_stop_epochs = early_stop_epochs

        # early stopping
        self.patience = 35
        self.min_delta = 1e-4

        # instantiate your LSTMModel
        self.model = LSTMModel(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            fc_units=fc_units,
            bi_lstm=bi_lstm,
            dropout_lstm=dropout_lstm,
            dropout_fc=dropout_fc,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_dataset, val_dataset=None, batch_size=16):
        """
        Train for self.num_epochs, logging train/val loss & accuracy.
        """

    
        stopper = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta
        )
        from collections import Counter

        # 1) Gather your labels (list or 1D array of ints 0…C-1)
        all_labels = [int(train_dataset[i][1]) for i in range(len(train_dataset))]
        counts = Counter(all_labels)
        num_classes = len(counts)
        total_samples = sum(counts.values())

        # 2) Base weights = inverse frequency
        #    w_raw[k] = total_samples / (counts[k] * num_classes)
        w_raw = torch.tensor(
            [ total_samples / (counts[k] * num_classes) for k in range(num_classes) ],
            dtype=torch.float,
            device=self.device
        )

        # 3) Normalize to sum to 1.0
        w = w_raw / w_raw.sum()

        # 4) Enforce a minimum fraction per class (e.g. 5%)
        #min_frac = 0.025  # replace with X (in [0,1])
        #w = torch.clamp(w, min=min_frac)

        # 5) Renormalize so they still sum to 1.0
        w = w / w.sum()
       

        # 6) Plug into your loss
        #self.criterion = nn.CrossEntropyLoss(weight=w)

        # # 3) build per-sample weights for the sampler
        sample_weights = w[all_labels]  # tensor index by label
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 4) now create a loader that uses the sampler (no shuffle!)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            #shuffle=True,
            collate_fn=collate_fn
        )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

                # Refactored Training Loop with Logger Integration
        epochs_to_train = self.early_stop_epochs if self.early_stop_epochs else self.num_epochs


        for epoch in range(epochs_to_train):
            from collections import Counter

        # initialize counter
            label_counts = Counter()
            
            # --- Training Phase ---
            self.model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0


            for sequences, labels, lengths in train_loader:
                # Move the data to device
                label_counts.update(labels.tolist())
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
            for label, count in sorted(label_counts.items()):
                    print(f"Label {label}: {count} samples")

                    # --- Validation Phase ---
            val_loss_epoch = None
            val_acc_epoch  = None
            val_macro_f1   = None
            val_bal_acc    = None

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
                })
            self.training_history.append(entry)

            # Print summary
            msg = (f"Epoch {epoch+1}/{self.num_epochs}  "
                f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_accuracy_epoch:.2f}%")
            if val_dataset is not None:
                msg += (f"  |  Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%  "
                        f"Macro-F1: {val_macro_f1:.3f}, BalAcc: {val_bal_acc:.3f}")
            print(msg)

            if self.early_stopping and val_dataset and stopper(val_loss_epoch):
                # record how many epochs we just did
                self.early_stop_epochs = epoch + 1 - self.patience
                print(f"No improvement for {self.patience} epochs. Stopping early at epoch {self.early_stop_epochs}.")
                break

        return self

    def predict(self, dataset, batch_size=1):
        """
        Makes predictions for given inputs.
        
        Args:
            input_dataset: A PyTorch Dataset where each sample is a tuple (input, label, length).
                The input is a tensor of shape (seq_length, channels).
                If lengths are not provided, assumes full-length sequences.
                
        Returns:
            numpy.ndarray: Predicted class indices for all input samples.
        """
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
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