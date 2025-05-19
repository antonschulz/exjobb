import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch.utils.data import DataLoader
from utils.data_loading import collate_fn
from sklearn.metrics import f1_score, balanced_accuracy_score


class LSTMMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, mlp_output_size,
                 batch_first=True, dropout=0.0, bidirectional=False):
        super(LSTMMLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        # MLP head takes the concatenated last hidden state if bidirectional
        self.fc = nn.Linear(hidden_size * self.num_directions, mlp_output_size)

    def forward(self, x, lengths=None, hidden=None):

        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
            if hidden is not None:
                packed_out, (h_n, c_n) = self.lstm(packed, hidden)
            else:
                packed_out, (h_n, c_n) = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_out, batch_first=self.batch_first)
        else:
            if hidden is not None:
                output, (h_n, c_n) = self.lstm(x, hidden)
            else:
                output, (h_n, c_n) = self.lstm(x)

        # h_n: (num_layers * num_directions, batch, hidden_size)
        # select the last layer for each direction
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, self.num_directions, h_n.size(1), self.hidden_size)
            last_forward = h_n[-1, 0]  # last layer, forward
            last_backward = h_n[-1, 1]  # last layer, backward
            last_hidden = torch.cat([last_forward, last_backward], dim=1)  # (batch, hidden_size*2)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_size)

        out = self.fc(last_hidden)
        return out, (h_n, c_n)
    
class LSTM_wrapper:
    def __init__(
        self,
        input_size: int=2,
        num_classes: int=4,
        hidden_size=128,
        num_layers: int = 2,
        bi_lstm: bool = False,
        dropout: float = 0.0,
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device=None,
        logger=None,
    ):
        """
        A wrapper that instantiates your LSTMModel and provides fit & predict.
        """
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.device = device or torch.device("cpu")
        self.logger = logger
        self.training_history = []

        # instantiate your LSTMModel
        self.model = LSTMMLP(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, mlp_output_size=num_classes, batch_first=True, bidirectional=bi_lstm).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_dataset, val_dataset=None, batch_size=16, debug=False):
        all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        num_classes = self.model.fc.out_features
        import numpy as np
        counts = np.bincount(all_labels, minlength=num_classes)

        # 2) build class_weights = 1/counts (then normalize so mean=1)
        class_weights = torch.tensor(
            counts.sum() / (counts * num_classes),
            dtype=torch.float,
            device=self.device
        )

        #self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        sample_weights = class_weights[all_labels]  
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            sampler=sampler,
            collate_fn=collate_fn
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        for epoch in range(1, self.num_epochs + 1):
            # --- Training Phase ---
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for x, y, lengths in train_loader:
                x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(x, lengths)
                loss = self.criterion(logits, y)
                loss.backward()

                if debug:
                    # Print gradient stats before the optimizer step
                    print(f"\n[DEBUG] Epoch {epoch} gradients:")
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(f"  {name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")

                self.optimizer.step()


                # accumulate metrics
                bs = x.size(0)
                running_loss += loss.item() * bs
                preds = logits.argmax(dim=1)
                total += bs
                correct += (preds == y).sum().item()

            train_loss_epoch = running_loss / total
            train_acc_epoch = 100 * correct / total

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
                        outputs, _ = self.model(sequences, lengths)
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

                # early stopping on val loss (or swap in macro-F1 if desired)
                # if self.early_stopping:
                #     if best_val_loss - val_loss_epoch > min_delta:
                #         best_val_loss = val_loss_epoch
                #         epochs_no_improve = 0
                #     else:
                #         epochs_no_improve += 1
                #     if epochs_no_improve >= patience:
                #         print(f"No improvement for {patience} epochs. Stopping early.")
                #         break

            # --- Logging ---
            entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss_epoch,
                "train_acc":  train_acc_epoch
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
                f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%")
            if val_dataset is not None:
                msg += (f"  |  Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%  "
                        f"Macro-F1: {val_macro_f1:.3f}, BalAcc: {val_bal_acc:.3f}")
            print(msg)

        return self


    def predict(self, dataset, batch_size=1):
        """
        Returns numpy array of predicted class indices for every sample in dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x, _, lengths in loader:
                x, lengths = x.to(self.device), lengths.to(self.device)
                logits, _ = self.model(x, lengths)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
        return torch.cat(all_preds).numpy()