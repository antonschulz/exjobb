import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from utils.data_loading import collate_fn


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
        num_layers: int = 2,
        fc_units=None,
        bi_lstm: bool = False,
        dropout_lstm: float = 0.0,
        dropout_fc: float = 0.0,
        num_epochs: int = 10,
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
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        for epoch in range(1, self.num_epochs + 1):
            # --- train ---
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for x, y, lengths in train_loader:
                x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x, lengths)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                bs = x.size(0)
                running_loss += loss.item() * bs
                preds = logits.argmax(dim=1)
                total += bs
                correct += (preds == y).sum().item()

            train_loss = running_loss / total
            train_acc = 100 * correct / total

            # --- validate ---
            if val_dataset is not None:
                self.model.eval()
                v_loss, v_correct, v_total = 0.0, 0, 0
                with torch.no_grad():
                    for x, y, lengths in val_loader:
                        x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                        logits = self.model(x, lengths)
                        loss = self.criterion(logits, y)
                        bs = x.size(0)
                        v_loss += loss.item() * bs
                        preds = logits.argmax(dim=1)
                        v_total += bs
                        v_correct += (preds == y).sum().item()
                val_loss = v_loss / v_total
                val_acc = 100 * v_correct / v_total
            else:
                val_loss, val_acc = None, None

            # --- record & log ---
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            self.training_history.append(record)

            print(
                f"Epoch {epoch}/{self.num_epochs}  "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%  "
                f"Val Loss={val_loss if val_loss is not None else 'N/A'}"
                f"{', Val Acc={:.2f}%'.format(val_acc) if val_acc is not None else ''}"
            )
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
                logits = self.model(x, lengths)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
        return torch.cat(all_preds).numpy()