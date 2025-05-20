class EarlyStopping:
    def __init__(self, patience: int = 1, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def __call__(self, current_loss: float) -> bool:
        # return True if we should stop
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience