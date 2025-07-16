
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, logger, patience=5, verbose=False, delta=0.0):
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        # if this is the first epoch or the validation loss improved enough, save the model
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            # save a deep copy of the best model state (on CPU to avoid device issues later)
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
            if self.verbose:
                self.logger.log_message(f'Validation loss improved to {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                self.logger.log_message(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True