class EarlyStopper: 
    def __init__(self, patience=5, min_delta=0.0):
        """
        Initializes the EarlyStopper.

        Args:
        Reference: https://keras.io/api/callbacks/early_stopping/        
            patience (int): Number of iterations with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        """
        Checks if training should be stopped based on the validation loss.
        Args:
            validation_loss (float): The current validation loss.
        """

        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset counter if validation loss improves
            self.print_status(validation_loss)
        else:
            self.counter += 1  # increment counter if validation loss does not improve
            self.print_status(validation_loss)
            if self.counter == self.patience:
                return True  # stop training
        return False  # continue training
    
    def print_status(self, validation_loss):
            print('\n' + '*' * 30 + '\n')
            print(f'EarlyStopper: validation_loss={validation_loss:.4f}, min_validation_loss={self.min_validation_loss - self.min_delta:.4f}, counter={self.counter}/{self.patience}')
            print('\n' + '*' * 30 + '\n')