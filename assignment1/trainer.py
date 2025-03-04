import numpy as np
import utils


class BaseTrainer:

    def __init__(
            self,
            model,
            learning_rate: float,
            batch_size: int,
            shuffle_dataset: bool,
            X_train: np.ndarray, Y_train: np.ndarray,
            X_val: np.ndarray, Y_val: np.ndarray,) -> None:
        """
            Initialize the trainer responsible for performing the gradient descent loop.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model
        self.shuffle_dataset = shuffle_dataset

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        pass

    def train_step(self):
        """
            Perform forward, backward and gradient descent step here.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        pass

    def train(
            self,
            num_epochs: int):

        """
        Training loop for model.
        Implements stochastic gradient descent with num_epochs passes over the train dataset.
        Returns:
            train_history: a dictionary containing loss and accuracy over all training steps
            val_history: a dictionary containing loss and accuracy over a selected set of steps
        """
        # Utility variables
        num_batches_per_epoch = self.X_train.shape[0] // self.batch_size
        num_steps_per_val = num_batches_per_epoch // 5
        print("num_batches_per_epoch", num_batches_per_epoch, "num_steps_per_val", num_steps_per_val)
        # A tracking value of loss over all training steps
        train_history = dict(
            loss={},
            accuracy={}
        )
        val_history = dict(
            loss={},
            accuracy={}
        )
        num_non_improvements = 0
        best_val_loss = float('inf')
        global_step = 0
        for epoch in range(num_epochs):
            train_loader = utils.batch_loader(
                self.X_train, self.Y_train, self.batch_size, shuffle=self.shuffle_dataset)
            for X_batch, Y_batch in iter(train_loader):
                loss = self.train_step(X_batch, Y_batch)
                # Track training loss continuously
                train_history["loss"][global_step] = loss

                # Track validation loss / accuracy every time we progress 20% through the dataset
                if global_step % num_steps_per_val == 0:
                    val_loss, accuracy_train, accuracy_val = self.validation_step()
                    train_history["accuracy"][global_step] = accuracy_train
                    val_history["loss"][global_step] = val_loss
                    val_history["accuracy"][global_step] = accuracy_val

                    # TODO (Task 2d): Implement early stopping here.
                    # You can access the validation loss in val_history["loss"]

                    """
                    val_10_history = list(val_history["loss"].values())[-10:]
                    #test = (val_10_history > val_loss)
                    val_10_history_sum = np.sum((val_10_history >= list(val_history["loss"].values())[-11]))
                   # print("global_step:", global_step, "epoch:", epoch)
                    if val_10_history_sum == 0:
                        print("Early stop", val_10_history)
                        print("epoch:", epoch)
                        return train_history, val_history
                    """
                global_step += 1
        #print(val_10_history, val_10_history_sum, "test:\n", test)
        print(global_step)
        return train_history, val_history
