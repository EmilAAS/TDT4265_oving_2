import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    #accuracy = 0
    """y_hat = model.forward(X)
    pred = (y_hat >= 0.5).astype(int)
    accuracy = np.sum(pred==targets)/y_hat.shape[0]"""
    y_hat = model.forward(X)
    max_index = np.argmax(y_hat, axis=1)
    pred = one_hot_encode(max_index, 10) #ndarray [0,0,1,0,0,0,0,0,0]
    assert pred.shape == targets.shape,\
            f"Grad shape: {pred}, w: {targets}"
    equal_rows = np.all(pred == targets, axis=1)
    accuracy = np.sum(equal_rows)/y_hat.shape[0]
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        #loss = 0
        y_hat = self.model.forward(X_batch)
        self.model.backward(X_batch, y_hat, Y_batch)
        loss = cross_entropy_loss(Y_batch, y_hat)
        self.model.w = np.subtract(self.model.w, self.learning_rate * self.model.grad)

        return loss

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
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0.0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)
    
    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    
    # Plotting of softmax weights (Task 4b)
    w2 = model.w[:-1, :]
    weight = w2.reshape((28, 28, 10))
    weight = np.concatenate([weight[:, :, i] for i in range(10)], axis=1)
    plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    
    for element in l2_lambdas:
        model1 = SoftmaxModel(l2_reg_lambda=element)
        trainer = SoftmaxTrainer(
            model1, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
        
        plt.ylim([0.5, 1])
        utils.plot_loss(val_history_reg01["accuracy"], f"Validation Accuracy: lambda ={element}")
        plt.xlabel("Number of Training Steps")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()
    

    # Task 4d - Plotting of the l2 norm for each weight
    for element in l2_lambdas:
        model1 = SoftmaxModel(l2_reg_lambda=element)
        trainer = SoftmaxTrainer(
            model1, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
        length = np.sum(model1.w**2,axis=None)
        
        print(length)
        plt.scatter(element, length, label=f"L2 norm for lamda = {element}")             
        plt.xlabel("Lamda")
        plt.ylabel("L_2 norm")
        plt.legend()

    plt.savefig("task4d_l2_reg_norms.png")    
    plt.show()
    


if __name__ == "__main__":
    main()
