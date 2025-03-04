import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_relu = False
    use_improved_weight_init = True
    use_improved_sigmoid = True    
    use_momentum = True
    use_relu = False
    learning_rate = .02

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64, val_history_64 = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!



    neurons_per_layer = [32, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer.train(num_epochs)
    neurons_per_layer = [128, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128 = trainer.train(num_epochs)


    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history_64["loss"], "64", npoints_to_average=10)
    utils.plot_loss(train_history_32["loss"], "32", npoints_to_average=10)
    utils.plot_loss(train_history_128["loss"], "128", npoints_to_average=10)

    plt.ylim([0, .4])

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.01])
    utils.plot_loss(val_history_64["accuracy"], "64")
    utils.plot_loss(val_history_32["accuracy"], "32")
    utils.plot_loss(val_history_128["accuracy"], "128")
    plt.ylabel("Validation Accuracy")

    plt.legend()
    plt.savefig("task4ab_train_loss_validation_accuracy.png")
    plt.show()



if __name__ == "__main__":
    main()
