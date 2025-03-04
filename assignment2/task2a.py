import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    mean_value = np.mean(X)
    std_dev = np.std(X)
    X = (X - mean_value) / std_dev
    print("Mean value:", mean_value, "Standard deviation:", std_dev)
    print("X.shape =", X.shape)
    X = np.hstack((X, np.ones((X.shape[0], 1)))) # bias trick
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    Cn = - np.sum(targets * np.log(outputs), axis=1)
    return Cn.mean()
    #raise NotImplementedError

def sigmoid(X, use_improved_sigmoid):
    if use_improved_sigmoid:
        return(1.7159*np.tanh(2/3*X))
    else:
        return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X, use_improved_sigmoid):
    if use_improved_sigmoid:
        return 1.1439 * (-np.tanh(2/3*X)**2 + 1)
    else:
        sig = sigmoid(X, False)
        return sig * (1 - sig)

def softmax(X):
    exp = np.exp(X)
    a = exp / exp.sum(axis=1, keepdims=True) 
    return a

class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
       
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        self.hidden_layer_output = []
        self.hidden_layer_input = []
        #self.z_j = None
        #self.a_j = None
        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        self.num_hidden_layers = len(self.neurons_per_layer) - 1
        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]


        print("ws.shape", self.ws[0].shape, self.ws[1].shape)
        if not self.use_improved_weight_init:
            for i in range(len(self.ws)):
                self.ws[i] = np.random.uniform(-1, 1, self.ws[i].shape)
        else:
            for i in range(len(self.ws)):
                fan_in = self.ws[i].shape[0]
                self.ws[i] = np.random.normal(loc=0, scale=1/np.sqrt(fan_in), size=self.ws[i].shape)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        #self.ws[0].shape = (785, 64)
        #self.ws[1].shape = (64, 10)

        input_data = X
        self.hidden_layer_input = []
        self.hidden_layer_output = []
        for layer_index in range(self.num_hidden_layers):
            # Compute the linear transformation for the current hidden layer
            z_j = input_data @ self.ws[layer_index]
            self.hidden_layer_input.append(z_j)
            # Apply the activation function for the current hidden layer
            a_j = sigmoid(z_j, self.use_improved_sigmoid)
            
            # Update the input for the next hidden layer or output layer
            self.hidden_layer_output.append(a_j)
            input_data = a_j


        z_k = input_data @ self.ws[-1]
        y_output = softmax(z_k)
        


        """
        self.z_j = X @ self.ws[0]
        self.hidden_layer_input = [self.z_j]
        self.a_j = sigmoid(self.z_j, self.use_improved_sigmoid)
        self.hidden_layer_output = [self.a_j]
        z_k = self.a_j @ self.ws[1] 
        y_output = softmax(z_k)
        """
        return y_output

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # X: images of shape [batch size, 785]
        # For example, self.grads[0] will be the gradient for the first hidden layer

        #self.ws[0].shape = (785, 64) w_ji = grad[0].shape
        #self.ws[1].shape = (64, 10) w_kj = grad[1].shape 
        #delta_j.shape = (64,10)
        #y (batch size, 10)
        #a_j (batch size,64)
        #z_k (batchsize, 10)
       
        # Initialize a list to store gradients for each layer
        #self.grads = [0] * (self.num_hidden_layers + 1)

        # Compute the gradient for the output layer
        
        self.zero_grad()
        delta_k = -(targets - outputs)
        self.grads[-1] = self.hidden_layer_output[-1].T.dot(delta_k) / X.shape[0]
        delta_j = delta_k
        
        # Backpropagate the error through the hidden layers
        for layer_index in range(self.num_hidden_layers - 1, -1, -1):
            
            delta_j = (sigmoid_derivative(self.hidden_layer_input[layer_index], self.use_improved_sigmoid) * delta_j.dot(self.ws[layer_index + 1].T))
        
            if layer_index > 0:
                self.grads[layer_index] = self.hidden_layer_output[layer_index - 1].T.dot(delta_j) / X.shape[0] 
            else:
                self.grads[layer_index] = X.T.dot(delta_j) / X.shape[0]
        


        """
        self.grads = []
        delta_k = -(targets - outputs)
        delta_j = sigmoid_derivative(self.z_j, self.use_improved_sigmoid) * (delta_k).dot(self.ws[1].T)
        self.grads.append(X.T.dot(delta_j) / X.shape[0])
        self.grads.append(self.a_j.T.dot(delta_k) / X.shape[0])
        """


        #self.grad = -(X.T @ np.subtract(targets, outputs))
        # outputs: outputs of model of shape: [batch size, num_outputs]
        #X: images of shape [batch size, 785]
        # w shape [785, num_outputs]
        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    # Ensure Y is a 1D array
    Y = Y.flatten()
    # Get the number of examples
    Num_examples = len(Y)
    # Create an empty array with zeros
    Y_one_hot = np.zeros((Num_examples, num_classes))
    # Use advanced indexing to set the appropriate elements to 1
    Y_one_hot[np.arange(Num_examples), Y] = 1
    return Y_one_hot
    #raise NotImplementedError


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
    #print("test passed")

if __name__ == "__main__":
    main()
