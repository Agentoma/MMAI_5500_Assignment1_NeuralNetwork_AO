import numpy as np

# Load the data from the present working directory
fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# Standardize the input features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Xavier initialization
        limit = np.sqrt(6 / (n_inputs + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

    def backward(self, dz):
        # Backward pass
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dz, self.weights.T)

class LeakyReLU:
    def forward(self, z):
        # Apply Leaky ReLU activation
        self.z = z
        self.activity = np.where(z > 0, z, 0.01 * z)

    def backward(self, dactivity):
        # Backward pass through Leaky ReLU
        self.dz = np.where(self.z > 0, dactivity, 0.01 * dactivity)

class Softmax:
    def forward(self, z):
        # Apply Softmax activation
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        # Compute cross-entropy loss
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean()

    def backward(self, probs, oh_y_true):
        # Gradient of the loss
        batch_sz = probs.shape[0]
        self.dz = (probs - oh_y_true) / batch_sz

class SGD:
    def __init__(self, learning_rate=0.1):
        # Initialize optimizer
        self.learning_rate = learning_rate

    def update_params(self, layer):
        # Update parameters
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

def predictions(probs):
    # Convert probabilities to class predictions
    y_preds = np.argmax(probs, axis=1)
    return y_preds

def accuracy(y_preds, y_true):
    # Calculate accuracy
    return np.mean(y_preds == y_true)

def forward_pass(X, y_true, oh_y_true):
    # Perform a forward pass through the network
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    dense3.forward(activation2.activity)
    probs = output_activation.forward(dense3.z)
    loss = crossentropy.forward(probs, oh_y_true)
    y_preds = predictions(probs)
    acc = accuracy(y_preds, y_true)
    return probs, loss, acc

def backward_pass(probs, y_true, oh_y_true):
    # Perform a backward pass through the network
    crossentropy.backward(probs, oh_y_true)
    dense3.backward(crossentropy.dz)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)

# Hyperparameters
n_inputs = X_train.shape[1]
n_hidden1 = 4
n_hidden2 = 8
n_outputs = len(np.unique(y_train))
learning_rate = 0.1
epochs = 10
batch_sz = 32
n_batch = int(np.ceil(X_train.shape[0] / batch_sz))

# Initialize the network
dense1 = DenseLayer(n_inputs, n_hidden1)
activation1 = LeakyReLU()
dense2 = DenseLayer(n_hidden1, n_hidden2)
activation2 = LeakyReLU()
dense3 = DenseLayer(n_hidden2, n_outputs)
output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD(learning_rate=learning_rate)

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    # Shuffle the training data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    for batch_i in range(n_batch):
        # Get a batch of data
        start = batch_i * batch_sz
        end = start + batch_sz
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        # One-hot encode the labels
        oh_y_batch = np.eye(n_outputs)[y_batch]
        # Forward pass
        probs, loss, acc = forward_pass(X_batch, y_batch, oh_y_batch)
        # Backward pass
        backward_pass(probs, y_batch, oh_y_batch)
        # Update parameters
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
    # Evaluate on the training set
    oh_y_train = np.eye(n_outputs)[y_train]
    _, train_loss, train_acc = forward_pass(X_train, y_train, oh_y_train)
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

# Evaluate on the test set
oh_y_test = np.eye(n_outputs)[y_test]
probs, test_loss, test_acc = forward_pass(X_test, y_test, oh_y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
