from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape: ", X_train.shape)
print("Test data shape:", X_test.shape)

# data preprocessing
X_train = X_train.reshape(-1, 784).astype('float32') / 255
X_test = X_test.reshape(-1, 784).astype('float32') / 255

model = Sequential([
  Dense(128, activation='relu', input_shape=(784,)), # hidden layer with 128 neurons
  Dense(10, activation='softmax') # output layer with 10 neurons
])

model.summary()