# CIFAR-10 dataset 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# check shapes
print("Training data shape", X_train.shape)
print("Test data shape", X_test.shape)

# for CNNs, it's beneficial to scale the pixel values to a range of 0-1
# normalize pixel values between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0



model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  MaxPooling2D((2,2)),
  Conv2D(64, (3, 3,), activation='relu'),
  MaxPooling2D((2,2)),
  Conv2D(64, (3, 3), activation='relu'),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax') # 10 classes in CIFAR-10
])

# Display the model architecture
model.summary()

# for a multi-class classification problem, use sparse_categorical_crossentropy as the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")