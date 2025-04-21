import torch.nn as nn
import torch
import torch.nn.functional as F
from Ex1.Task4 import IrisPaser
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the file paths
test_file = "/Datasets/iris/test.csv"
train_file = "/Datasets/iris/train.csv"


# Define the neural network architecture
class IrisNetwork(nn.Module):
    def __init__(self, in_features=4, h1=2, h2=3, out_features=3):
        super(IrisNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)  # Input layer to hidden layer
        self.fc2 = nn.Linear(h1, h2)   # Hidden layer 1 to hidden layer 2
        self.out = nn.Linear(h2, out_features) # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Activation function for hidden layer 1
        x = F.relu(self.fc2(x)) # Activation function for hidden layer 2
        x = self.out(x) # No activation function for output layer
        return x


# Initialize the model
model = IrisNetwork()

# Load the dataset
x_train = IrisPaser.get_attributes(train_file)
y_train = IrisPaser.get_labels(train_file)
x_test = IrisPaser.get_attributes(test_file)
y_test = IrisPaser.get_labels(test_file)

# encode labels
# one-hot encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = np.eye(3)[y_train]
y_test = np.eye(3)[y_test]

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define loss function and optimizer -> what works best?
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train the model
print("Training the model:")
num_epochs = 500
for i in range(num_epochs):
    y_predict = model.forward(x_train)
    loss = criterion(y_predict, y_train)

    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss}")

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print()

# Evaluate the model
print("Evaluating the model on test data:")
with torch.no_grad(): # turn off gradients
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
    print(f"Test Loss: {loss}")

    # Get the predicted clasS
    _, predicted = torch.max(y_eval, 1)
    # Convert predictions to class labels
    predicted = label_encoder.inverse_transform(predicted)
    # Convert y_test to class labels
    _, actual = torch.max(y_test, 1)
    actual = label_encoder.inverse_transform(actual)

    right = 0
    for i in range(len(x_test)):
        print(f"{i+1}) {y_eval[i]} -> Predicted: {predicted[i]}, Actual: {actual[i]} ")
        if predicted[i] == actual[i]:
            right += 1
    print(f"{right} correct. Accuracy: {right / len(x_test) * 100}%")


