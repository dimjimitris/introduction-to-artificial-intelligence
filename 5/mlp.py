import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

class MultilayerPerceptron(nn.Module):
    """
    A simple multilayer perceptron model for classification tasks.
    The model consists of a sequence of fully connected layers with
    a specified activation function in between them. The number of
    layers and the size of each layer can be specified by the user.

    Parameters:
    - input_size: The size of the input data
    - output_size: The size of the output data
    - layer_sizes: A list of sizes for each hidden layer
    - activation_function: The activation function to use between layers
    """
    def __init__(
            self,
            input_size = 28*28,
            output_size = 10,
            layer_sizes = [120, 84],
            activation_function = nn.ReLU(),
        ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function

        # Define the layers
        self.flatten = nn.Flatten()   # Prepares input for fully connected layers
        self.layers = nn.Sequential() # Sequential container for network layers
        
        # Construct neural network layers dynamically based on 'layer_sizes'
        # (works even when layer_sizes is empty)
        complete_layer_sizes = [input_size] + layer_sizes + [output_size]
        for i in range(len(complete_layer_sizes) - 1):
            # Add layers to the network
            self.layers.append(nn.Linear(complete_layer_sizes[i], complete_layer_sizes[i + 1]))
            # Add activation functions after each layer except the last one (the output layer)
            if i < len(complete_layer_sizes) - 2:
                self.layers.append(self.activation_function)
        # In training we use CrossEntropyLoss, which already uses softmax, thus
        # we don't add an activation layer here

    # Forward the input across all layers of the network
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Function to evaluate the model on test data
def test(model, data, batch_size=64, device = 'cpu'):
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval() # Put the model in evaluation mode

    predlist = torch.zeros(0, dtype=torch.long, device=device)
    lbllist = torch.zeros(0, dtype=torch.long, device=device)

    with torch.no_grad(): # No gradients needed for evaluation
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get the predictions from the model outputs
            _, predicted = torch.max(outputs.data, 1)

            # Append predictions and labels for accuracy calculation
            predlist = torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

    
    #print('Confusion Matrix')
    #print(confusion_matrix(lbllist.numpy(), predlist.numpy()))
    #print('Classification Report')
    #print(classification_report(lbllist.numpy(), predlist.numpy()))

    return predlist, lbllist

# Helper function to calculate accuracy from the predictions
def get_accuracy_from_test(model, data, batch_size=64, device = 'cpu'):
    predlist, lbllist = test(model, data, batch_size, device)
    return accuracy_score(lbllist.numpy(), predlist.numpy())

# Main function to train the model
def train(
        model,
        train_data,
        test_data,
        num_epochs,
        optimizer_type = 'SGD',
        learning_rate = 0.001,
        batch_size = 64,
        device = 'cpu',
    ):
    loss_function = nn.CrossEntropyLoss() # Loss function for classification

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize the optimizer based on the type specified
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD-Momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    model.train() # Put the model in training mode

    # Lists to store loss and accuracy values over epochs
    loss_values = []
    accuracy_values_train = []
    accuracy_values_test = []

    # Loop over the epochs
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}')

        local_loss_values = []

        # Loop over the batches of data
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad() # Clear the gradients from previous steps
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_function(outputs.squeeze(-1), labels) # Compute the loss

            loss.backward() # Backpropagation to compute gradients

            optimizer.step() # Update model parameters

            local_loss_values.append(loss.item()) # Store the loss value

        # Compute and store mean loss for the epoch
        mean_loss = np.mean(local_loss_values)
        loss_values.append(local_loss_values)
    
        print(f'--Epoch {epoch + 1}--')
        print(f'Mean Loss: {mean_loss}')

        # Calculate and store accuracy for the training and test sets
        accuracy_score_train = get_accuracy_from_test(model, train_data, batch_size, device)
        accuracy_score_test = get_accuracy_from_test(model, test_data, batch_size, device)

        print(f'Accuracy Score On Train Set: {accuracy_score_train}')
        print(f'Accuracy Score On Test Set: {accuracy_score_test}')

        accuracy_values_train.append(accuracy_score_train)
        accuracy_values_test.append(accuracy_score_test)

    # Flatten the list of loss values to make it easier to plot
    loss_values = np.array(loss_values).flatten()
            
    return loss_values, accuracy_values_train, accuracy_values_test