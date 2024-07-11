import random
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mlp import (
    MultilayerPerceptron,
    train,
    test,
)

def set_seed(seed):
    """Set the random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)

    # Determine the device to use for tensor computations (GPU or CPU)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define a transformation to convert images to tensors
    Transform = transforms.ToTensor()

    # Load the MNIST dataset
    """
    The MNIST dataset is a dataset of 60,000 28x28 grayscale
    images of the 10 digits, along with a test set of 10,000
    images.
    """
    mnist_train = datasets.MNIST(
        root='',
        train=True,
        download=True,
        transform=Transform)
    
    mnist_test = datasets.MNIST(
        root='',
        train=False,
        transform=Transform)
    
    print("\n","--Information about the sets--","\n")

    print(mnist_train)
    print()
    print(mnist_test)

    print("\n","--Information about some image--","\n")

    # Get the first image and its label
    image, label = mnist_train[0]
    print("Image shape:", image.shape)
    print("Label:", label)

    # Plot the image
    # images are 28x28 pixels in gray scale
    #plt.imshow(image.reshape((28, 28)), cmap='gray')
    #plt.show()

    # default configuration
    default_config = {
        'lr': 0.002,
        'batch_size': 100,
        'hidden_layers': [200, 50],
        'optimizer_type': 'Adam',
    }

    # parameter grid to test different configurations
    parameter_grid = {
        'lr': [0.008, 0.016],
        'batch_size': [1, 1000],
        'hidden_layers': [
            [],             # No hidden layers
            [200, 100, 50], # 3 hidden layers with respective sizes
            [100, 50],      # A narrower network
            [200, 100],     # A wider network
        ],
        'optimizer_type': ['SGD', 'SGD-Momentum'],
    }

    # Start with the default configuration
    configurations = [default_config]

    # create configurations that vary one parameter at a time
    for param, values in parameter_grid.items():
        for value in values:
            config = default_config.copy()
            config[param] = value
            configurations.append(config)

    # train and test the models
    results = [] # Store the results from each configuration

    # this line is for debugging purposes
    #configurations = configurations[:1]

    for i, config in enumerate(configurations):
        set_seed(seed) # Ensure each configuration uses the same seed
        # print the configuration
        print(f"Configuration {i + 1}/{len(configurations)}")
        print(config)

        # initialize the model with the current configuration
        mlp = MultilayerPerceptron(
            input_size=28*28,
            output_size=10,
            layer_sizes=config['hidden_layers'],
        )

        # train the model and gather the loss and accuracy metrics
        (
            loss_values,
            accuracy_values_train,
            accuracy_values_test,
        ) = train(
            model=mlp,
            train_data=mnist_train,
            test_data=mnist_test,
            num_epochs=6,
            optimizer_type=config['optimizer_type'],
            learning_rate=config['lr'],
            batch_size=config['batch_size'],
            device=device
        )

        results.append({
            'config': config,
            'loss_values': loss_values,
            'accuracy_values_train': accuracy_values_train,
            'accuracy_values_test': accuracy_values_test,
        })

    # Plot the loss values for each configuration
    max_subplots_per_fig = 6
    num_figs = len(results) // max_subplots_per_fig + (len(results) % max_subplots_per_fig > 0)

    for fig_idx in range(num_figs):
        # Indexes for calculating which subset of the 'results' list should be plotted in this figure
        start_idx = fig_idx * max_subplots_per_fig
        end_idx = min(start_idx + max_subplots_per_fig, len(results))
        
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        axs = axs.flatten()

        for plot_idx, result_idx in enumerate(range(start_idx, end_idx)):
            result = results[result_idx]

            axs[plot_idx].plot(result['loss_values'])
            axs[plot_idx].set_title(f"(LR, BS, HL, OT) = {list(result['config'].values())}")
            axs[plot_idx].set_xlabel("Iteration")
            axs[plot_idx].set_ylabel("Loss")

        plt.tight_layout()
    plt.show()

    # Plot the accuracy values for each configuration
    for fig_idx in range(num_figs):
        # Indexes for calculating which subset of the 'results' list should be plotted in this figure
        start_idx = fig_idx * max_subplots_per_fig
        end_idx = min(start_idx + max_subplots_per_fig, len(results))
        
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        axs = axs.flatten()

        for plot_idx, result_idx in enumerate(range(start_idx, end_idx)):
            result = results[result_idx]

            axs[plot_idx].plot(range(1, len(result['accuracy_values_train']) + 1), result['accuracy_values_train'], label='train')
            axs[plot_idx].plot(range(1, len(result['accuracy_values_test']) + 1), result['accuracy_values_test'], label='test')
            axs[plot_idx].set_title(f"(LR, BS, HL, OT) = {list(result['config'].values())}")
            axs[plot_idx].set_xlabel("Epoch")
            axs[plot_idx].set_ylabel("Accuracy")
            axs[plot_idx].legend()

        plt.tight_layout()
    plt.show()
