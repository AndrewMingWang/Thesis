from datasets import *
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def train_model(input_folder, output_folder):

    # Create dataset
    data = ProcessedMeshDataset(input_folder + "\\processed_data", input_folder + "\\label.txt")

    # Specify hyperparameters
    num_epochs = 50
    batch_size = 200
    percent_train = 0.9
    learning_rate = 0.05
    num_train_samples = round(len(data) * percent_train)
    num_val_samples = len(data) - num_train_samples

    # Do train, validation split
    train, val = torch.utils.data.random_split(data, [num_train_samples, num_val_samples])

    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Specify model
    model = CNN(5)
    model.to(device)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # track losses and accuracies for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss_this_epoch = 0
        train_correct_this_epoch = 0
        val_loss_this_epoch = 0
        val_correct_this_epoch = 0

        # Training
        for train_samples, train_labels in train_loader:
            train_samples, train_labels = train_samples.to(device), train_labels.to(device)

            optimizer.zero_grad()

            out = model(train_samples)

            loss = criterion(out, train_labels)

            loss.backward()

            optimizer.step()

            train_loss_this_epoch += loss.cpu().detach().numpy()
            train_correct_this_epoch += calc_num_correct(out, train_labels)

        # Validation
        with torch.no_grad():
            for val_samples, val_labels in val_loader:
                val_samples, val_labels = val_samples.to(device), val_labels.to(device)

                # Compute output for validation examples
                out = model(val_samples)

                # Compute loss
                loss = criterion(out, val_labels)

                val_loss_this_epoch += loss.cpu().detach().numpy()
                val_correct_this_epoch += calc_num_correct(out, val_labels)

        print("Epoch " + str(epoch) + ":")
        print("Train loss: " + str(train_loss_this_epoch / num_train_samples))
        print("Valid loss: " + str(val_loss_this_epoch / num_val_samples))
        print("Train acc: " + str(train_correct_this_epoch / num_train_samples * 100) + "%")
        print("Valid acc: " + str(val_correct_this_epoch / num_val_samples * 100) + "%")

        train_losses.append(train_loss_this_epoch / num_train_samples)
        train_accuracies.append(train_correct_this_epoch / num_train_samples)
        val_losses.append(val_loss_this_epoch / num_val_samples)
        val_accuracies.append(val_correct_this_epoch / num_val_samples)

    plot_results(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs)

def calc_num_correct(pred, labels):
    pred, labels = pred.cpu(), labels.cpu()
    pred_argmax = torch.argmax(pred, dim=1)
    return torch.sum(torch.eq(pred_argmax, labels)).item()

def plot_results(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs):
    # Plotting
    fig, (loss_plot, accuracy_plot) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.5)

    # Loss plot
    loss_plot.plot(range(1, num_epochs + 1), train_losses)
    loss_plot.plot(range(1, num_epochs + 1), val_losses)
    max_loss = max(max(val_losses), max(train_losses))
    min_loss = min(min(val_losses), min(train_losses))
    loss_plot.axis([1, num_epochs, 0, max_loss + min_loss])
    loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_plot.set_ylabel("Loss")
    loss_plot.set_xlabel("Number of epochs")
    loss_plot.set_title("Loss")
    loss_plot.legend(["Training", "Validation"])

    # Accuracy plot
    accuracy_plot.plot(range(1, num_epochs + 1), train_accuracies)
    accuracy_plot.plot(range(1, num_epochs + 1), val_accuracies)
    accuracy_plot.axis([1, num_epochs, 0, 1 + 0.05])
    accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    accuracy_plot.set_ylabel("Accuracy (correct/incorrect)")
    accuracy_plot.set_xlabel("Number of epochs")
    accuracy_plot.set_title("Accuracy")
    accuracy_plot.legend(["Training", "Validation"])

    plt.show()

if __name__ == "__main__":
    train_model(".\\data", ".")
