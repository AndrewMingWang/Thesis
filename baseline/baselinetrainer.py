from baselinedatasets import *
from baselinemodels import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def train_model():
    # Save run data to dir
    output_dir = "../output/Baseline/" + "MNIST"  # +"MNIST" +"MNIST_01" +"MNIST_27"

    #voxelized_data_folder = "../data/Thingi10k/baseline_data/voxelized"
    #labels_file = "../data/Thingi10k/output.pkl"
    voxelized_data_folder = "../data/MNIST/baseline_data/voxelized"
    labels_file = "../data/MNIST/label.txt"

    # Create dataset
    # data = BinaryVoxelDataset(voxelized_data_folder, labels_file, label0=0, label1=1)
    # data = BinaryVoxelDataset(voxelized_data_folder, labels_file, label0=2, label1=7)
    data = VoxelDataset(voxelized_data_folder, labels_file)

    print("Dataset created, number of points: " + str(len(data)))
    print("Class counts:")
    print(data.class_counts)

    # Specify hyperparameters
    num_epochs = 15
    batch_size = 64
    percent_train = 0.8
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
    model = BaselineCNN(num_kernels=5, num_classes=10)
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
            correct_this_batch,_,_ = calc_num_correct(out, train_labels)
            train_correct_this_epoch += correct_this_batch

        # Validation
        with torch.no_grad():
            for val_samples, val_labels in val_loader:
                val_samples, val_labels = val_samples.to(device), val_labels.to(device)

                # Compute output for validation examples
                out = model(val_samples)

                # Compute loss
                loss = criterion(out, val_labels)

                val_loss_this_epoch += loss.cpu().detach().numpy()
                correct_this_batch, correct_indices, incorrect_indices = calc_num_correct(out, val_labels)
                val_correct_this_epoch += correct_this_batch

        print("Epoch " + str(epoch) + ":")
        print("Train loss: " + str(train_loss_this_epoch / num_train_samples))
        print("Valid loss: " + str(val_loss_this_epoch / num_val_samples))
        print("Train acc: " + str(train_correct_this_epoch / num_train_samples * 100) + "%")
        print("Valid acc: " + str(val_correct_this_epoch / num_val_samples * 100) + "%")

        train_losses.append(train_loss_this_epoch / num_train_samples)
        train_accuracies.append(train_correct_this_epoch / num_train_samples)
        val_losses.append(val_loss_this_epoch / num_val_samples)
        val_accuracies.append(val_correct_this_epoch / num_val_samples)

    # Save results to output dir
    file = open(output_dir + "/res.txt", "w")
    file.write(str(train_accuracies[-1]) + "\n")
    file.write(str(max(train_accuracies)) + "\n")
    file.write(str(np.argmax(train_accuracies)) + "\n")
    file.write(str(val_accuracies[-1]) + "\n")
    file.write(str(max(val_accuracies)) + "\n")
    file.write(str(np.argmax(val_accuracies)) + "\n")
    file.close()

    plot_results(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs, output_dir)

def calc_num_correct(pred, labels):
    pred, labels = pred.cpu(), labels.cpu()
    pred_argmax = torch.argmax(pred, dim=1)
    correct_array = torch.eq(pred_argmax, labels)

    return torch.sum(correct_array).item(), 0, 0

def plot_results(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs, output_dir):
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

    plt.savefig(output_dir + "/lossaccgraphs.png", format="png")
    plt.show()

if __name__ == "__main__":
    train_model()
