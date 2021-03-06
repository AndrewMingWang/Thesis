from datasets import *
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator

def train_model():
    # Save run data to dir
    output_dir = "./output/FaceBased/" +"MNIST" #+"MNIST" +"MNIST_01" +"MNIST_27"

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    binary = False
    label0 = 2
    label1 = 7
    thingi10k = False
    data, model = None, None

    # Create dataset
    if thingi10k:
        data = Thingi10kProcessedMeshDataset("./data/Thingi10k/padded_data",
                                             "./data/Thingi10k/padded_data",
                                             "./data/Thingi10k/output.pkl")
    else:
        if binary:
            data = MNISTBinaryProcessedMeshDataset("./data/MNIST/padded_data",
                                                   "./data/MNIST/padded_data",
                                                   "./data/MNIST/label.txt",
                                                   label0=label0 , label1=label1)
        else:
            data = MNISTProcessedMeshDataset("./data/MNIST/padded_data",
                                             "./data/MNIST/padded_data",
                                             "./data/MNIST/label.txt")

    # Specifiy model
    num_faces = 14000 if thingi10k else 9000
    num_classes = 2 if binary else 10
    model = CNN(num_kernels=5, num_classes=num_classes, device=device, num_faces=num_faces)
    model.to(device)

    print("Dataset created, number of points: " + str(len(data)))
    print("Number of adjacencies loaded: " + str(len(data.adjacency_dict)))
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

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # track losses and accuracies for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # track correct and incorrect values
    correct = -np.ones(num_val_samples)
    incorrect = -np.ones(num_val_samples)

    for epoch in range(num_epochs):
        train_loss_this_epoch = 0
        train_correct_this_epoch = 0
        val_loss_this_epoch = 0
        val_correct_this_epoch = 0

        # Training
        for train_samples, train_labels in train_loader:
            labels = train_labels[0]
            features = train_samples[0]
            adjacencies = train_samples[1]

            labels, features, adjacencies = \
                labels.to(device), features.to(device), adjacencies.to(device)

            optimizer.zero_grad()

            out = model(features, adjacencies)

            loss = criterion(out, labels)

            loss.backward()

            optimizer.step()

            train_loss_this_epoch += loss.cpu().detach().numpy()
            train_correct_this_epoch += calc_num_correct(out, labels)

        # Validation
        with torch.no_grad():
            batch = 0
            for val_samples, val_labels in val_loader:
                labels = val_labels[0]
                indices = val_labels[1].numpy()
                features = val_samples[0]
                adjacencies = val_samples[1]

                labels, features, adjacencies = \
                    labels.to(device), features.to(device), adjacencies.to(device)

                # Compute output for validation examples
                out = model(features, adjacencies)

                # Compute loss
                loss = criterion(out, labels)
                val_loss_this_epoch += loss.cpu().detach().numpy()

                # Compute accuracy
                correct_this_batch = calc_num_correct(out, labels)
                val_correct_this_epoch += correct_this_batch

                # On the last epoch, get indices of correct and incorrect predictions
                if epoch == num_epochs-1:
                    correct_this_batch, correct_indices, incorrect_indices = calc_num_correct_with_indices(out, labels)
                    correct_indices = indices[correct_indices]
                    incorrect_indices = indices[incorrect_indices]
                    correct[batch*batch_size:batch*batch_size + len(correct_indices)] = correct_indices
                    incorrect[batch*batch_size:batch*batch_size + len(incorrect_indices)] = incorrect_indices
                    batch += 1

        print("Epoch " + str(epoch) + ":")
        print("Train loss: " + str(train_loss_this_epoch / num_train_samples))
        print("Valid loss: " + str(val_loss_this_epoch / num_val_samples))
        print("Train acc: " + str(train_correct_this_epoch / num_train_samples * 100) + "%")
        print("Valid acc: " + str(val_correct_this_epoch / num_val_samples * 100) + "%")

        train_losses.append(train_loss_this_epoch / num_train_samples)
        train_accuracies.append(train_correct_this_epoch / num_train_samples)
        val_losses.append(val_loss_this_epoch / num_val_samples)
        val_accuracies.append(val_correct_this_epoch / num_val_samples)

    print("Max Train acc: " + str(max(train_accuracies)))
    print("at: " + str(np.argmax(train_accuracies)))
    print("Max Validation acc: " + str(max(val_accuracies)))
    print("at: " + str(np.argmax(val_accuracies)))

    # Save results to output dir
    with open(output_dir + "/correct.pkl", 'wb') as file:
        pickle.dump(correct, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir + "/incorrect.pkl", 'wb') as file:
        pickle.dump(incorrect, file, protocol=pickle.HIGHEST_PROTOCOL)
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
    return torch.sum((torch.eq(pred_argmax, labels))).item()

def calc_num_correct_with_indices(pred, labels):
    pred, labels = pred.cpu(), labels.cpu()
    pred_argmax = torch.argmax(pred, dim=1)
    correct_array = torch.eq(pred_argmax, labels).cpu().numpy()
    correct_indices = np.where(correct_array == 1)
    incorrect_indices = np.where(correct_array == 0)

    return np.sum(correct_array), correct_indices, incorrect_indices

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
