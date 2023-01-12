import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import click
import matplotlib.pyplot as plt
from model import ConvNet


@click.group()
def cli():
    pass


# Datapath
main_path = "./data/processed"


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=5, help="learning rate to use for training")
def train(lr, epochs):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    # Pytorch train and test sets
    images = torch.unsqueeze(torch.load(f"{main_path}/train_images.pt"), dim=1)
    labels = torch.load(f"{main_path}/train_labels.pt")
    train = TensorDataset(images, labels)
    train_set = DataLoader(train, batch_size=8, shuffle=True)

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, torch.flatten(labels))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_set)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)

    torch.save(model.state_dict(), "./models/cnn_checkpoint.pth")

    plt.figure()
    plt.plot([i + 1 for i in range(epochs)], losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./reports/figures/convergence.png", dpi=200)


cli.add_command(train)

if __name__ == "__main__":
    cli()
