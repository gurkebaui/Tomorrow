import torch
import torch.nn as nn
import torch.optim as optim
from torch._inductor.config import optimize_scatter_upon_const_tensor

import vis as vis
from cnn import CNN
from dataloader import get_mnist_loaders


def train():
    # 1. Setup Device (Use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 2. Initialize Model, Loss, and Optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Muon(model.parameters(), lr=0.001)

    # 3. Get Data
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    # Initialize Visualizer once for the whole training session
    visualizer = vis.TrainingVisualizer()

    # 4. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)

            # --- THE CORE STEPS ---
            optimizer.zero_grad()  # 1. Clear old gradients
            output = model(data)  # 2. Forward pass (predict)
            loss = criterion(output, target)  # 3. Calculate loss (error)
            loss.backward()  # 4. Backward pass (calculate gradients)
            optimizer.step()  # 5. Update weights!
            # ----------------------

            running_loss += loss.item()
            if batch_idx % 100 == 99:  # Print every 100 batches
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {running_loss / 100:.4f}"
                )
                visualizer.update(batch_idx, running_loss / 100)
                running_loss = 0.0

        # 5. Validation (Test the model after each epoch)
        model.eval()  # Set model to evaluation mode
        correct = 0
        with torch.no_grad():  # Turn off gradient calculation for speed/memory
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # Get the index of the max value
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(
            f"\nEpoch {epoch + 1} Summary: Accuracy: {100.0 * correct / len(test_loader.dataset):.2f}%\n"
        )

        # --- X-RAY VISUALIZATION STEP ---
        print("Generating layer visualization for a test sample...")
        spy = vis.FeatureSpy(model)
        spy.register_hooks()

        # Grab one sample from the test set to see what the model "sees"
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            spy.visualize_layers(data[0], target[0].item())
            break  # Just do one image per epoch

        import matplotlib.pyplot as plt

        plt.close("all")
        # --------------------------------


if __name__ == "__main__":
    train()
