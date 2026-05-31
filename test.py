import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple Multi-Layer Perceptron (MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 2
    batch_size = 4
    learning_rate = 0.01
    epochs = 100

    # Model, Loss Function, and Optimizer
    model = SimpleMLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Dummy data: random inputs and targets
    X = torch.randn(batch_size, input_size)
    y = torch.randn(batch_size, output_size)

    print("Starting training...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("\nTraining complete.")

    # Test the model with one sample
    test_input = torch.randn(1, input_size)
    prediction = model(test_input)
    print(f"Test Input: {test_input}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
