import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import loss
from torch.optim import optimizer


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_dim = 10
hidden_dim = 20
output_dim = 2
learning_rate = 0.01
epochs = 100

x = torch.randn(64, input_dim)
y = torch.randint(0, output_dim, (64,))

model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("start training")
for epoch in range(epochs):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("training finished")

with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
