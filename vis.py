import matplotlib.pyplot as plt
import torch


class TrainingVisualizer:
    def __init__(self):
        self.losses = []
        # Use a non-interactive backend if needed, but for live plots we want interactive
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], label="Training Loss")
        self.ax.set_xlabel("Batch Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Live Training Progress")
        self.ax.legend()

    def update(self, batch_idx, loss):
        self.losses.append(loss)
        self.line.set_data(range(len(self.losses)), self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def show(self):
        plt.show()


class FeatureSpy:
    def __init__(self, model):
        self.model = model
        self.features = {}

    def hook_fn(self, name):
        def hook(module, input, output):
            # We store the output of the layer
            self.features[name] = output.detach()

        return hook

    def register_hooks(self):
        # Register hooks on all convolutional layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(self.hook_fn(name))

    def visualize_layers(self, image, target_digit):
        self.model.eval()
        with torch.no_grad():
            # Pass the image through the model
            output = self.model(
                image.unsqueeze(0).to(next(self.model.parameters()).device)
            )
            prediction = output.argmax(dim=1).item()

        # Prepare Plotting
        num_layers = len(self.features)
        if num_layers == 0:
            print("No convolutional layers found to spy on!")
            return

        fig, axes = plt.subplots(1, num_layers + 1, figsize=(20, 5))

        # Show Original Image
        axes[0].imshow(image.squeeze(), cmap="gray")
        axes[0].set_title(f"Input: {target_digit}")
        axes[0].axis("off")

        # Show Feature Maps for each layer
        for i, (name, feature_map) in enumerate(self.features.items()):
            # A feature map has shape [1, channels, H, W]
            # We show the average of all channels to see "where" the layer is looking
            avg_map = torch.mean(feature_map, dim=1).squeeze()

            axes[i + 1].imshow(
                avg_map.cpu(), cmap="magma"
            )  # 'magma' makes patterns pop
            axes[i + 1].set_title(f"Layer: {name}")
            axes[i + 1].axis("off")

        plt.suptitle(f"Model Interpretation (Prediction: {prediction})", fontsize=16)
        plt.tight_layout()
        plt.show()


class FeatureSpy:
    def __init__(self, model):
        self.model = model
        self.features = {}

    def hook_fn(self, name):
        def hook(module, input, output):
            # We store the output of the layer
            self.features[name] = output.detach()

        return hook

    def register_hooks(self):
        # Register hooks on all convolutional layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(self.hook_fn(name))

    def visualize_layers(self, image, target_digit):
        self.model.eval()
        with torch.no_grad():
            # Pass the image through the model
            output = self.model(image.unsqueeze(0))
            prediction = output.argmax(dim=1).item()

        # Prepare Plotting
        num_layers = len(self.features)
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(20, 5))

        # Show Original Image
        axes[0].imshow(image.squeeze(), cmap="gray")
        axes[0].set_title(f"Input: {target_digit}")
        axes[0].axis("off")

        # Show Feature Maps for each layer
        for i, (name, feature_map) in enumerate(self.features.items()):
            # A feature map has shape [1, channels, H, W]
            # We show the average of all channels to see "where" the layer is looking
            avg_map = torch.mean(feature_map, dim=1).squeeze()

            axes[i + 1].imshow(avg_map, cmap="magma")  # 'magma' makes patterns pop
            axes[i + 1].set_title(f"Layer: {name}")
            axes[i + 1].axis("off")

        plt.suptitle(f"Model Interpretation (Prediction: {prediction})", fontsize=16)
        plt.tight_layout()
        plt.show()


def run_inference():
    # 1. Load Model and Data
    model = CNN()
    # Note: In a real scenario, you'd load weights here: model.load_state_dict(torch.load('weights.pth'))

    _, test_loader = get_mnist_loaders()
    spy = FeatureSpy(model)
    spy.register_hooks()

    # 2. Grab one image from test set
    data, target = next(iter(test_loader))
    img = data[0]
    label = target[0].item()

    print("Performing X-Ray scan on image...")
    spy.visualize_layers(img, label)


if __name__ == "__main__":
    run_inference()
