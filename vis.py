import matplotlib.pyplot as plt


class TrainingVisualizer:
    def __init__(self):
        self.losses = []
        self.plt = plt.pyplot if hasattr(plt, "pyplot") else plt  # fallback
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
        plt.pause(0.01)  # Small pause to allow the UI to refresh

    def show(self):
        plt.show()
