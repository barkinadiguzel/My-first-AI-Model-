import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# Data
bias = 0.3
weight = 0.78
X = torch.arange(0,1,0.02).unsqueeze(1)
Y = weight * X + bias

train_split = int(0.8 * len(X))
x_train, y_train = X[:train_split], Y[:train_split]
x_test, y_test = X[train_split:], Y[train_split:]

# Plot function
def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Train")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Pred")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias

torch.manual_seed(42)
model0 = LinearRegressionModel()

# Initial prediction
with torch.inference_mode():
    y_preds = model0(x_test)
plot_predictions(predictions=y_preds)

# Training
train_loss_values = []
test_loss_values = []
epoch_count = []
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model0.parameters(), lr=0.01)
epochs = 1000

for epoch in range(epochs):
    model0.train()
    y_pred = model0(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.inference_mode():
        y_preds = model0(x_test)
        test_loss = loss_fn(y_preds, y_test)
    
    epoch_count.append(epoch)
    train_loss_values.append(loss.item())
    test_loss_values.append(test_loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")

# Final evaluation
model0.eval()
plot_predictions(predictions=y_preds)

# Loss curves
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Test")
plt.legend()
plt.show()

# Save model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(model0.state_dict(), MODEL_SAVE_PATH)
print(f"Saved model to: {MODEL_SAVE_PATH}")
