import torch.nn as nn
import torch.optim as optim

# Define the network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 28x28 input size flattened to 784
        self.fc2 = nn.Linear(128, 10)   # Output size 10 for classification

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the network
net = SimpleNet()
print(net)