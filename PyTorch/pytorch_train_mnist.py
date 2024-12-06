import torch                                    # torch: The core PyTorch library
import torch.nn as nn                           # torch.nn: A module to create and work wiht neural networks
import torch.optim as optim                     # torch.optim: Provides optimization algorithm for training
import torchvision                              # torchvision: Library with utilities for image processing, including datasets and transforms
import torchvision.transforms as transforms     # transforms: Utilites for data transformationa

# Defines a neural network class
class SimpleNet(nn.Module):
    def __init__(self):                 # Initializes the network layers. Here, we have two fully connected (linear) layers
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 28x28 input size flattened to 784. First layer, taking 784 input features and outputting 128 features
        self.fc2 = nn.Linear(128, 10)   # Output size 10 for classification. Second layer, taking 128 input features and outputting 10 features (for classification into 10 classes)

    def forward(self, x):               # Defines the forward pass through the network
        x = x.view(-1, 784)             # Flatten the input
        x = torch.relu(self.fc1(x))     # Applies ReLU activation to the output of the first layer
        x = self.fc2(x)                 # Passes the result to the second layeraa
        return x

# Create an instance of the network
net = SimpleNet()
print(net)


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])                                               # Converts images to PyTorch tensors
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # Loads the MNIST dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)                      # Creates iterators for the datasets to be used in the training loop

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Loads the MNIST dataset
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)                       # Creates iterators for the datasets to be used in the training loop

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()                               # Defines the loss function (cross-entropy loss)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # Defines the optimizer (Stochastic Gradient Descent with learning rate 0.01 and momentum 0.9)

# Training loop
for epoch in range(2):                                                              # Runs the training for 2 epochs 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):                                       # Iterates over batches of training data
        inputs, labels = data
        optimizer.zero_grad()                                                       # Zero the parameter gradients
        outputs = net(inputs)                                                       # Performs a forward pass
        loss = criterion(outputs, labels)                                           # Computes the loss
        loss.backward()                                                             # Computes the gradients
        optimizer.step()                                                            # Updates the model parameters

        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))  # Logs the loss every 200 batches
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():                                  # Disables gradient calculation during evaluation
    for data in testloader:                            # Iterates over batches of test data
        inputs, labels = data
        outputs = net(inputs)                          # Performs a forward pass
        _, predicted = torch.max(outputs.data, 1)      # Gets the predicted class
        total += labels.size(0)                        # Counts total number of test samples
        correct += (predicted == labels).sum().item()  # Counts correctly predicted samples

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))  # Prints the accuracy of the network