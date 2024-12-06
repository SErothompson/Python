import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
import torch.nn as nn
import torchvision.models as models
import av  # Import PyAV

# Set the device to CPU
device = torch.device('cpu')

# Define a simple transformation
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Reduce resolution to save memory
    transforms.ToTensor(),
])

# Load a sample video
video_path = 'movie.mov'
reader = av.open(video_path)

# Process video frames incrementally
video_frames = []
for frame in reader.decode(video=0):
    frame_tensor = transform(frame.to_image())
    video_frames.append(frame_tensor)

video_frames = torch.stack(video_frames).to(device)  # Move tensor to CPU
print(video_frames.shape)  # Check the shape of the processed video frames

# Define the neural network
class VideoProcessingNet(nn.Module):
    def __init__(self):
        super(VideoProcessingNet, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the classification layer
        self.fc = nn.Linear(2048, 3)  # Assuming 3 action classes

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

net = VideoProcessingNet().to(device)  # Move the model to CPU
print(net)

# Example labels (adjust according to your data)
labels = torch.tensor([0, 1, 2] * (len(video_frames) // 3 + 1))[:len(video_frames)].to(device)  # Move labels to CPU

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(2):
    running_loss = 0.0
    for i in range(len(video_frames)):
        inputs = video_frames[i].unsqueeze(0)
        target = labels[i]

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, target.unsqueeze(0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(net.state_dict(), 'trained_model.pth')

# Test on a new video
# Load the trained model
net = VideoProcessingNet()
net.load_state_dict(torch.load('trained_model.pth'))
net.to(device)
net.eval()  # Set the model to evaluation mode

# Load the test video
test_video_path = 'test_video.mov'
reader = av.open(test_video_path)

# Process test video frames incrementally
with torch.no_grad():
    frame_count = 0
    for frame in reader.decode(video=0):
        frame_tensor = transform(frame.to_image())
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        inputs = frame_tensor
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        print(f'Frame {frame_count}: Predicted action {predicted.item()}')
        frame_count += 1

print('Evaluation Complete')