import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')






class NCP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NCP, self).__init__()
        # Example architecture
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Pass through first layer and apply activation
        x = self.relu(self.fc2(x))  # Pass through second layer and apply activation
        x = self.fc3(x)  # Pass through output layer
        return x

class LNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32, hidden_size=128):
        super(LNN, self).__init__()

        # Convolutional layers as before
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Calculate size for NCP input
        def calc_size(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        size = calc_size(calc_size(img_size))  # Apply calc_size twice
        ncp_input_size = 64 * size * size

        self.ncp_model = NCP(
            input_size=ncp_input_size,
            hidden_size=hidden_size,
            output_size=num_classes
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        out = self.ncp_model(x)
        return out



# CIFAR-10 Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# Ensure to instantiate model with correct parameters
in_channels = 3  # CIFAR-10 images have 3 channels
num_classes = 10

# Model Initialization
model = LNN(in_channels=in_channels, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
start_time = time.time()
for epoch in range(10):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}')

# Evaluation Loop
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')





training_time = time.time() - start_time
print(f"Training time: {training_time}")

# Evaluation
#model.eval()
#start_time = time.time()
#total_images_processed = 0
#all_predicted = []
#all_labels = []
#with torch.no_grad():
    #for data in testloader:
        #images, labels = data
        #outputs = model(images)
        #, predicted = torch.max(outputs.data, 1)
        #total_images_processed += labels.size(0)
        #all_predicted.extend(predicted.numpy())
        #all_labels.extend(labels.numpy())

#elapsed_time = time.time() - start_time
#precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predicted, average='macro')
#accuracy = accuracy_score(all_labels, all_predicted)




#print(f"Precision: {precision}")
#print(f"Recall: {recall}")
#print(f"F1-score: {f1}")
#print(f"Accuracy: {accuracy}")
#print(f"Inference time: {elapsed_time}")
#print(f"Latency: {elapsed_time / len(testloader.dataset)} images/s")

#throughput = total_images_processed / elapsed_time
#print(f"Throughput: {throughput} images/s")

#model_complexity = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f"Model complexity: {model_complexity} parameters")

