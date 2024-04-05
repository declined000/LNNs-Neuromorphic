#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cnn2snn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# In[2]:


class NCP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NCP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Regularization with dropout
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32, hidden_size=128):
        super(LNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization

        self.img_size = img_size
        self.ncp_input_size = self._calc_ncp_input_size()
        self.ncp_model = NCP(input_size=self.ncp_input_size, hidden_size=hidden_size, output_size=num_classes)

    def _calc_ncp_input_size(self):
        # Placeholder for dynamic size calculation based on the convolutional layers
        # Implement the correct size calculation
        return 64 * (self.img_size // 4) * (self.img_size // 4)  # Example calculation

    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.ncp_model(x)
        return x


# In[3]:


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


# In[4]:


# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LNN(in_channels=in_channels, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)
#input("Press Enter to continue...")


# In[5]:


# Training Loop
print(f'Starting Training...')

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


# In[6]:


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


# In[7]:


print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

training_time = time.time() - start_time
print(f"Training time: {training_time}")


# In[8]:


sample, _ = next(iter(trainloader))
torch.onnx.export(model,
                  sample,
                  f="mnist_cnn.onnx",
                  input_names=["inputs"],
                  output_names=["outputs"],
                  dynamic_axes={'inputs': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}})



# In[9]:


import onnx
from quantizeml.models import quantize, QuantizationParams

qparams = QuantizationParams(weight_bits=8, activation_bits=8, per_tensor_activations=True)

# Read the exported ONNX model
model_onnx = onnx.load_model("mnist_cnn.onnx")

# Quantize
model_quantized = quantize(model_onnx, qparams=qparams, num_samples=200)
print(onnx.helper.printable_graph(model_quantized.graph))


#print(model_quantized)


# In[10]:


from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()


# In[11]:


images, labels = data[0].to(device), data[1].to(device)

# Add a batch dimension if not already present and reorder dimensions
if len(images.shape) == 3:  # Assuming (C, H, W) format for a single image
    images = images.unsqueeze(0).permute(0, 2, 3, 1)  # Changes to (N, H, W, C)
elif len(images.shape) == 4:  # Assuming (N, C, H, W) format for a batch of images
    images = images.permute(0, 2, 3, 1)  # Changes to (N, H, W, C)

# Ensure the images have the correct shape (N, 32, 32, 3)
# You might need to adjust this part based on your specific dataset
start_time = time.time()
accuracy = model_akida.evaluate(images, labels)
eval_time = time.time() - start_time
print(f"Eval time: {eval_time}")
print('Test accuracy after conversion:', accuracy)


# In[ ]:





# In[ ]:




