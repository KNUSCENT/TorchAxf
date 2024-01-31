import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from models import googlenet
import time

# using FP32 format
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('check your GPUs')
    exit()
print('Device: ', device)

batch_size = 100

###################################################################################################################################
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616))
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./datasets/CIFAR10_data/', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./datasets/CIFAR10_data/', train=False, download=True, transform=transform_test)

evens = list(range(0, len(train_dataset), 10))
train_dataset = torch.utils.data.Subset(train_dataset, evens)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
###################################################################################################################################

model = googlenet.googlenet(pretrained=True, device=device)
model = model.to(device)

loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

total = 0
correct = 0

model.eval()
with torch.no_grad():
    for iteraction, (images, labels) in enumerate(test_loader, 0):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
pre_training_accuracy = (100 * correct / total)

model.train()
training_start = time.time()
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    loss = loss(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
training_end = time.time()

correct = 0
total = 0

model.eval()
inference_start = time.time()
with torch.no_grad():
    for iteraction, (images, labels) in enumerate(test_loader, 0):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
inference_end = time.time()
re_training_accuracy = (100 * correct / total)

print('training time: ', training_end - training_start)
print('inference time: ', inference_end - inference_start)
print('pre-training accuracy: ', pre_training_accuracy)
print('re-training accuracy: ', re_training_accuracy)