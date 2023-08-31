import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as T
from models import googlenet
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('check your GPUs')
    exit()
print('Device: ', device)

# hyper parameter
batch_size = 100

# using FP32 format
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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

correct = 0
total = 0
model.eval()
start_run_model = time.time()
with torch.no_grad():
    for iteraction, (images, labels) in enumerate(test_loader, 0):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end_run_model = time.time()
run_model_accuracy = (100 * correct / total)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

model.train()
training_start = time.time()
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    loss = criterion(output, labels)
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
accuracy = (100 * correct / total)

print('run model time: ', end_run_model - start_run_model)
print('training time: ', training_end - training_start)
print('inference time: ', inference_end - inference_start)
print('pre-training accuracy: ', run_model_accuracy)
print('re-training accuracy: ', accuracy)