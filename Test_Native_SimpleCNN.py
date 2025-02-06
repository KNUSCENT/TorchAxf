import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import SimpleCNN
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    print('check your GPUs')
    exit()
#device = 'cpu'
print('Device: ', device)

# fixed random seed
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# using FP32 format
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# hyper parameter
learning_rate = 0.001
training_epochs = 1
batch_size = 100

# MNIST data load
mnist_train = datasets.MNIST(root='./datasets/MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./datasets/MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True)

model = SimpleCNN.Native_CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('Total Batch Size: {}'.format(total_batch))

# training
model.train()
train_start = time.time()
for epoch in range(training_epochs):
    avg_cost = 0
    for iteraction, (X, Y) in enumerate(data_loader, 0):
        X = X.to(device)
        Y = Y.to(device)
        #print("upload data time: ", time.time() - up_start)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
        #print("batch time: ", time.time() - start)
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
train_time = time.time() - train_start

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
test_time = time.time() - inference_start
accuracy = (100 * correct / total)

print("training time : ", train_time)
print("inference time: ", test_time)
print('Accuracy: ', accuracy)