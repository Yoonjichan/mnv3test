import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from mnv3 import mobilenetv3
##
import matplotlib.pyplot as plt
##
def evaluate_acc(net, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

##
if __name__ =='__main__':
    # Define transformations for data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    net = mobilenetv3(n_class=10, input_size=32, mode='small').to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.8, weight_decay=5e-4)
    
#[1,   200] loss: 2.031
#epoch 1: Train Accuracy 39.79%, Test Accuracy  41.42%
#[2,   200] loss: 1.578
#epoch 2: Train Accuracy 46.23%, Test Accuracy  47.45%    
    
    
##
    train_acc_list=[]
    test_acc_list=[]
##
    # Train the network
    for epoch in range(100):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # Print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        ##
        train_acc=evaluate_acc(net, trainloader, device)
        test_acc=evaluate_acc(net, testloader, device)
        print('epoch %d: Train Accuracy %.2f%%, Test Accuracy % .2f%%' %(epoch+1, train_acc, test_acc))
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        ##
    print('Finished Training')

    # Evaluate the network on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 


    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
       ##
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()
    ##
