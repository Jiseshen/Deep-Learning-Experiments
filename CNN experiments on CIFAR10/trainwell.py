from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchtoolbox.transform import Cutout
from torchtoolbox.tools import mixup_data, mixup_criterion
from torchtoolbox.nn import LabelSmoothingLoss
from torchsummary import summary
from ranger21 import Ranger21
from dense import DenseNet
from res import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

totensor = transforms.ToTensor()
train_transform = nn.Sequential(
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    # transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(20, interpolation=Image.BILINEAR),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)
)
test_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
trainset = torchvision.datasets.CIFAR10(root=".", train=True, download=True,
                                        transform=transforms.Compose([Cutout(), totensor]))
trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
testset = torchvision.datasets.CIFAR10(root=".", train=False, download=True, transform=totensor)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


def evaluate(model):
    with torch.no_grad():
        accuracy = 0
        n = 0
        for image, label in testloader:
            image = image.to(device)
            image = test_transform(image)
            label = label.to(device)
            logit = model(image)
            pred = logit.argmax(dim=1)
            accuracy += torch.eq(pred, label).float().sum().item()
            n += image.size(0)
        accuracy /= n
        return accuracy


def ranger(model, epoch, criterion, mixup):
    print("Using Ranger")
    converge_hist = []
    step = 0
    optimizer = Ranger21(model.parameters(), 1e-3, num_epochs=epoch, num_batches_per_epoch=len(trainloader))
    for i in tqdm(range(epoch)):
        model.train()
        for j, (image, label) in enumerate(trainloader):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            image = train_transform(image)
            if mixup:
                image, label1, label2, lam = mixup_data(image, label)
            optimizer.zero_grad()
            logit = model(image)
            loss = mixup_criterion(criterion, logit, label1, label2, lam) if mixup else criterion(logit, label)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            step += 1
        model.eval()
        accuracy = evaluate(model)
        print("Epoch{}, Train Loss:{}, Test Acc:{}".format(i, train_loss, accuracy))
    return accuracy, converge_hist


def phaseAdam(model, epoch, criterion, mixup):
    if epoch == 0:
        return 0, []
    print("Adam Phase")
    converge_hist = []
    step = 0
    optimizer = optim.Adam(model.parameters(), 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2, 1e-4)
    for i in tqdm(range(epoch)):
        model.train()
        for j, (image, label) in enumerate(trainloader):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            image = train_transform(image)
            if mixup:
                image, label1, label2, lam = mixup_data(image, label)
            optimizer.zero_grad()
            logit = model(image)
            loss = mixup_criterion(criterion, logit, label1, label2, lam) if mixup else criterion(logit, label)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(i + j / len(trainloader))
            step += 1
        model.eval()
        accuracy = evaluate(model)
        print("Epoch{}, Train Loss:{}, Test Acc:{}".format(i, train_loss, accuracy))
    return accuracy, converge_hist


def phaseSGD(model, epoch, criterion, mixup):
    if epoch == 0:
        model.eval()
        accuracy = evaluate(model)
        return accuracy, []
    print("SGD Phase")
    converge_hist = []
    step = 0
    optimizer = optim.SGD(model.parameters(), 1e-2, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, 1e-3)
    for i in range(epoch):
        model.train()
        for image, label in trainloader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            image = train_transform(image)
            if mixup:
                image, label1, label2, lam = mixup_data(image, label)
            optimizer.zero_grad()
            logit = model(image)
            loss = mixup_criterion(criterion, logit, label1, label2, lam) if mixup else criterion(logit, label)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            step += 1
        scheduler.step()
        model.eval()
        accuracy = evaluate(model)
        print("Epoch{}, Train Loss:{}, Test Acc:{}".format(i, train_loss, accuracy))
    return accuracy, converge_hist


def train(model, epoch1, epoch2, criterion, mixup):
    _, c1 = phaseAdam(model, epoch1, criterion, mixup)
    acc, c2 = phaseSGD(model, epoch2, criterion, mixup)
    return acc, c1 + c2


"""
12 Layers
"""

torch.manual_seed(30)
net = ResNet12(nn.LeakyReLU).to(device)
acc, loss = ranger(net, 80, LabelSmoothingLoss(10, 0.1), mixup=False)
torch.save(net, "model.pth")
print(acc)

"""
18 Layers
"""

# torch.manual_seed(30)
# net = ResNet18(nn.Mish).to(device)
# acc, loss = train(net, 75, 75, LabelSmoothingLoss(10, 0.1), mixup=False)
# torch.save(net, "big_model.pth")
# print(acc)

"""
Try Dense
"""

# torch.manual_seed(30)
# net = DenseNet().to(device)
# acc, loss = train(net, 75, 75, nn.CrossEntropyLoss(), mixup=False)
# torch.save(net, "dense_model.pth")
# print(acc)

