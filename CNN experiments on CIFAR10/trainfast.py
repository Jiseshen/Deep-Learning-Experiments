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
trainset = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transforms.Compose(totensor))  # No cutout in training ResNet9
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)  # Bigger batch-size
testset = torchvision.datasets.CIFAR10(root=".", train=False, download=True, transform=totensor)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)


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


def fastTrain(model, epoch=10, criterion=nn.CrossEntropyLoss()):
    optimizer = optim.Adam(model.parameters(), 1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, epochs=epoch, steps_per_epoch=len(trainloader))
    for i in tqdm(range(epoch)):
        model.train()
        for image, label in trainloader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            image = train_transform(image)  # In-training data augmentation
            optimizer.zero_grad()
            logit = model(image)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    accuracy = evaluate(model)
    return accuracy


torch.manual_seed(30)
net = ResNet9(nn.LeakyReLU).to(device)
acc = fastTrain(net, 12)
torch.save(net, "light_model.pth")
print(acc)
