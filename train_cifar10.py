"""

Train CIFAR10 with PyTorch
written by @patrick-pagni

"""

import os
import argparse
import csv
import time
import wandb

import torch
import torck.backends.cudnn as cudnn
from torch import nn, utils

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from models.cnn import AdvancedCNN
from randomaug import RandAugment
from utils import weights_init, progress_bar

torch.manual_seed(42)


parser = argparse.ArgumentParser(description="CIFAR10 Training pipeline")
parser.add_argument("--noaug", action="store_false", help="disable random augmentation")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--dp', action='store_true', help='use data parallel')

args = parser.parse_args()

watermark = f"advances_lr{args.lr}"
wandb.init(project="cifar10-challenge", name = watermark)
wandb.config.update(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0
start_epoch = 0

#Data
print("===> Preparing Data")
from torchvision.transforms import v2

"""
Load CIFAR 10 dataset.

Apply two random transformations from set of transformations defined above to
training data.

Do not apply transformations to test data.
"""

train_augmentation = transforms.Compose([
    RandAugment(2, 14),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_augmentation = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.noaug:
    N = 2; M=14;
    train_augmentation.transforms.insert(0, RandAugment(N, M))

train = CIFAR10(
    "./data",
    train = True,
    download = True,
    transform = train_augmentation
    )
train_dl = utils.data.DataLoader(
    train,
    32,
    shuffle = True,
    num_workers = 8
    )

test = CIFAR10(
    "./data",
    train = False,
    download = True,
    transform = test_augmentation
    )
test_dl = utils.data.DataLoader(
    test,
    100,
    num_workers = 8
    )

# Load model
block_config = {
    "block_1": {
        "input_channels": 256,
        "output_channels": 256,
        "output_volume": 32,
        "units": 4,
    },
    "block_2": {
        "input_channels": 256,
        "output_channels": 256,
        "output_volume": 26,
        "units": 4,
      },
    "block_3": {
        "input_channels": 256,
        "output_channels": 256,
        "output_volume": 20,
        "units": 4,
    },
    "block_4": {
        "input_channels": 256,
        "output_channels": 256,
        "output_volume": 20,
        "units": 4,
    },
    "block_5": {
        "input_channels": 256,
        "output_channels": 256,
        "output_volume": 18,
        "units": 4,
    }
}

net=AdvancedCNN(block_config).apply(weights_init)

if "cuda" in device:
   print(device)
   if args.dp:
      print("using data parallel")
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True

if args.resume:
    # Load checkpoint
    print("==> Resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load(".checkpoint/advanced.ckpt.t7")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

# Loss is Cross-Entropy
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

scaler = torch.cuda.amp.GradScaler(enabled = True)

# Training
def train(epoch):
    print(f"\nEpoch: {epoch}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = input.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enable=True):
          outputs = net(inputs)
          loss = criterion(outputs,targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scaler.zero_grad()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct += predicted.eq(targets).sum().item()

    progress_bar(batch_idx, len(train_dl), f"Loss: {train_loss/(batch_idx + 1)} Acc: {100*correct/total} ({correct}/{total})")
    return train_loss/(batch_idx+1)

# Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_dl), f"Loss: {test_loss/(batch_idx + 1)} Acc: {100*correct/total} ({correct}/{total})")
    # Save checkpoint
    acc = 100 * correct/total

    if acc > best_acc:
        print("Saving")
        state = {
           "model": net.state_dict(),
           "optimizer": optimizer.state_dict(),
           "scaler": scaler.state_dict()
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/advanced-ckpt.t7")
        best_acc = acc
    
    os.mkdirs("log", exist_ok =True)
    content = time.ctime() + " " + f"Epoch {epoch}, lr: {optimizer.para_groupd[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}"
    print(content)

    with open(f"log/log_advanced.txt", "a") as file:
       file.write(content + "\n")

    return test_loss, acc
    
list_loss = []
list_acc = []

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    scheduler.step(epoch-1)

    list_loss.append(val_loss)
    list_acc.append(acc)

    wandb.log(
        {
            "epoch": epoch,
            "train_loss": trainloss,
            "val_loss": val_loss,
            "val_acc": acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start
        }
    )

    with open(f"log/log-advanced.casv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print(list_loss)

    wandb.save("wandb-advanced.h5")
