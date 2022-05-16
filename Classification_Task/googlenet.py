
#train.py

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision
import json
import matplotlib.pyplot as plt
import os
import logging
import time
import sys
import torch.optim as optim
from model_googlenet import GoogLeNet
import random
import numpy as np
import datetime
from warmup_scheduler import GradualWarmupScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

image_path = '/media/jie/DATA/Junjie/PhD_project/RockData/classifier-data/'
save_path = '/media/jie/DATA/Junjie/PhD_project/Classification-Task/ShuffleNet-Series-master/DetNAS_models/'
job_name = '{}'.format(datetime.datetime.now().strftime('%b%d-%H'))
compare_result = os.path.join(save_path, 'models-{}'.format(job_name))
os.makedirs(compare_result, exist_ok=True)
train_dataset = datasets.ImageFolder(root=image_path + "train",
                                     transform=data_transform["train"])
# 将数据集整体打乱
random.shuffle(train_dataset.imgs)
train_dataset.targets = list(np.array(train_dataset.imgs)[:, 1].astype(np.int))
#
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# Log
log_format = '[%(asctime)s] %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%d %I:%M:%S')
t = time.time()
local_time = time.localtime(t)
log_path = './log-{}/'.format(job_name)
os.makedirs(log_path, exist_ok=True)
fh = logging.FileHandler(os.path.join(log_path, 'train-{}{:02}{}-googlenetwarmup-Adam'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
# 将数据集整体打乱
random.shuffle(validate_dataset.imgs)
validate_dataset.targets = list(np.array(validate_dataset.imgs)[:, 1].astype(np.int))
#
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = GoogLeNet(num_classes=3, aux_logits=True, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)
# optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                     lambda step : (1.0-step/50000) if step <= 50000 else 0, last_epoch=-1)

# scheduler_warmup is chained with schduler_steplr
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=0, last_epoch=-1)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=cosine_scheduler)

best_acc = 0.0

for epoch in range(300):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        # scheduler.step()
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        # print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        # printInfo = "\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss)
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch*len(train_loader) + step, cosine_scheduler.get_lr()[0],loss.item())
        logging.info(printInfo)
    scheduler_warmup.step()
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        state = {'state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'lr_scheduler_state_dict':cosine_scheduler.state_dict()}
        filename = os.path.join(os.path.join(compare_result, "googleNet-{:06}-warmup-Adam.pth.tar".format(epoch+1)))
        torch.save(state, filename)
        # print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
        #       (epoch + 1, running_loss / step, val_accurate))
        logInfo = '[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %(epoch + 1, running_loss / step, val_accurate)
        logging.info(logInfo)

print('Finished Training')