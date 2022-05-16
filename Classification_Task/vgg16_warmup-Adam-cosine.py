
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
from model_vgg import *
from warmup_scheduler import GradualWarmupScheduler
import random
import numpy as np
import datetime
from sklearn.metrics import classification_report, confusion_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
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
# #
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
fh = logging.FileHandler(os.path.join(log_path, 'train-{}{:02}{}-vgg16warmup-Adam-cosine'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
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
# #
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

# net = GoogLeNet(num_classes=3, aux_logits=True, init_weights=True)
net = vgg16_bn(num_classes=3)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)
# optimizer = optim.RMSprop(net.parameters(), lr=0.0003, alpha=0.9)
# optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                     lambda epoch : (1.0-epoch/200) if epoch <= 200 else 0, last_epoch=-1)

# scheduler_warmup is chained with schduler_steplr
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=0, last_epoch=-1)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=cosine_scheduler)

best_acc = 0.0

for epoch in range(1, 301):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data

        optimizer.zero_grad()
        logits= net(images.to(device))
        loss = loss_function(logits, labels.to(device))
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
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format((epoch-1)*len(train_loader) + step, scheduler_warmup.get_lr()[0],loss.item())
        logging.info(printInfo)
    scheduler_warmup.step(epoch)
    # scheduler.step()
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    # add by junjie to compute precision/recall/f1-scores
    y_pred_list = []
    y_true_list = []
    y_pred_lists = []
    y_true_lists = []
    #
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

            y_pred_list.append(predict_y.cpu().numpy())
            y_true_list.append(val_labels.cpu().numpy())

        val_accurate = acc / val_num
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        state = {'state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'lr_scheduler_state_dict':cosine_scheduler.state_dict()}
        filename = os.path.join(os.path.join(compare_result, "vgg-{:06}-WarmUp-Adam-cosine.pth.tar".format(epoch+1)))
        torch.save(state, filename)
        # print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
        #       (epoch + 1, running_loss / step, val_accurate))
        logInfo = '[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %(epoch + 1, running_loss / step, val_accurate)
        logging.info(logInfo)

        # add by junjie to compute precision/recall/f1-scores
        for pre, tru in zip(y_pred_list, y_true_list):
            y_pred_lists += list(pre)
            y_true_lists += list(tru)
        # y_pred_list += [list(i[:]) for i in y_pred_list]
        # y_true_list += [list(i[:]) for i in y_true_list]
        target_name = list(class_indict.values())
        indicators = classification_report(y_true_lists, y_pred_lists, target_names=target_name)
        matrix_map = confusion_matrix(y_true_lists, y_pred_lists)

        with open(
                '/media/jie/DATA/Junjie/PhD_project/Classification-Task/ShuffleNet-Series-master/DetNAS_models/VGG16/log-Adam-cosine.txt',
                'a+') as fd:
            fd.write("# AP and precision/recall/f1 per class\n" + "Epoch: %06d\n" % (epoch))
            fd.write(indicators + '\n')

print('Finished Training')