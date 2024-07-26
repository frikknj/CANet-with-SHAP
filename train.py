from config import Opt
opt = Opt()
opt.isTrain = True
from mildataset import create_dataset
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.optim as optim
from torch.autograd import Variable
from model import ClsNet
import numpy as np
from torch.optim import lr_scheduler

def get_scheduler(optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1- 50) / float(50 + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


train_dataset = create_dataset(opt)
train_dataset_size = len(train_dataset)
print('The number of training images = %d' % train_dataset_size)
opt.isTrain = False
val_dataset = create_dataset(opt)
val_dataset_size = len(val_dataset)
print('The number of validation images = %d' % val_dataset_size)
opt.isTrain = True

cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)
    print('\nGPU is ON!')
model = ClsNet(opt)
if cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=10e-5)
lr_scheduler = get_scheduler(optimizer)
def train(epoch):
    model.train()
    train_loss = 0.
    train_e_loss = []
    for batch_idx, data_ in enumerate(train_dataset):
        bag_label = data_['label1']
        img1 = data_['img1']
        img1, bag_label = img1.cuda(), bag_label.cuda()
        img1, bag_label = Variable(img1), Variable(bag_label)
        optimizer.zero_grad()
        loss = model.calculate_loss(img1, bag_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().detach().numpy()
        train_e_loss.append(loss.cpu().detach().numpy())

    train_loss /= 67

    print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))
    return train_loss, train_e_loss

def val(epoch):
    model.eval()
    all_target = []
    all_pre_list = []
    all_prob_list = []
    with torch.no_grad():
        for i, data_ in enumerate(val_dataset):
            bag_label = data_['label1']
            img1 = data_['img1']
            img1, bag_label = img1.cuda(), bag_label.cuda()
            img1, bag_label = Variable(img1), Variable(bag_label)
            allimg_pre, allimg_prob = model.calculate_acc(img1)
            all_target.append(bag_label.cpu().data.numpy())
            all_pre_list.append(allimg_pre)
            all_prob_list.append(allimg_prob)
        acc = accuracy_score(all_target, all_pre_list)
        auc = roc_auc_score(all_target, all_prob_list)
        print('Epoch: {}, allimg_acc: {:.4f}, allimg_auc: {:.4f}'.format(epoch, acc, auc))

loss_list = []
i_loss_list = []


for epoch in range(1, 101):
    train_loss, train_i_loss = train(epoch)
    loss_list.append(train_loss)
    i_loss_list.append(train_i_loss)

    if epoch % 5 == 0:
        print('saving the model at the end of epoch %d' % (epoch))
        torch.save(model.cpu().state_dict(), 'checkpoint/mil_%s.pth' % epoch)
        model.cuda(0)

    if epoch % 5 == 0:
        val(epoch)
    lr_scheduler.step()
    lr = (lr_scheduler.get_lr()[0])
    print('learning rate = %.7f' % lr)
np.save('loss.npy', loss_list)
np.save('i_loss.npy', i_loss_list)