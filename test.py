import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)
    np.save('fpr.npy', fpr)
    np.save('tpr.npy', tpr)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("AUC.png")

from config import Opt

opt = Opt()
opt.isTrain = False
from mildataset import create_dataset
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch.autograd import Variable
from model import ClsNet
import numpy as np

val_dataset = create_dataset(opt)
val_dataset_size = len(val_dataset)
print('The number of training images = %d' % val_dataset_size)

cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)
    print('\nGPU is ON!')
model = ClsNet(opt)
model_dict = model.state_dict()
pretrained_dict = torch.load('checkpoint/mil_100.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

if cuda:
    model.cuda()

model.eval()
all_target = []
all_pre_list = []
all_prob_list = []
all_ids = []

with torch.no_grad():
    for i, data_ in enumerate(val_dataset):
        bag_label = data_['label1']
        bag_label = data_['label1']
        img1 = data_['img1']
        img1, bag_label = img1.cuda(), bag_label.cuda()
        img1, bag_label = Variable(img1), Variable(bag_label)
        allimg_pre, allimg_prob = model.calculate_acc(img1)
        all_target.append(bag_label.cpu().data.numpy())
        all_ids.append(data_['id_'][0])
        all_pre_list.append(allimg_pre)
        all_prob_list.append(allimg_prob)

acc = accuracy_score(all_target, all_pre_list)
AUC = roc_auc_score(all_target, all_prob_list)
precision = precision_score(all_target, all_pre_list)
recall = recall_score(all_target, all_pre_list)
f1score = f1_score(all_target, all_pre_list)
print('acc: {:.4f}, auc: {:.4f}, prc: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(acc, AUC, precision, recall, f1score))
acu_curve(all_target, all_prob_list)

with open("test_results.txt", "w") as file:
    counter_1 = 0
    counter_2 = 0
    correct_predictions = 0
    file.write("Image ID, Prediction, Probability, Target\n")
    for img_id, pred, prob, target in zip(all_ids, all_pre_list, all_prob_list, all_target):
        file.write(f"{img_id}, {pred}, {prob}, {target}\n")
        if pred == target:
            correct_predictions += 1
        else:
            if pred == [0] and target == [1]:
                counter_1 += 1
            elif pred == [1] and target == [0]:                
                counter_2 += 1
    file.write(f"Correct predictions: {correct_predictions}/{len(all_ids)}")
    file.write(f"counter_1: {counter_1}, counter_2: {counter_2}")
    

with open("test_filtered_results.txt", "w") as file:
    correct_predictions = 0
    number_of_predictions = 0
    file.write("Image ID, Prediction, Probability\n")
    for img_id, pred, prob, target in zip(all_ids, all_pre_list, all_prob_list, all_target):
        if img_id.startswith("20051020"):
            number_of_predictions += 1
            file.write(f"{img_id}, {pred}, {prob}, {target}\n")
            if pred == target:
               correct_predictions += 1
    file.write(f"Correct predictions: {correct_predictions}/{number_of_predictions}")
    