import torch
from PIL import Image
import glob
import xlrd
import numpy as np
from torchvision import transforms
import os

def load_csv(path):
    dictLabels_DR = {}
    dictLabels_DME = {}
    for per_path in path:
        # open xlsx
        xl_workbook = xlrd.open_workbook(per_path)
        xl_sheet = xl_workbook.sheet_by_index(0)
        for rowx in range(1, xl_sheet.nrows):
            cols = xl_sheet.row_values(rowx)
            filename = cols[0]
            label1 = int(cols[2])
            label2 = int(cols[3])

            if label1 < 2:
                label1 = 0
            else:
                label1 = 1

            if label1 in dictLabels_DR.keys():
                dictLabels_DR[label1].append(filename)
            else:
                dictLabels_DR[label1] = [filename]

            if label2 in dictLabels_DME.keys():
                dictLabels_DME[label2].append(filename)
            else:
                dictLabels_DME[label2] = [filename]
    return dictLabels_DR, dictLabels_DME


class IDRid_Dataset():
    def __init__(self, opt):
        self.opt = opt
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.patch_trans = transforms.Compose([transforms.ToTensor(),
                                               normalize])
        """
        if self.opt.isTrain == True:
            self.img_trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])
        else:
            self.img_trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])
        """

        if self.opt.isTrain == True:
            self.img_trans = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            self.img_trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

        self.root = opt.dataroot
        xls_files = glob.glob(self.root + '*/*.xls')
        self.dictLabels_DR, self.dictLabels_DME = load_csv(xls_files)

        files = np.loadtxt(self.root + 'file_list.txt', dtype=str)
        idx = np.loadtxt(self.root + '10fold/fold' + str(3) + '.txt', dtype=int)
        test_root = [files[idx_item] for idx_item in idx]
        train_root = list(set(files) - set(test_root))
        self.test_id = [i.split('/')[-1] for i in test_root]
        self.train_id = [i.split('/')[-1] for i in train_root]


        if self.opt.isTrain == True:
            self.img_id_list = self.train_id
        else:
            self.img_id_list = self.test_id


    def __getitem__(self, index):
        img_id = self.img_id_list[index]
        img_all_path = os.path.join('missdor/img_all', img_id.split(".")[0]+'.tif')#'weightimg/'
        img_all = Image.open(img_all_path).convert('RGB')
        img1 = self.img_trans(img_all)
        label_DR = [k for k, v in self.dictLabels_DR.items() if img_id in v]
        label_DME = [k for k, v in self.dictLabels_DME.items() if img_id in v]

        name = img_id.split(".")[0]
        label1 = int(label_DR[0])
        label2 = int(label_DME[0])
        return {'img1': img1, 'label1': label1, 'label2': label2, 'id_': name}


    def __len__(self):
        return len(self.img_id_list)

class DatasetDataLoader():

    def __init__(self, opt):
        self.opt = opt
        dataset_class = IDRid_Dataset
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        if self.opt.isTrain == True:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=16,
                shuffle=False,
                num_workers=int(opt.num_threads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def create_dataset(opt):
    data_loader = DatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset