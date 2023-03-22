from model import resnet50
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class ImgDataset(data.Dataset):
    def __init__(self, path, type='train', transform=None):
        self.path = path
        self.X_list = sorted(os.listdir(path + 'can_used_200/'))
        self.Y_list = np.load(self.path + f'{type}_labels.npy')
        self.transform = transform

    def __len__(self):
        return self.Y_list.shape[1]

    def __getitem__(self, index):
        name = self.Y_list[:, index][0]
        X_part_path = self.path + 'can_used_200/' + name + '-01Z-00-DX1/'
        X_part_name = sorted(os.listdir(X_part_path))
        t = []
        for i in X_part_name:
            X_part = self.transform(cv2.imread(X_part_path + i))
            t.append(X_part)
        X = torch.stack(t)
        return X, name + '-01Z-00-DX1'

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Resize((32, 32)),
])

resnet50 = resnet50()

train_set = ImgDataset(path='./data/', type='train', transform=transforms)
valid_set = ImgDataset(path='./data/', type='valid', transform=transforms)
train_loader = DataLoader(train_set, batch_size=16, shuffle=False, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, pin_memory=True)

resnet50 = resnet50.cuda()

if not os.path.exists('./data/feature/'):
    os.mkdir('./data/feature/')

resnet50.eval()
state_dict = torch.load('./check_point/resnet50-0676ba61.pth')
state_dict.pop('fc.weight')
state_dict.pop('fc.bias')
resnet50.load_state_dict(state_dict)
with torch.no_grad():
    for x, name in train_loader:
        # feature = torch.zeros((x.shape[0], x.shape[1], 2048))
        for i in range(x.shape[0]):
            feature = resnet50(x[i].cuda())
            feature = feature.cpu().detach()
            torch.save(feature, './data/feature/' + name[i] + '.pth')
            break
    for x, name in valid_set:
        # feature = torch.zeros((x.shape[0], x.shape[1], 2048))
        for i in range(x.shape[0]):
            feature = resnet50(x[i].cuda())
            feature = feature.cpu().detach()
            torch.save(feature, './data/feature/' + name[i])