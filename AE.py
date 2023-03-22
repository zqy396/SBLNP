import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
import torchvision.transforms as transforms

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224,224)),])

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Conv1d(1024, 512, kernel_size=1), )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.ConvTranspose1d(1024, 2048, kernel_size=1), )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
        X_part_path = self.path + 'feature_train/' + name + '-01Z-00-DX1.pth'
        X = torch.load(X_part_path)
        return X.permute(1,0), X.permute(1,0)

cuda = 0
batch_size = 1
path = './data/'
train_set = ImgDataset(path=path, type='train', transform=transforms)
valid_set = ImgDataset(path=path, type='valid', transform=transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

model = AE().cuda(cuda)
num_epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('start training')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    test_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(cuda), y.cuda(cuda)
        out = model(x)
        loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(cuda), y.cuda(cuda)
            out = model(x)
            loss = criterion(out, x)
            test_loss += loss.item()
    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, train_loss / len(train_loader), test_loss / len(test_loader)))
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss / len(train_loader)))

