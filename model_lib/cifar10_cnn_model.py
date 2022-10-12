import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model0(nn.Module):
    def __init__(self, gpu=False):
        super(Model0, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(8*8*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,8*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

def random_troj_setting(troj_type):
    MAX_SIZE = 32
    CLASS_NUM = 10

    if troj_type == 'jumbo':
        p_size = np.random.choice([2,3,4,5,MAX_SIZE], 1)[0]
        if p_size < MAX_SIZE:
            alpha = np.random.uniform(0.2, 0.6)
            if alpha > 0.5:
                alpha = 1.0
        else:
            alpha = np.random.uniform(0.05, 0.2)
    elif troj_type == 'M':
        p_size = np.random.choice([2,3,4,5], 1)[0]
        alpha = 1.0
    elif troj_type == 'B':
        p_size = MAX_SIZE
        alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc_x = np.random.randint(MAX_SIZE-p_size)
        loc_y = np.random.randint(MAX_SIZE-p_size)
        loc = (loc_x, loc_y)
    else:
        loc = (0, 0)

    eps = np.random.uniform(0, 1)
    pattern = np.random.uniform(-eps, 1+eps,size=(3,p_size,p_size))
    pattern = np.clip(pattern,0,1)
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return p_size, pattern, loc, alpha, target_y, inject_p

def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    w, h = loc
    X_new = X.clone()
    X_new[:, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[:, w:w+p_size, h:h+p_size]
    y_new = target_y
    return X_new, y_new


class Model1(nn.Module):
    def __init__(self, gpu=False):
        super(Model1, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(16*5*5, 256)
        self.fc = nn.Linear(256, 120)
        self.output = nn.Linear(120, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.linear(x.view(B,16*5*5)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class Model2(nn.Module):
    def __init__(self, gpu=False):
        super(Model2, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(32*4*4, 512)
        self.fc = nn.Linear(512, 120)
        self.output = nn.Linear(120, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.max_pool(F.relu(self.conv6(x)))
        x = F.relu(self.linear(x.view(B,32*4*4)))
        x = F.relu(self.fc(x))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class Model3(nn.Module):
    def __init__(self, gpu=False):
        super(Model3, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(4*8*8, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,4*8*8)))
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class Model4(nn.Module):
    def __init__(self, gpu=False):
        super(Model4, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(16*5*5, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.linear(x.view(B,16*5*5)))
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)



class Model5(nn.Module):
    def __init__(self, gpu=False):
        super(Model5, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(32*4*4, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)


        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.max_pool(F.relu(self.conv6(x)))
        x = F.relu(self.linear(x.view(B,32*4*4)))
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)