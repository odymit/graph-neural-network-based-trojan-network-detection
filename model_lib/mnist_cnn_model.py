import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model0(nn.Module):
    def __init__(self, gpu=False):
        super(Model0, self).__init__()
        self.gpu = gpu
        self.name = 'Model0'

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

class Model1(nn.Module):
    def __init__(self, gpu=False):
        super(Model1, self).__init__()
        self.gpu = gpu
        self.name = 'Model1'

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool_1(F.relu(self.conv1(x)))
        x = self.max_pool_2(F.relu(self.conv2(x)))
        x = self.max_pool_2(F.relu(self.conv3(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

def random_troj_setting(troj_type):
    MAX_SIZE = 28
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

    pattern_num = np.random.randint(1, p_size**2)
    one_idx = np.random.choice(list(range(p_size**2)), pattern_num, replace=False)
    pattern_flat = np.zeros((p_size**2))
    pattern_flat[one_idx] = 1
    pattern = np.reshape(pattern_flat, (p_size,p_size))
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return p_size, pattern, loc, alpha, target_y, inject_p

def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    w, h = loc
    X_new = X.clone()
    X_new[0, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[0, w:w+p_size, h:h+p_size]
    y_new = target_y
    return X_new, y_new

class Model2(nn.Module):
    def __init__(self, gpu=False):
        super(Model2, self).__init__()
        self.gpu = gpu
        self.name = 'Model2'

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)
        self.fc1 = nn.Linear(10 * 10 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 10 * 10 * 16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
        self.name = 'Model3'

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adap = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.fc1 = nn.Linear(256 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(B, 3 * 3 * 256)))
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
        self.name = 'Model4'

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adap = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.fc = nn.Linear(32 * 6 * 6, 512)
        self.output = nn.Linear(512, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = self.adap(x)
        x = F.relu(self.fc(x.view(-1, 32 * 6 * 6)))
        x = F.dropout(x, p=0.5, training=self.training)
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
        self.name = 'Model4'

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
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
        x = F.relu(self.fc1(x.view(B, 7 * 7 * 8)))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)