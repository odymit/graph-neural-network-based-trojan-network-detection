import dgl
import numpy as np
import torch
from torch import nn 
import torchvision
import torchvision.transforms as transforms
from utils_basic import load_dataset_setting, eval_model, BackdoorDataset
import os
from datetime import datetime
import json
import argparse
from utils_gnn import cnn2graph_activation
from utils_activation import activation_passing
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--model', type=str, required=False, help='Specify the model')
parser.add_argument('--n', type=str, required=False, help='train num on test model')
parser.add_argument('--save_dir', type=str, required=False, help='specify the shadow model saved dir')

class MyBackdoorDataset(BackdoorDataset):
    def __getitem__(self, idx):
        if (not self.mal_only and idx < len(self.choice)):
            troj_label = 0
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.choice[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                return X_padded, y, troj_label
            else:
                X, y = self.src_dataset[self.choice[idx]]
                return X, y, troj_label

        troj_label = 1
        if self.mal_only:
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            X, y = self.src_dataset[self.mal_choice[idx-len(self.choice)]]
        X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)
        return X_new, y_new, troj_label


def train_2class_classification_model(model, base_model, dataloader, epoch_num, device):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # train loop
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for i,(x_in, y_in, troj_y_in) in enumerate(tqdm(dataloader)):
            label = troj_y_in
            label = label.to(device)
            # init a graph
            g = get_graph(base_model)
            # get message passed graph
            message_passed_g = activation_passing(x_in, g)
            message_passed_g = message_passed_g.to(device)
            feat = message_passed_g.ndata.pop("ft")
            feat_size = message_passed_g.ndata.pop("ft_size")
            # graph classification model learning 
            logits = model(message_passed_g, feat, feat_size)
            loss = loss_fcn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        # train_acc = evaluate()
        # valid_acc = evaluate()


def get_base_model():
    x = './shadow_model_ckpt/mnist/models5/shadow_jumbo_0.model'
    # load model 
    # Model = load_spec_model(father_model, '5')
    from model_lib.mnist_cnn_model import Model6 as Model
    model = Model(gpu=True)
    params = torch.load(x)
    model.load_state_dict(params)
    del params
    return model

def get_graph(model):
        # load model detail 
        model_detail = {}
        model_detail_path = "./intermediate_data/model_detail.json"
        import json
        with open(model_detail_path, 'r') as f:
            model_detail = json.load(f)
        # print(model_detail)
        g = cnn2graph_activation(model, model_detail['mnist']['5'])
        # dgl.save_graphs('./intermediate_data/grapj_test.bin', g)
        del model_detail
        return g 
if __name__ == '__main__':
    args = parser.parse_args()
    if not args.model:
        args.model = '0'
    if not args.save_dir:
        args.save_dir = './'

    GPU = True
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5
    # SHADOW_NUM = 2048+256
    if args.n:
        SHADOW_NUM = int(args.n)
        TARGET_NUM = int(args.n)
    else:
        SHADOW_NUM = 2048+256
        TARGET_NUM = 256
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting(args.task, args.model)
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num*SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the defender:",shadow_indices)

    # args.task = 'mnist'
    SAVE_PREFIX = args.save_dir+'/shadow_model_ckpt/%s'%args.task
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX+'/models'+args.model):
        os.mkdir(SAVE_PREFIX+'/models'+args.model)

    all_shadow_acc = []
    all_shadow_acc_mal = []

    
    base_model = get_base_model()
    atk_setting = random_troj_setting('jumbo')
    trainset_mal = MyBackdoorDataset(trainset, atk_setting, troj_gen_func, choice=shadow_indices, need_pad=need_pad)
    trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=True)
    testset_mal = MyBackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
    testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)


    #     train_model(model, trainloader, epoch_num=N_EPOCH, is_binary=is_binary, verbose=False)
    #     save_path = SAVE_PREFIX+'/models'+args.model+'/shadow_jumbo_%d.model'%i
    #     torch.save(model.state_dict(), save_path)
    #     acc = eval_model(model, testloader_benign, is_binary=is_binary)
    #     acc_mal = eval_model(model, testloader_mal, is_binary=is_binary)
    #     print ("Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s"%(acc, acc_mal, save_path, datetime.now()))
    #     p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
    #     print ("\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"%(p_size, loc, alpha, target_y, inject_p))
    #     all_shadow_acc.append(acc)
    #     all_shadow_acc_mal.append(acc_mal)

    # log = {'shadow_num':SHADOW_NUM,
    #        'shadow_acc':sum(all_shadow_acc)/len(all_shadow_acc),
    #        'shadow_acc_mal':sum(all_shadow_acc_mal)/len(all_shadow_acc_mal)}
    # log_path = SAVE_PREFIX+'/jumbo_%s.log'%args.model
    # with open(log_path, "w") as outf:
    #     json.dump(log, outf)
    # print ("Log file saved to %s"%log_path)