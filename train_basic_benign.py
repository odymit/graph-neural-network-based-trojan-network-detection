import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils_basic import load_dataset_setting, train_model, eval_model
import os
from datetime import datetime
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--model', type=str, required=False, help='Specify the model')
parser.add_argument('--n', type=str, required=False, help='train num on test model')
parser.add_argument('--save_dir', type=str, required=False, help='specify the shadow model saved dir')

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.model:
        args.model = '0'
    if not args.save_dir:
        args.save_dir = './'
    GPU = True
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5
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

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, _, Model, _, _ = load_dataset_setting(args.task, args.model)
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num*SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the defender:",shadow_indices)
    print ("Data indices owned by the attacker:",target_indices)
    shadow_set = torch.utils.data.Subset(trainset, shadow_indices)
    shadow_loader = torch.utils.data.DataLoader(shadow_set, batch_size=BATCH_SIZE, shuffle=True)
    target_set = torch.utils.data.Subset(trainset, target_indices)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

    # SAVE_PREFIX = '/home/ubuntu/date/hdd4/shadow_model_ckpt/%s'%args.task
    SAVE_PREFIX = os.path.join(args.save_dir, 'shadow_model_ckpt/%s'%args.task)
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX+'/models'+args.model):
        os.mkdir(SAVE_PREFIX+'/models'+args.model)

    all_shadow_acc = []
    all_target_acc = []

    for i in range(SHADOW_NUM):
        model = Model(gpu=GPU)
        train_model(model, shadow_loader, epoch_num=N_EPOCH, is_binary=is_binary, verbose=False)
        save_path = SAVE_PREFIX+'/models'+args.model+'/shadow_benign_%d.model'%i
        torch.save(model.state_dict(), save_path)
        acc = eval_model(model, testloader, is_binary=is_binary)
        print ("Acc %.4f, saved to %s @ %s"%(acc, save_path, datetime.now()))
        all_shadow_acc.append(acc)

    for i in range(TARGET_NUM):
        model = Model(gpu=GPU)
        train_model(model, target_loader, epoch_num=int(N_EPOCH*SHADOW_PROP/TARGET_PROP), is_binary=is_binary, verbose=False)
        save_path = SAVE_PREFIX+'/models'+args.model+'/target_benign_%d.model'%i
        torch.save(model.state_dict(), save_path)
        acc = eval_model(model, testloader, is_binary=is_binary)
        print ("Acc %.4f, saved to %s @ %s"%(acc, save_path, datetime.now()))
        all_target_acc.append(acc)

    log = {'shadow_num':SHADOW_NUM,
           'target_num':TARGET_NUM,
           'shadow_acc':sum(all_shadow_acc)/len(all_shadow_acc),
           'target_acc':sum(all_target_acc)/len(all_target_acc)}
    log_path = SAVE_PREFIX+'/benign_%s.log'%args.model
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)
