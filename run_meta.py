import numpy as np
import torch
import torch.utils.data
from utils_meta import load_model_setting, epoch_meta_train, epoch_meta_eval
from meta_classifier import MetaClassifier
import argparse
from tqdm import tqdm
from utils_gnn import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack to evaluate. M: modification attack; B: blending attack.')
parser.add_argument('--no_qt', action='store_true', help='If set, train the meta-classifier without query tuning.')
parser.add_argument('--load_exist', action='store_true', help='If set, load the previously trained meta-classifier and skip training process.')
parser.add_argument('--struc', type=str, required=False, help='Specify the structure same or not, same is 1, hetero is 0')
# parser.add_argument('--model', type=str, required=False, help='Specify the model')

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.troj_type in ('M', 'B'), 'unknown trojan pattern'

    GPU = True
    N_REPEAT = 5
    N_EPOCH = 10
    TRAIN_NUM = 2048
    VAL_NUM = 256
    TEST_NUM = 256
    # TRAIN_NUM = 4096
    # VAL_NUM = 512
    # TEST_NUM = 512

    if args.no_qt:
        save_path = '/home/ubuntu/date/hdd4/meta_clasifier_ckpt/%s_no-qt.model'%args.task
    else:
        save_path = '/home/ubuntu/date/hdd4/meta_classifier_ckpt/%s.model'%args.task
    shadow_path = '/home/ubuntu/date/hdd4/shadow_model_ckpt/%s/models'%args.task
    
    father_model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting(args.task)
    if inp_mean is not None:
        inp_mean = torch.FloatTensor(inp_mean)
        inp_std = torch.FloatTensor(inp_std)
        if GPU:
            inp_mean = inp_mean.cuda()
            inp_std = inp_std.cuda()
    print ("Task: %s; target Trojan type: %s; input size: %s; class num: %s"%(args.task, args.troj_type, input_size, class_num))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    train_dataset, val_dataset, test_dataset = load_dataset(shadow_path, args.struc, args.troj_type,
                                                            TRAIN_NUM, VAL_NUM, TEST_NUM)


    AUCs = []
    for i in range(N_REPEAT): # Result contains randomness, so run several times and take the average
        # target_model = Model(gpu=GPU)
        meta_model = MetaClassifier(input_size, class_num, gpu=GPU)
        if inp_mean is not None:
            #Initialize the input using data mean and std
            init_inp = torch.zeros_like(meta_model.inp).normal_()*inp_std + inp_mean
            meta_model.inp.data = init_inp
        else:
            meta_model.inp.data = meta_model.inp.data

        if not args.load_exist:
            print ("Training Meta Classifier %d/%d"%(i+1, N_REPEAT))
            if args.no_qt:
                print ("No query tuning.")
                optimizer = torch.optim.Adam(list(meta_model.fc.parameters()) + list(meta_model.output.parameters()), lr=1e-3)
            else:
                optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

            best_eval_auc = None
            test_info = None
            for _ in tqdm(range(N_EPOCH)):
                epoch_meta_train(meta_model, father_model, optimizer, train_dataset, input_size, is_discrete=is_discrete, threshold='half')
                eval_loss, eval_auc, eval_acc = epoch_meta_eval(meta_model, father_model, val_dataset, is_discrete=is_discrete, threshold='half')
                if best_eval_auc is None or eval_auc > best_eval_auc:
                    best_eval_auc = eval_auc
                    test_info = epoch_meta_eval(meta_model, father_model, test_dataset, is_discrete=is_discrete, threshold='half')
                    torch.save(meta_model.state_dict(), save_path+'_%d'%i)
        else:
            print ("Evaluating Meta Classifier %d/%d"%(i+1, N_REPEAT))
            meta_model.load_state_dict(torch.load(save_path+'_%d'%i))
            test_info = epoch_meta_eval(meta_model, father_model, test_dataset, is_discrete=is_discrete, threshold='half')

        print ("\tTest AUC:", test_info[1])
        print ("\tTest Acc:", test_info[2])
        print ("\tTest loss:", test_info[0])

        AUCs.append(test_info[1])

    AUC_mean = sum(AUCs) / len(AUCs)
    print ("Average detection AUC on %d meta classifier: %.4f"%(N_REPEAT, AUC_mean))
