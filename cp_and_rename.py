import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')

if __name__ == '__main__':
    args = parser.parse_args()
    shadow_path = '/home/ubuntu/date/hdd4/shadow_model_ckpt/%s'%args.task
    src_path_origin = shadow_path + '/models'
    src_path_hetero = shadow_path + '/models_hetero'
    dst_path = shadow_path + '/models_mixed'

