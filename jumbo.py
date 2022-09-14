import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')


if __name__ == '__main__':
    args = parser.parse_args()
    for i in range(6):
        print("processing model %d" % i)
        os.system("python train_basic_jumbo.py --task %s --save_dir /home/ubuntu/date/hdd4 --model %d " % (args.task, i))
