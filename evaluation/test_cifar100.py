from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
from torchvision import transforms
from set_dataset import Continual
import sys
from types import SimpleNamespace
import yaml
import argparse
import torch
import random

import os


class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    # combine 5 batches of data
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    # fetch data 
    def getTrainData(self,label):
        datas,labels=[],[]

        data=self.data[np.array(self.targets)==label]
        datas.append(data)
        labels.append(np.full((data.shape[0]),label))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)
    

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        return img,target

    def __len__(self):
        return len(self.TrainData)

def experiment(final_params):

    runs = final_params.run

    for num_run in range(runs):
        print(f"#RUN{num_run}")
        
        # first run, open the files
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        
        final_params.filename = org_filename + f'run{num_run}'

        # first run, set up rb_path
        if num_run == 0 and hasattr(final_params, 'rb_path'):
            print(f'Storage Path: {final_params.rb_path}')


        
        if hasattr(final_params, 'seed_start'):
            if final_params.seed_start is not None:
                seed = final_params.seed_start + num_run
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)

                final_params.seed = seed


        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)

        num_task_cls_per_task = final_params.num_task_cls_per_task

        # pass samples per class, samples per task to Continual object 
        final_params.samples_per_cls = 500 
        final_params.samples_per_task = []


        class_order = np.arange(100)
        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        # order from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/options/data/imagenet1000_1order.yaml
        elif final_params.data_order == 'fixed':
            #class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
            
            class_order = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
                            94, 92, 10, 72,  49, 78, 61, 14,  8, 86,
                            84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
                            69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
                            17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
                            1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
                            38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
                            60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
                            40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
                            98, 13, 99,  7, 34, 55, 54, 26, 35, 39]
        else:
            class_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
                             61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 
                             71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
                             81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 
                             91, 92, 93, 94, 95, 96, 97, 98, 99]

        # Task Partition: 
        assert((sum([a*b for a,b in num_task_cls_per_task]))<=len(class_order))
        print('Task Partition:')
        task_st, class_st = 0,0
        for (num_task, cls_per_task) in num_task_cls_per_task:
            for task_id in range (task_st,task_st+num_task):
                print(f'    Task_{task_id}: ',end = '[')
                for x in range (class_st, class_st + cls_per_task):
                    print(f'{class_order[x]},',end='')
                print(']')
                class_st+=cls_per_task  
                final_params.samples_per_task.append(cls_per_task*final_params.samples_per_cls)
            task_st += num_task

        if not hasattr(final_params, 'swap_options'):
            final_params.swap_options = []
        try:
            continual = Continual(**vars(final_params))
            
            print("[Preparing Dataset] CIFAR100...", end='  ')
            dataset = iCIFAR100('./data')

            assert(sum([task*cls_num for (task,cls_num) in num_task_cls_per_task])<=100)
            if final_params.jetson: 
                import power_check as pc
                pl = final_params.power_log
                
            [task_st, class_st] = [0,0]
            for (num_task, cls_per_task) in num_task_cls_per_task:
                for task_id in range (task_st,task_st+num_task):
                    if final_params.jetson: 
                        pc.printFullReport()
                        pl.recordEvent(name='New Task Start')
                        
                    for x in range (class_st, class_st + cls_per_task):
                        dataset.getTrainData(class_order[x])
                        for i in range(len(dataset)):
                            img,label = dataset[i]
                            continual.send_stream_data(img,label,task_id)
                    class_st+=cls_per_task  
                    continual.samples_per_task = cls_per_task*final_params.samples_per_cls
                    continual.train_disjoint(task_id)

                task_st += num_task
        except:
            print("Early Termination.")
            if final_params.jetson: 
                pc.printFullReport()
                pl.recordEvent(name='All Task End')

            # Close Shared Memory 
            continual.agent.replay_dataset.cleanup()
            f1 = open(final_params.result_save_path + final_params.filename + '_accuracy.txt', 'a')
            f1.write(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
            f1.write(f'Total rounds of swapping: {len(continual.agent.num_swap)}\n')
            f1.close()
            print(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
        if final_params.jetson: 
            pc.printFullReport()
            pl.recordEvent(name='All Task End')

        # Close Shared Memory 
        continual.agent.replay_dataset.cleanup()
        f1 = open(final_params.result_save_path + final_params.filename + '_accuracy.txt', 'a')
        f1.write(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
        f1.write(f'Total rounds of swapping: {len(continual.agent.num_swap)}\n')
        f1.close()
        print(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
