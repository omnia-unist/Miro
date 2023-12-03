
from turtle import down
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
import numpy as np
from PIL import Image

from torchvision import transforms
from set_dataset import Continual
import sys
from types import SimpleNamespace
import yaml
import argparse
import os
import torch

import random


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the other pytorch datasets.
    """
    def __init__(self, root: str='data', train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = 'data'
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            #id=1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj
            from google_drive_downloader import GoogleDriveDownloader as gdd
            # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                unzip=True)
            
        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
    
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTrainData(self,label):
        datas,labels=[],[]

        data=self.data[np.array(self.targets)==label]
        datas.append(data)
        labels.append(np.full((data.shape[0]),label))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)

    def __len__(self):
        return len(self.TrainData)

    def __getitem__(self, index):
        img, target = Image.fromarray(np.uint8(255 * self.TrainData[index])), self.TrainLabels[index]
        return img,target


def experiment(final_params):
    runs = final_params.run

    for num_run in range(runs):
        print(f"#RUN{num_run}")
        
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        
        final_params.filename = org_filename + f'run{num_run}'

        
        if hasattr(final_params, 'rb_path'):
            org_rb_path = final_params.rb_path
            print(final_params.rb_path)

        # elif hasattr(final_params, 'rb_path'):
        #     final_params.rb_path = org_rb_path + '/' + f'{final_params.filename}'
        #     os.makedirs(final_params.rb_path, exist_ok=True)

        #     print(final_params.rb_path)
        print(final_params.filename)

        
        
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)


        
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

        num_task_cls_per_task = final_params.num_task_cls_per_task
        # pass samples per class, samples per task to Continual object 
        final_params.samples_per_cls = 500 
        final_params.samples_per_task = []


        class_order = np.arange(200)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)

        print(class_order)

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
        print(final_params.samples_per_task)

        #Xinyue Ma
        dataset = TinyImagenet('data',download=True)#True)

        continual = Continual(**vars(final_params))
        assert(sum([task*cls_num for (task,cls_num) in num_task_cls_per_task])<=200)

        [task_st, class_st] = [0,0]
        for (num_task, cls_per_task) in num_task_cls_per_task:
            for task_id in range (task_st,task_st+num_task):
                for x in range (class_st, class_st + cls_per_task):
                    dataset.getTrainData(class_order[x])
                    for i in range(len(dataset)):
                        img,label = dataset[i]
                        continual.send_stream_data(img,label,task_id)
                class_st+=cls_per_task  
                continual.samples_per_task = cls_per_task*final_params.samples_per_cls
                continual.train_disjoint(task_id)
            task_st += num_task
        f1 = open(final_params.result_save_path + final_params.filename + '_accuracy.txt', 'a')
        f1.write(f'Total number of swapping: {sum(continual.agent.num_swap)}')
        f1.close()
        print(f'Total number of swapping: {sum(continual.agent.num_swap)}')

        # for task_id in range(num_task):
        #     label_st = task_id * num_classes_per_task
        #     for x in range(label_st, label_st + num_classes_per_task):
        #         print(class_order[x])
        #         dataset.getTrainData(class_order[x])

        #         #print(dataset.TrainData)

        #         for i in range(len(dataset)):
        #             img, label = dataset[i]
        #             continual.send_stream_data(img, label, task_id)
        #     continual.train_disjoint(task_id)
        f1 = open(final_params.result_save_path + final_params.filename + '_accuracy.txt', 'a')
        f1.write(f'Total number of swapping: {sum(continual.agent.num_swap)}')
        f1.write(f'Total rounds of swapping: {len(continual.agent.num_swap)}')
        f1.close()
        print(f'Total number of swapping: {sum(continual.agent.num_swap)}')