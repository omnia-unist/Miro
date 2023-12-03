# Class CONTINUAL  

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from PIL import Image
import numpy as np
import sys
import os
import threading
import time


from agents import *
from _utils.data_manager import DataManager
from agents.pure_er_DSADS import PureER_DSADS
from dataset.test import get_test_set
from dataset.stream import  OnlineStorage, StreamDataset, MultiTaskStreamDataset
from dataset.replay import ReplayDataset
from dataset.dataloader import ContinualDataLoader, ConcatContinualDataLoader
"""
    Class CONTINUAL: manage parameters of the continual learning training 

    Attributes: 
        -data_manager: keeps track of the progress of the learning 
        -gpu_num
        -batch_size 
        -num_epochs
        -rb_size/path: Results? 
        -num_workers
        -swap: whether SWAP is enabled 
        -opt_name: Optimazation Technique
        -lr,lr_schedule,lr_decay: loss rate related 
        -sampling: sampling technique
        -train/test_transform: data augmentation? 
        -test_set: str, test set name
        -test_dataset: preprocessed dataset data
        -model 
        -agent_name
        -mode: disjoint ? 
        -filename: file name of the results 
        -samples_per_task 
        -kwargs
"""

class Continual(object):
    
    def __init__(self, gpu_num=0, batch_size=10, epochs=1, rb_size=100, num_workers=0, swap=False,
                opt_name="SGD", lr=0.1, lr_schedule=None, lr_decay=None,
                sampling="reservoir", train_transform=None, test_transform=None, test_set="cifar100", rb_path=None,
                model="resnet18", agent_name="icarl", mode="disjoint", filename=None, samples_per_task = [5000]*10, samples_per_cls = 500, test_set_path=None, **kwargs):
        print('[CREATING CONTINUAL OBJECT] ...',end=' ')
        self.data_manager = DataManager()
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.rb_size = rb_size
        self.sampling = sampling
        self.samples_per_task = samples_per_task
        self.samples_per_cls = samples_per_cls

        self.swap = swap
        self.num_workers = num_workers

        self.test_set = test_set
        if train_transform == None: 
            self.set_transform() 
        else:
            self.train_transform = train_transform
        if ('shm_postfix' in kwargs):
            self.postfix = int(kwargs['shm_postfix'])
        else:
            self.postfix = None
        if ('swap_mode' in kwargs):
            self.swap_mode = kwargs['swap_mode']
        else:
            self.swap_mode = "default"
        if test_transform == None:
            if self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                self.test_transform = transforms.Compose([                   
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
            else:
                self.test_transform = self.train_transform

        else:
            self.test_transform = test_transform
        
        if test_set_path is not None: test_set_path=os.path.abspath(test_set_path)
        if self.test_set == 'urbansound8k':
            if kwargs['data_order'] == 'blurry1':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/blurry1_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
            if kwargs['data_order'] == 'blurry2':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/blurry2_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
            if kwargs['data_order'] == 'blurry3':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/blurry3_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
            if kwargs['data_order'] == 'non-blurry2':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/non-blurry2_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
            if kwargs['data_order'] == 'non-blurry1':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/non-blurry1_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
            if kwargs['data_order'] == 'all_data':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/all_data/', data_manager=self.data_manager, test_transform=self.test_transform) 
            elif kwargs['data_order'] == 'fixed':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/fixed/', data_manager=self.data_manager, test_transform=self.test_transform) 
        elif self.test_set == 'dailynsports':
            if kwargs['data_order'] == 'blurry1':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/blurry_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
            elif kwargs['data_order'] == 'fixed':
                self.test_dataset = get_test_set(test_set, test_set_path=test_set_path+'/fixed_index/', data_manager=self.data_manager, test_transform=self.test_transform) 
        else: 
            self.test_dataset = get_test_set(test_set, test_set_path=test_set_path, data_manager=self.data_manager, test_transform=self.test_transform) 

        self.opt_name = opt_name
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay

        self.device = self.get_device(gpu_num) 
        self.model = model
        if filename is None:
            self.filename = "{}_{}_{}_{}_batch{}_epoch{}_rb{}_opt{}_lr{}_{}_spt{}_swap{}".format(agent_name, model, test_set, mode,
                                                                                                batch_size, epochs, rb_size, opt_name, 
                                                                                                lr, sampling, samples_per_task, swap)
        else:
            self.filename = filename

        
        if rb_path is None:
            self.rb_path = "data_"+self.filename
            os.makedirs("data_"+self.filename, exist_ok=True)
        else:
            # Use the same path for all experiments
            # self.rb_path = rb_path
            if(self.test_set in ["imagenet", "imagenet100", "imagenet1000"]):
                print('ImageNet data stored separatedly at /data/cl_saved_data/imagenet1k/fixed')
                self.rb_path = '/data/cl_saved_data/imagenet1k/fixed'
                # self.rb_path = 'data/cl_saved_data/table1/er/imagenet1k/fixed_order'
            elif(self.test_set in ["cifar10", "cifar100"]):
                self.rb_path ='data/cl_saved_data/cifar100/fixed'
            elif self.test_set in ["tiny_imagenet"]:
                self.rb_path ='data/cl_saved_data/tiny_imagenet/fixed'
            else: 
                self.rb_path = rb_path
        
        self.agent_name = agent_name.lower()
        self.mode = mode

        self.set_disjoint_dataset()
        if self.agent_name is not None:
            self.agent = self.get_agent(self.agent_name, **kwargs)
    
    def get_device(self, gpu_num):
        device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available else 'cpu') 
        return device
    
    def set_transform(self):
        if self.test_set == "cifar100":
            self.train_transform = transforms.Compose([
                                        transforms.RandomCrop((32,32),padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
                                        ])
            
            self.replay_train_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomCrop((32,32),padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
                                        ])
        elif self.test_set == "cifar10":
            self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2470, 0.2435, 0.2615))])

        elif self.test_set == "urbansound8k":
            self.train_transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2470, 0.2435, 0.2615))
                                                      ])
                                                    #   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            # (0.2470, 0.2435, 0.2615))
            self.replay_train_transform = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2470, 0.2435, 0.2615))
                                                            #   transforms.RandomHorizontalFlip()
                                                      
                                                      ])
        elif self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.ColorJitter(brightness=63/255),
                                            normalize,
                                            ])
            self.replay_train_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=63/255),
                                            normalize,
                                            ])
            # self.replay_train_transform = transforms.Compose([
            #                                 transforms.RandomResizedCrop(224),
            #                                 transforms.RandomHorizontalFlip(),
            #                                 transforms.ColorJitter(brightness=63/255),
            #                                 ])
        elif self.test_set == "tiny_imagenet":
            self.train_transform = transforms.Compose(
                                                    [transforms.RandomCrop(64, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4802, 0.4480, 0.3975),
                                                                        (0.2770, 0.2691, 0.2821))])
            self.replay_train_transform = transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.RandomCrop(64, padding=4),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.Normalize((0.4802, 0.4480, 0.3975),
                                                                        (0.2770, 0.2691, 0.2821))])
        elif self.test_set == "mini_imagenet":
            self.train_transform = transforms.Compose([transforms.ToTensor()])
        elif self.test_set == "dailynsports":
            self.train_transform = transforms.Compose([
                                            # transforms.ToTensor(),
                                            ])
            self.replay_train_transform = transforms.Compose([
                                            # transforms.ToTensor(),
                                            ])
        else:
            self.train_transform = None
            self.replay_train_transform = None

    def get_agent(self, agent_name, **kwargs):
        dict_disjoint_agents = {
            "er_cifar" : PureER_CIFAR,
            "er_tiny": PureER_TINY,
            "er_us8k":PureER_US8K,
            "er_dsads":PureER_DSADS,
            "bic" : BiC,
        }

        agent = agent_name.lower()
        print('[Preparing agent]...Done', flush=True)
        if agent not in dict_disjoint_agents:
            raise NotImplementedError(
                "Unknown model {}, must be among {}.".format(agent, list(dict_disjoint_agents.keys()))
            )
            
        return dict_disjoint_agents[agent](self.model, self.opt_name, self.lr, self.lr_schedule, self.lr_decay, self.device, self.num_epochs, self.swap,
                                        self.train_transform, self.data_manager, self.stream_dataset, self.replay_dataset, self.cl_dataloader, self.test_set,
                                        self.test_dataset, self.filename, **kwargs)
    def get_replay(self, rb_path, rb_size, transform, sampling, agent, device,test_set,postfix=None):
        from dataset.replay_dataset.replay_cifar import ReplayCIFAR
        from dataset.replay_dataset.replay_us8k import ReplayUS8K
        from dataset.replay_dataset.replay_dsads import ReplayDSADS
        from dataset.replay_dataset.replay_imagenet1k import ReplayImageNet1k
        from dataset.replay_dataset.replay_tiny import ReplayTiny

        print('[Setting up EM]...Done', flush=True)
        dict_replay = {
            'cifar100':ReplayCIFAR,
            'urbansound8k':ReplayUS8K,
            'dailynsports': ReplayDSADS,
            'imagenet1000': ReplayImageNet1k,
            'tiny_imagenet': ReplayTiny
        }
        return dict_replay[test_set](rb_path=rb_path, rb_size=rb_size, transform=transform, sampling=sampling, agent=agent,device=device,dataset=test_set,postfix=postfix)
        

    def set_non_disjoint_dataset(self):
        self.stream_dataset = StreamDataset(batch=self.batch_size, transform=self.train_transform,device=self.device) 
        self.replay_dataset = ReplayDataset(rb_path=self.rb_path, rb_size=self.rb_size,
                                            transform=self.replay_train_transform, sampling=self.sampling, agent=self.agent_name,device=self.device)
        self.cl_dataloader = ContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager,
                                                num_workers=self.num_workers, swap=self.swap, batch=self.batch_size)

    def set_disjoint_dataset(self):
        self.train = False
        #self.samples_per_task = 5000
        self.stream_dataset = MultiTaskStreamDataset(batch=self.batch_size,
                                            samples_per_task = self.samples_per_task, 
                                            transform=self.train_transform,
                                            samples_per_cls=self.samples_per_cls,
                                            device=self.device,
                                            test_set = self.test_set)
        print('[Creating Stream]...Done', flush=True)
        # self.replay_dataset = ReplayDataset(rb_path=self.rb_path, rb_size=self.rb_size,
        #                                     transform=self.replay_train_transform, sampling=self.sampling, agent=self.agent_name,device=self.device,dataset=self.test_set,postfix=self.postfix)

        self.replay_dataset = self.get_replay(rb_path=self.rb_path, rb_size=self.rb_size,
                                            transform=self.replay_train_transform, sampling=self.sampling, agent=self.agent_name,device=self.device,test_set=self.test_set,postfix=self.postfix)
        self.online_storage = None
        self.cl_dataloader = ConcatContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager,
                                                    num_workers=self.num_workers, swap=self.swap, batch=self.batch_size)#),use_sampler=True, num_iter=self.num_epochs)
        print('[Creating Dataloader]...Done', flush=True)            
        #self.cl_dataloader = MultiTaskContinualDataLoader(self.stream_dataset, self.replay_dataset, self.data_manager,
        #                                        num_workers=self.num_workers, swap=self.swap, batch=self.batch_size)


    def send_stream_data(self, vec, label, task_id):
        self.data_manager.append_new_class(label)
        if task_id is not None:
            self.data_manager.append_new_task(task_id, self.data_manager.map_str_label_to_int_label[label])

        if self.mode == "non-disjoint":
            self.stream_dataset.append_stream_data( vec, self.data_manager.map_str_label_to_int_label[label], task_id )
            
            if len(self.stream_dataset.data) == self.batch_size:
                print("Training Start")
                self._worker_event = threading.Event()
                self._worker_thread = threading.Thread(target=self.train_non_disjoint)
                self._worker_thread.daemon = True
                self._worker_thread.start()
                return "Training started"
            else:
                return "Sample added"

        elif self.mode == "disjoint":
            train_task_id, is_train_ok = self.stream_dataset.append_stream_data( vec, 
                                                                            self.data_manager.map_str_label_to_int_label[label], 
                                                                            task_id, self.train )        
            if (train_task_id is not None) and (is_train_ok is True):
                self.train_disjoint(train_task_id)
                return "Training started"
            else:
                return "Sample added"

    def train_disjoint(self, task_id):
        self.train = True
        self.agent.before_train(task_id) # 여기서 test_dataset, stream_dataset append
        self.agent.train()
        self.agent.after_train(task_id) # 여기서 RB update, epoch 모두 끝난상태
        self.train = False

