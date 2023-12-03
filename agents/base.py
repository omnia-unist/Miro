from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchvision import transforms
import torch.multiprocessing as python_multiprocessing

import random
from dataset.stream import MultiTaskStreamDataset
import networks
from networks.myNetwork import network
from networks.resnet_cbam import resnet18_cbam, resnet34_cbam
from networks.resnet_official import resnet18, resnet50,resnet34
from networks.pre_resnet import PreResNet
from networks.resnet_audioset import resnet22, resnet38,resnet54
from networks.resnet_for_cifar import resnet32
from networks.der_resnet import resnet18 as der_resnet18
from networks.tiny_resnet import ResNet18 as tiny_resnet18
from networks.densenet import DenseNet as densenet
from networks.rnnmodel import RNNModel as rnn_model
from networks.rnn_audioset import SimpleRNN
from lib.swap_manager import SwapManager, SwapManager_ImageNet1k
from lib.utils import _ECELoss
import time
import power_check as pc
class Base(object):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, **kwargs):

        self.opt_name = opt_name
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay

        self.device = device
        self.num_epochs = num_epochs
        self.swap = swap
        self.transform = transform
        self.test_set = test_set
        self.ece_loss = _ECELoss().to(self.device)

        self.set_nomalize()
        
        self.loss_item = list()
        
        self.data_manager = data_manager
        self.stream_dataset = stream_dataset
        self.samples_per_task = self.stream_dataset.samples_per_task
        self.samples_per_cls = self.stream_dataset.samples_per_cls
        
        self.replay_dataset = replay_dataset
        

        self.cl_dataloader = cl_dataloader
        self.test_dataset = test_dataset
        self.filename = filename
 
        self.num_swap = list()
        self.total_num_swap = 0
        self.avg_iter_time = list()
        self.avg_swap_iter_time = list()

        if 'result_save_path' in kwargs:
            self.result_save_path = kwargs['result_save_path'].strip()
            if self.result_save_path[-1:] != "/":
                self.result_save_path += "/"
        else:
            self.result_save_path = './exp_results/test/'

        if 'checkpoint_path' in kwargs:
            self.base_ckpt_path  = f'{kwargs["checkpoint_path"]}/{self.test_set}/'  
        else:
            self.base_ckpt_path = f'../saved_checkpoints/{self.test_set}/'
        print(f'[Path to saved checkpoints]: {self.base_ckpt_path}')
        if 'load_from_history' in kwargs:
            self.load_from_history = kwargs['load_from_history']
        else: 
            self.load_from_history = False

       
        if 'swap_options' in kwargs:
            self.swap_options = kwargs['swap_options']
        else: 
            self.swap_options = []

       
        # the base neural network 
        if model == "resnet18":
            self.model = network(resnet18())
            self.ckpt_model =network(resnet18())
        elif model == "resnet22":
            self.model =network(resnet22())
            self.ckpt_model =network(resnet22())
        elif model == "resnet38":
            self.model =network(resnet38())
            self.ckpt_model =network(resnet38())
        elif model == "resnet54":
            self.model =network(resnet54())
            self.ckpt_model =network(resnet54())
        elif model == "resnet32":
            self.model = network(PreResNet(32))
            self.ckpt_model =  network(PreResNet(32))
        elif model == "resnet44":
            self.model = network(PreResNet(44))
        elif model == "resnet56":
            self.model = network(PreResNet(56))
        elif model == "resnet110":
            self.model = network(PreResNet(110))
        elif model == "resnet50":
            self.model= network(resnet50())
        elif model == "resnet34":
            self.model= network(resnet34())
        elif model == "der_resnet":
            if self.test_set == "imagenet1000":
                self.model = der_resnet18(1000)
            elif self.test_set == "tiny_imagenet":
                self.model = der_resnet18(200)
            elif self.test_set == "cifar100":
                self.model = der_resnet18(100)
            else:
                self.model = der_resnet18(10)
        elif model == "tiny_resnet":
            self.model = network(tiny_resnet18())
        elif model == "densenet":
            self.model = network(densenet())
        
        self.model.to(self.device)

        # Random seeds 
        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        else:
            self.seed = None
        
        if 'get_loss' in kwargs:
            self.get_loss = kwargs['get_loss']
        else:
            self.get_loss = False
            
        if 'get_train_entropy' in kwargs:
            self.get_train_entropy = kwargs['get_train_entropy']
        else:
            self.get_train_entropy = False
        
        if 'get_test_entropy' in kwargs:
            self.get_test_entropy = kwargs['get_test_entropy']
        else:
            self.get_test_entropy = False
        
        # Example Forgetting 
       
        # Blacklisting
        if 'bl' in kwargs:
            self.bl = kwargs['bl']
        else:
            self.bl = False
        
        # Swap-Mode
        if 'swap_mode' in kwargs:
            self.swap_mode = kwargs['swap_mode']
        else:
            self.swap_mode = 'default' 
        
        
        # swap == true --> swapping enabled 
        if self.swap == True:
            self.expected_num_swap = []
            if 'swap_base' in kwargs:
                self.swap_base = kwargs['swap_base']
            else:
                self.swap_base = 'all'
            
            if self.swap_base == 'difficulty':
                self.difficulty = list()
            
            if 'total_random' in kwargs:
                self.total_random = kwargs['total_random']
            else: 
                self.total_random = False

            if 'total_balance' in kwargs:
                self.total_balance = kwargs['total_balance']
            else: 
                self.total_balance = False

            # The fraction of samples that will be swapped out     
            if 'threshold' in kwargs:
                threshold =kwargs['threshold']
            else:
                threshold = [0.5]
            
            # Number of swappers, increase this number to reduce memory usage 
            if 'swap_workers' in kwargs:
                self.swap_num_workers = int(kwargs['swap_workers'])
            else:
                self.swap_num_workers = 1
                
            # [120,150,180,200] The number of epoch the blacklisting model is trained. Toggle this to control the smartness of the teacher 
            if 'p_epoch' in kwargs:
                self.p_epoch = kwargs['p_epoch']
            else:
                self.p_epoch = 150

            # The fraction of mispredicted samples that will be blacklisted     
            if 'bl_percent' in kwargs:
                self.bl_percent = kwargs['bl_percent']
            else:
                self.bl_percent = 1
            
            # ?? Xinyue Ma 
            if 'bl_percent_e' in kwargs:
                self.bl_percent_e = kwargs['bl_percent_e']
            else:
                self.bl_percent_e = 1

            # Partial blacklisting     
            if 'partial' in kwargs:
                self.partial = kwargs['partial']
            else:
                self.partial = 0
                
            if 'store_ratio' in kwargs:
                self.store_ratio = float(kwargs['store_ratio'])

                store_budget = int(self.replay_dataset.rb_size * self.store_ratio)
                self.store_budget = store_budget
            else:
                self.store_budget = None

        # MULTIPROCESSING MANAGER
        if (self.swap == True and self.swap_num_workers> 0) or self.cl_dataloader.num_workers > 0:
            print("[swap]: "+str(self.swap))

            if self.test_set == 'imagenet1000':
                self.manager = python_multiprocessing.Manager()
                print('MANAGER_1 PID:', self.manager._process.ident)
                self.replay_dataset.data = self.manager.list(self.replay_dataset.data)
                self.replay_dataset.targets = self.manager.list(self.replay_dataset.targets)
                self.replay_dataset.filenames = self.manager.list(self.replay_dataset.filenames)
            
        
        if self.swap == True:
            if 'saver' in kwargs: 
                saver = kwargs['saver']
            else: 
                saver = False
            if self.test_set == 'imagenet1000':
                self.swap_manager = SwapManager_ImageNet1k(self.replay_dataset, self.swap_num_workers, 
                                            self.swap_base, threshold=threshold, store_budget=self.store_budget, 
                                            filename=self.filename, result_save_path = self.result_save_path, get_entropy=self.get_train_entropy, seed=self.seed,dataset=self.test_set,saver=saver)
            else:
                self.swap_manager = SwapManager(self.replay_dataset.get_meta(), self.swap_num_workers, 
                                        self.swap_base, threshold=threshold, store_budget=self.store_budget, 
                                        filename=self.filename, result_save_path = self.result_save_path, get_entropy=self.get_train_entropy, seed=self.seed,dataset=self.test_set,swap_options=self.swap_options,saver=saver)            
        if 'save_tasks' in kwargs:
            self.save_tasks = kwargs['save_tasks']
        else: 
            self.save_tasks = False
        
        if 'st_size' in kwargs:
            self.st_size = int(kwargs['st_size'])
        else: self.st_size = 0


        if 'swap_period' in kwargs:
            self.swap_period = int(kwargs['swap_period'])
            if self.swap_period < 0 :
                self.swap_skip = True
                self.swap_period = -self.swap_period
            else: 
                self.swap_skip = False
        else: 
            self.swap_period = 1
            self.swap_skip = False
    def to_onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot
    
    def set_nomalize(self):
        if self.test_set in ["cifar10"]:
            self.nomalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2615])
        if self.test_set in ["cifar100"]:
            self.nomalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                std=[0.2009, 0.1984, 0.2023])
        elif self.test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            self.nomalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        elif self.test_set in ["tiny_imagenet"]:
            self.nomalize = transforms.Normalize(mean = [0.4802, 0.4480, 0.3975],
                                std=[0.2770, 0.2691, 0.2821])
    def get_entropy(self, outputs, targets):
        
        if self.get_test_entropy == False:
            return

        softmax = torch.nn.Softmax(dim=1)
        soft_output = softmax(outputs)
        entropy = torch.distributions.categorical.Categorical(probs=soft_output).entropy()
        #
        # if wrong predicted sample with low entropy, don't make it swap (make swap FALSE)
        #
        predicts = torch.max(outputs, dim=1)[1]
        r_predicted = (predicts.cpu() == targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        r_entropy = entropy[r_predicted]

        w_predicted = (predicts.cpu() != targets.cpu()).squeeze().nonzero(as_tuple=True)[0]
        w_entropy = entropy[w_predicted]
        
        return r_entropy.tolist(), w_entropy.tolist()

    def reset_opt(self, step=None):
        if self.test_set == "cifar100":
            #icarl setting
            self.opt = torch.optim.SGD(self.model.parameters(), lr=2.0, momentum=0.9, weight_decay=0.00001)
            
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    self.opt, [49,63], gamma=0.2
                                )
            if self.num_epochs > 80:
                lr_change_point = list(range(0,self.num_epochs,40))
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, lr_change_point, gamma=0.25)
            
        elif self.test_set in ["tiny_imagenet", "imagenet100", "imagenet1000"]:
            #icarl setting
            self.opt = torch.optim.SGD(self.model.parameters(), lr=2.0, momentum=0.9,  weight_decay=0.00001) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [20,30,40,50], gamma=0.20) #imagenet
        
        elif self.test_set == 'urbansound8k':
            self.opt = torch.optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.001)
            # self.opt = torch.optim.SGD(self.model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.001)
            # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #                         self.opt, [25,50,75], gamma=2
            #                     )
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    self.opt, [30,60], gamma=2
                                )
        elif self.test_set == 'audioset':
            # self.opt= torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=[0.9, 0.999],
            #                          weight_decay=0)
            self.opt= torch.optim.Adam(self.model.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-08,amsgrad=True,
                                     weight_decay=0)
            self.lr_scheduler = None
        elif self.test_set == "dailynsports":
            self.opt = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
            self.lr_scheduler = None
            
        else:
            raise f"No pre-set optimizer for {self.testset}"
    

    def before_train(self):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def after_train(self):
        raise NotImplementedError
