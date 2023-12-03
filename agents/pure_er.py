import torch
import torch.nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import datetime 
import gc
import numpy as np 
import os
import random
import time
from math import inf

# internal imports  
from optim.optimizer import Optimizer, set_opt_for_profiler
from agents.base import Base
from _utils.sampling import multi_task_sample_update_to_RB
from lib.swap_manager import fetch_from_storage
import power_check as pc # For Jetson experiments

# Debugging 
torch.set_printoptions(threshold=30_000,linewidth=30_000) # more details in printed tensor
DEBUG = False # Enabling some time and memory profiling 


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

class PureER(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename,  **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)
      
        # Statististics of observed data
        self.classes_so_far = 0                         # Number of classes in the model
        self.classes_seen=[]                            # List of classes which are model have seen
        self.tasks_so_far = 0                           # Number of tasks in the model 
        self.samples_seen = 0
        self.soft_incremental_top1_acc = list()         # Top 1 accuracy
        self.soft_incremental_top5_acc = list()         # Top 5 accuracy
        self.num_files_per_label = dict()
        
        self.rb_size = self.replay_dataset.rb_size

        self.checkpoint_path = self.base_ckpt_path
        self.ckpt_model_exist = False
            
        if 'manual_config' in kwargs:
            self.manual_config = kwargs['manual_config']
        else:
            self.manual_config = None
        
        if 'optimizer' in kwargs: 
            kwargs['optimizer']['log_file'] = f'{self.result_save_path}_{self.filename}_optimizer.csv'
            kwargs['optimizer']['test_set'] = self.test_set
            if DEBUG: 
                kwargs['optimizer']['time_log_file'] = self.time_log_file
            self.optimizer = Optimizer(self.observe,kwargs['optimizer'],self.device,self.optimizer_create_ckpt)
        else: self.optimizer=None

        if 'jetson' in kwargs and kwargs['jetson'] is True: 
            self.jetson = True
            self.power_log = kwargs['power_log']
        else: 
            self.jetson = False 
            self.power_log = None
        
        self.train_iter = 0
        self.overhead = 0        
    
    def optimizer_create_ckpt(self, trail_duration=30,**kwargs):
        raise NotImplementedError
    def observe(self,config:tuple, ckpt_model=None, ckpt_opt=None, ckpt_lr_scheduler=None, trail_duration=None,**kwargs):
        raise NotImplementedError
    def before_train(self, task_id):
        raise NotImplementedError
    def after_train(self,task_id):
        raise NotImplementedError
    def train(self):     
        raise NotImplementedError
    def eval_task(self,num_tasks=None,model=None):
        if num_tasks == None: 
            num_tasks = self.tasks_so_far
        avg_top1_acc, task_top1_acc, class_top1_acc = {},{},{}
        avg_top5_acc, task_top5_acc, class_top5_acc = {},{},{}
        task_size =len(self.stream_dataset.classes_in_dataset)
        self.test_dataset.append_task_dataset(num_tasks-1)
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        ypreds, ytrue = self.compute_accuracy(test_dataloader,model=model)    
        avg_top1_acc, task_top1_acc, class_top1_acc = self.accuracy_per_task(ypreds, ytrue, task_size=task_size, class_size=task_size, topk=1)
        if self.classes_so_far>=5:
            avg_top5_acc, task_top5_acc, class_top5_acc = self.accuracy_per_task(ypreds, ytrue, task_size=task_size, class_size=task_size, topk=5)
        
        del test_dataloader
        gc.collect()
        return avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc
        
    def compute_accuracy(self, loader, model=None):
        ypred, ytrue = [], []
        if model == None:
            self.model.eval()
        else: model.eval()
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                if model == None: 
                    outputs = self.model(imgs) 
                else: outputs = model(imgs)
                outputs = outputs.clone().detach()

                ytrue.append(labels.numpy())
                ypred.append(torch.softmax(outputs, dim=1).cpu().numpy())
        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)
        return ypred, ytrue

    def compute_task_accuracy(self,ypreds,ytrue,task_size,topk=1,):

        class_acc = {}
        class_st = np.min(ytrue)
        for class_idx in range(class_st, class_st + task_size):
            idxes_c = np.where(ytrue == class_idx)[0]
            class_acc[class_idx] = self.accuracy(ypreds[idxes_c], ytrue[idxes_c], topk=topk) *100
        idxes = np.where(np.logical_and(ytrue >= class_st, ytrue < class_st + task_size))[0]
        task_acc = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk) * 100
        return task_acc, class_acc

    def accuracy_per_task(self, ypreds, ytrue, task_size=10, class_size=10, topk=1):
        """Computes accuracy for the whole test & per task.
        :param ypred: The predictions array.
        :param ytrue: The ground-truth array.
        :param task_size: The size of the task.
        :return: A dictionnary.
        """
        avg_acc = self.accuracy(ypreds, ytrue, topk=topk) * 100
        
        task_acc = {}
        class_acc = {}
        if task_size is not None:
            for task_id, class_id in enumerate(range(0, np.max(ytrue) + task_size, task_size)):
                if class_id > np.max(ytrue):
                    break
                idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]
                task_acc[task_id] = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk) * 100
                
                for class_idx in range(class_id, class_id + class_size):
                    idxes_c = np.where(ytrue == class_idx)[0]
                    class_acc[class_idx] = self.accuracy(ypreds[idxes_c], ytrue[idxes_c], topk=topk) * 100

        return avg_acc, task_acc, class_acc

    def accuracy_per_cls(self,ypreds,ytrue,topk=1):
        avg_acc = self.accuracy(ypreds, ytrue, topk=topk) * 100
        class_acc = {}
        for class_idx in self.classes_seen: 
            idxes_c = np.where(ytrue == class_idx)[0]
            class_acc[class_idx] = self.accuracy(ypreds[idxes_c], ytrue[idxes_c], topk=topk) * 100
            
        return avg_acc, class_acc
    def accuracy(self,output, targets, topk=1):
        """Computes the precision@k for the specified values of k"""
        output, targets = torch.tensor(output), torch.tensor(targets)

        batch_size = targets.shape[0]
        if batch_size == 0:
            return 0.
        nb_classes = len(np.unique(targets))
        if nb_classes != 1: 
            topk = min(topk, nb_classes)
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].reshape(-1).float().sum(0).item()
        return round(correct_k / batch_size, 4)
    
