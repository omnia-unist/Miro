from math import inf
from torch.nn import functional as F
import torch
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import torchvision.transforms as T

import numpy as np
import copy

import os
from agents.base import Base
# from lib.graph import Graph
from _utils.sampling import multi_task_sample_update_to_RB
from lib import utils
import math
import gc
import time

import gc
import random

from dataset.validate import ValidateDataset

class SplitedStreamDataset(Dataset):
    def __init__(self, data, targets, ac_idx=None):
        self.data = data
        self.targets = targets
        self.actual_idx = ac_idx
        self.classes_in_dataset = set(targets)
        print("CLASSES IN DATASET : ", self.classes_in_dataset)
    def __len__(self):
        assert len(self.data) == len(self.targets)
        return self.data
    def get_sub_data(self, label):
        sub_data = []
        sub_label = []
        sub_idx = []
        for idx in range(len(self.data)):
            if self.targets[idx] == label:
                sub_data.append(self.data[idx])
                sub_label.append(label)
                sub_idx.append(self.actual_idx[idx])
        return sub_data, sub_label, sub_idx

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())


class BiC(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename, online_storage,**kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename,online_storage, **kwargs)
        
        #self.scaler = torch.cuda.amp.GradScaler() 
            
        if test_set in ["imagenet", "imagenet100", "imagenet1000"]:
            val_transform = transforms.Compose([
                                                    #transforms.Resize(256),
                                                    #transforms.CenterCrop(224),                                               
                                                                                                
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.ColorJitter(brightness=63/255),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    self.nomalize,
                                                ])

        elif test_set in ["cifar10", "cifar100", "tiny_imagenet", "mini_imagenet"]:
            val_transform = transforms.Compose([
                                            transforms.RandomCrop((32,32),padding=4),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            self.nomalize])
        
        else: val_transform = None

        self.classes_so_far = 0
        self.tasks_so_far = 0

        self.soft_incremental_top1_acc = list()
        self.soft_incremental_top5_acc = list()
        
        self.num_swap = list()
        self.bias_layers = list()

        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        #self.swap_manager.swap_loss = nn.CrossEntropyLoss(reduction="none")

        self.old_model = None
        #
        # add validation set
        #
        self.val_dataset = ValidateDataset(int(self.replay_dataset.rb_size * 0.1), val_transform, sampling="ringbuffer")
        
        self.replay_dataset.rb_size = int(self.replay_dataset.rb_size * 0.9)

        if 'distill' in kwargs:
            self.distill = kwargs['distill']
        else:
            self.distill = True
        print("======================== DISTILL : ",self.distill)

        
        self.how_long_stay = list()
        self.how_much_reused = list()

        self.num_files_per_label = dict()
        self.rb_size = self.replay_dataset.rb_size

        # Xinyue, config already set in Base.py
        if 'dynamic' in self.swap_options:
            self.dynamic = True
        else:
            self.dynamic = False
            
        if 'ckpt' in kwargs: #str
            self.ckpt = kwargs['ckpt'] #swap-in single
        else:
            self.ckpt = False
        
        # Xinyue 
        if 'ckpt_rand' in kwargs:
            self.ckpt_rand = kwargs['ckpt_rand']
        else:
            self.ckpt_rand = 0
            
        if 'p_epoch' in kwargs:
                self.p_epoch = kwargs['p_epoch']
        else:
            self.p_epoch = 150
            
        if 'bl_percent' in kwargs:
            self.bl_percent = kwargs['bl_percent']
        else:
            self.bl_percent = None
            
        if 'bl_percent_e' in kwargs:
            self.bl_percent_e = kwargs['bl_percent_e']
        else:
            self.bl_percent_e = 1
            
        if 'partial' in kwargs:
            self.partial = kwargs['partial']
        else:
            self.partial = 0
            
        # Drawing graph
        if 'graph' in kwargs:
            self.draw_graph = kwargs['graph']
        else:
            self.draw_graph = False

        if self.save_tasks:
            os.makedirs(f'{self.result_save_path}/checkpoints/', mode = 0o777, exist_ok = True)
        print("Online Swapping: ", self.swap_online)
        print(f'Swap mode: {self.swap_mode}')

        print(self.online_storage)
    def before_train(self, task_id):
        
        self.curr_task_iter_time = []
        self.curr_task_swap_iter_time = []


        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        # Add NEW TASK to the online storage 
        if self.get_history or (self.swap and self.swap_base == 'forgetting_based'):
            self.online_storage.append_task_dataset(self.stream_dataset)    
        # self.test_dataset.append_task_dataset(task_id)
        
        # self.cl_dataloader.update_loader()

        if(self.swap_mode != 'one_level'):
            print(f'USED DEFAULT update_loader')
            self.cl_dataloader.update_loader()
        else:
            self.cl_dataloader._update_loader(self.replay_dataset) 
        replay_classes = self.classes_so_far
        self.alpha = self.classes_so_far / (self.classes_so_far + len(self.stream_dataset.classes_in_dataset))
        
        self.classes_so_far += len(self.stream_dataset.classes_in_dataset)
        print("classes_so_far : ", self.classes_so_far)
        self.tasks_so_far += 1
        print("tasks_so_far : ", self.tasks_so_far)
        
        self.model.Incremental_learning(self.classes_so_far,self.device)

        #Xinyue Ma
        if(self.load_first_task == True and task_id==0):

            self.model.load_state_dict(torch.load(self.saved_model_name, map_location = self.device))
            # ASSUME 10-10 TASK_PARTITION
            self.tasks_so_far = int(self.classes_so_far/10)
            print(f'LOAD, change tasks_so_far to {self.tasks_so_far}')
            # load_rp_loc 
            self.rp_loc = utils.load_rp_from_file(self.saved_rp_loc_name,self.rb_size, self.replay_dataset,self.online_storage,self.stream_dataset)
            print(f'Replay_loaded from file, {len(self.replay_dataset)}',flush=True)
            # load_dataset history if needed 
            if self.swap and self.get_history:
                import pickle
                with open(self.saved_dataset_history,'rb') as handle:
                    self.online_storage.dataset_history = pickle.load(handle)
        # self.model.Incremental_learning(self.classes_so_far,self.device)
        
        if self.ckpt == True: #Swap-in Single
            self.pretrained_model.Incremental_learning(100)
            
        self.replay_size = len(self.replay_dataset)
        print("length of replay dataset : ", self.replay_size)
        
        self.stayed_iter = [0] * self.replay_size
        self.num_called = [0] * self.replay_size
        self.stream_losses, self.replay_losses = list(), list()

        #
        # add bias correction layer
        #
        bias_layer = BiasLayer().to(self.device)
        self.bias_layers.append(bias_layer)

        #
        # update validation set (val set should contain data of new classes before training)
        #

        if self.swap is True:
            self.swap_manager.before_train()
            #set threshold for swap_class_dist
            
            # Calculate swap threshold
            if  task_id == 0:
                self.swap_manager.swap_thr = inf
            else:
                batch_size = self.cl_dataloader.batch_size

                #self.swap_manager.swap_thr = int( self.num_epochs * self.swap_manager.threshold * (replay_size / replay_classes) ) + 1
                self.swap_manager.swap_thr = (batch_size * self.replay_size * self.swap_manager.threshold * self.num_epochs * 
                                            math.ceil((self.replay_size + len(self.stream_dataset))/batch_size) / (replay_classes * (self.replay_size + len(self.stream_dataset))) )
                
            print("SWAP dist threshold : ", (self.swap_manager.swap_thr))
            
            st = 0
            while True:
                en = st + 500
                self.swap_manager.saver.save(self.stream_dataset.data[st:en], self.stream_dataset.targets[st:en], self.stream_dataset.filename[st:en])
                print("SAVING ALL STREAM SAMPLES...WORKER")
                st = en
                if st > len(self.stream_dataset):
                    break



        eviction_list = multi_task_sample_update_to_RB(self.val_dataset, self.stream_dataset, True)
        print("\n\nLENGTH OF EVICTION LIST  : ", len(eviction_list))

        # """
        print(len(self.stream_dataset))
        for idx in sorted(eviction_list, reverse=True): 
            del self.stream_dataset.data[idx]
            del self.stream_dataset.targets[idx]
            del self.stream_dataset.filename[idx] 
        # """
        print(len(self.stream_dataset))
        print("DUPLICATED DATA IN STREAM IS EVICTED\n")
        
        print("VAL SET IS UPDATED")

        
        self.cl_val_dataloader = DataLoader(self.val_dataset, batch_size=128, pin_memory=True, shuffle=True)

        
        """
        if self.tasks_so_far <= 1 and (self.DP == True) and self.test_set in ["imagenet", "imagenet1000"]:
            self.model = nn.DataParallel(self.model, device_ids = self.gpu_order)
        """ 

        
        if self.old_model is not None:
            self.old_model.eval()
            self.old_model.to(self.device)
            """
            if (self.DP == True) and self.test_set in ["imagenet", "imagenet1000"]:
                self.old_model = nn.DataParallel(self.old_model, device_ids = self.gpu_order)
            """
            print("old model is available!")
            

    def after_train(self,task_id):
        if self.save_tasks == True :
            if(self.load_first_task==False or task_id!=0):
                print(f'SAVING TASK{task_id}',flush=True)
                f'{self.result_save_path}/checkpoints/'
                path = f'{self.result_save_path}/checkpoints/Task_{self.tasks_so_far}.pt'
                torch.save(self.model.state_dict(), path)
                if self.swap and self.get_history:
                    import pickle 
                    path = f'{self.result_save_path}/checkpoints/Task_{self.tasks_so_far}_DatasetHistory.pickle'
                    with open(path,'wb') as handle:
                        pickle.dump(self.online_storage.dataset_history, handle, protocol = pickle.HIGHEST_PROTOCOL)

        if self.ckpt == True: self.oracle_dataset.concat(self.stream_dataset.data, self.stream_dataset.targets, self.stream_dataset.filename)
                
        self.model.eval()
        if (self.get_history and self.draw_graph): 
            g = Graph()
            g.draw(filename = self.filename, save_path = self.result_save_path, model = self.model, 
                    storage = self.online_storage, replay = self.replay_dataset, rp_loc = self.rp_loc, 
                    tasks = self.tasks_so_far, classes = self.classes_so_far, device = self.device) # agents/graph.py
        
        if self.swap == True: self.swap_manager.after_train()
        
        # Add stream to replay, make sure each class has the same number of seats 
        if self.swap_mode == 'default':
            print(f'DEFAULT sample update to RB')
            multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)

        # Clear stream to take new class 
        self.num_files_per_label.update(self.stream_dataset.num_files_per_label)
        self.stream_dataset.clean_stream_dataset()

        gc.collect()

        #temp acc
        print("SOFTMAX")

        avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc = self.eval_task(get_entropy=self.get_test_entropy,num_tasks=task_id)
        
        print("task_accuracy : ", task_top1_acc)
        # if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
        print("task_top5_accuracy : ", task_top5_acc)
            
        print("class_accuracy : ", class_top1_acc)
        # if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
        print("class_top5_accuracy : ", class_top5_acc)

        print("current_accuracy : ", avg_top1_acc)
        # if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
        print("current_top5_accuracy : ", avg_top5_acc)

        self.soft_incremental_top1_acc.append(avg_top1_acc)
        self.soft_incremental_top5_acc.append(avg_top5_acc)

        print("incremental_top1_accuracy : ", self.soft_incremental_top1_acc)
        # if self.test_set in ["cifar100", "imagenet", "imagenet1000", "imagenet100"]:
        print("incremental_top5_accuracy : ", self.soft_incremental_top5_acc)


        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        import datetime 
        f.write(str(datetime.datetime.now())+'\n')        
        f.write("task_accuracy : "+str(task_top1_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("task_top5_accuracy : "+str(task_top5_acc)+"\n")

        f.write("class_accuracy : "+str(class_top1_acc)+"\n")
        if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
            f.write("class_top5_accuracy : "+str(class_top5_acc)+"\n")

        f.write("incremental_accuracy : "+str(self.soft_incremental_top1_acc)+"\n")
        # if self.test_set in ["imagenet", "imagenet1000", "imagenet100"]:
        f.write("incremental_top5_accuracy : "+str(self.soft_incremental_top5_acc)+"\n\n")
        f.close()

        
        f = open(self.result_save_path + 'time.txt','a')
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        self.avg_swap_iter_time.append(np.mean(np.array(self.curr_task_swap_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.write("avg_swap_iter_time : "+str(self.avg_swap_iter_time)+"\n")
        f.close()
        


        self.old_model=copy.deepcopy(self.model)


        gc.collect()
    
    
    def bias_forward(self, input, train=False):
        outputs_for_bias = list()
        min_class = 0

        for task_id in range(len(self.bias_layers)):
            
            max_class = max(self.data_manager.classes_per_task[task_id])
            inp_for_bias = input[:, min_class:max_class + 1]
            
            out_for_bias = self.bias_layers[task_id](inp_for_bias)

            outputs_for_bias.append(out_for_bias)
            min_class = max_class + 1
        
        return torch.cat(outputs_for_bias, dim=1)

    def train(self):
        
        if self.test_set == "cifar100":
            
            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [100,150,200], gamma=0.1)

            if self.num_epochs > 300:
                lr_change_point = list(range(100,self.num_epochs,50))
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, lr_change_point, gamma=0.2)
    
            
            self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [200,300,400], gamma=0.1)
            
            #self.bias_opt = torch.optim.Adam(params=self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.001)
            #self.bias_opt_scheduler = None

        else:
            print("IMAGENET OPTIMIZER...")

            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,  weight_decay=1e-4) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [30,60,80,90], gamma=0.1) #imagenet
            
            self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [30,60,80,90], gamma=0.1)


            """
            num_total_tasks = 10

            w_d = 1e-4 * (num_total_tasks/self.tasks_so_far)
            self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,  weight_decay=w_d) #imagenet
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [30,60,80,90], gamma=0.1) #imagenet
            
            self.bias_opt = optim.SGD(self.bias_layers[len(self.bias_layers)-1].parameters(), lr=0.1, momentum=0.9, weight_decay=w_d)
            self.bias_opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.bias_opt, [60,120,160,180], gamma=0.1)
            """

        if self.ckpt == True: trained = { f : 0 for f in self.oracle_dataset.filename } 
        
        # snapshot_st = tracemalloc.take_snapshot() #memory detector
        
        rand_nl, sele_nl = [], []
        inputs_batch, targets_batch, outputs_batch = [], [], []
        swapout_batch, losses_batch = [], []

        epoch_expected_num_swap = 0
        for epoch in range(self.num_epochs):
            if self.swap_mode == 'one_level' and self.tasks_so_far>1 and ((epoch+1)%1==0):
                # del self.replay_dataset

                if self.bl_percent is not None: 
                    if not self.dynamic:
                        bl=True
                    elif self.dynamic and epoch >= int(0.5*self.num_epochs):
                        bl = True 
                    else: 
                        bl = False
                else:
                    bl = False

                self.replay_dataset = self.online_storage.get_subset(self.rb_size,self.rb_size,bl)
                # self.replay_size = len(self.replay_dataset)
                self.cl_dataloader._update_loader(self.replay_dataset)
            else: 
                # print(f'one_level replay update not excuted')
                pass


            self.model.train()

            #stage 1
            print("BIAS LAYER PARAM!!!!")
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
                self.bias_layers[_].printParam(_)

            # time measure            
            iter_times = []
            swap_iter_times = []
            iter_st = None
            swap_st,swap_en = 0,0
            stream_loss, replay_loss = [],[]
            
            rand_nl_e, sele_nl_e = [], []
            # inputs_epoch, targets_epoch, outputs_epoch = [], [], []
            temp_targets_epoch, temp_outputs_epoch = [], []
            # losses_epoch = []
            
            newly_learned, newly_forgotten = {x:0 for x in range(self.classes_so_far)}, {x:0 for x in range(self.classes_so_far)}
            # End of profiling 
            if self.swap_skip: 
                condition = ((epoch+1)%self.swap_period!=0) 
            else: 
                condition = ((epoch+1)%self.swap_period==0) 
            if self.swap and condition: 
                epoch_expected_num_swap = 0 # reset per epoch 
            for i, (idxs, inputs, targets,filenames) in enumerate(self.cl_dataloader):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.model.train()
                outputs = self.model(inputs)
                outputs = self.bias_forward(outputs, train=False)    
                self.model.eval()
                iter_en = time.perf_counter()
                if i > 0 and iter_st is not None:
                    iter_time = iter_en - iter_st
                    print(f"EPOCH {epoch}, ITER {i}, ITER_TIME {iter_time} SWAP_TIME {swap_en-swap_st}...")
                    iter_times.append(iter_time)
                    if i % 10 == 0:
                        print(f"EPOCH {epoch}, ITER {i}, ITER_TIME {iter_time} SWAP_TIME {swap_en-swap_st}...")
                iter_st = time.perf_counter()        
                
                if self.swap == True and self.tasks_so_far > 1 and  condition and self.total_balance==False:
                    swap_iter_st = time.perf_counter()

                    # check size of the replay buffer 
                    # print(sys.getsizeof(self.replay_dataset.data[0])+sys.getsizeof(self.replay_dataset.targets[0]))
                    # curriculum epoch < added for early stage training
                    #if epoch >= 125 : # prof_ver. 50% gate, random
                    #if epoch < 125: # rev_prof_ver. 50% random, gate
                    #if epoch < 0: # only ver.
                    #if self.dynamic == True and epoch < (len(self.stream_dataset) * (self.tasks_so_far-1) / self.replay_size) * (1/self.swap_manager.threshold) * 5: # dynamic_ver. at least five epochs for all dataset
                    if self.ckpt == True: 
                        print("Swap-in: blacklisting")
                
                        filenames = {x : [] for x in range(self.classes_so_far)}
                        for f in self.oracle_dataset.filename:
                            filenames[int(f.split('_')[0])].append(f)
                        num_f, num_t = 0, 0

                        for t in swap_targets:
                            num_t += 1
                            file_list = filenames[t.item()]
                            while True:
                                random_file = random.choice(file_list) 
                                num_f += 1
                                if pre_rank[t.item()][random_file] != inf:
                                    break
                            swap_in_files.append(random_file)
                        rand_nl_e.append(num_t)
                        sele_nl_e.append(num_f)

                        for f in swap_in_files:
                            trained[f] +=1
                            
                        self.swap_manager.swap_pt(swap_idx.tolist(), swap_targets.tolist(), swap_in_files)
                    else:
                        print('Send to Swapper')
                        self.swap_manager.swap_pt(idxs.tolist(),targets.tolist(),filenames,data_ids=None)
                        
                    swap_en = time.perf_counter()
                    swap_iter_en = time.perf_counter()
                    swap_iter_time = swap_iter_en - swap_iter_st
                    swap_iter_times.append(swap_iter_time)
                # BIC_swap_img : imagenet + swap + no distill 
                #if self.swap == True and self.test_set in ["imagenet", "imagenet1000"]:
                #    print("THIS IS IMAGENET SWAP VER")
                #    loss = nn.CrossEntropyLoss()(outputs, targets)

                    
                if self.distill == False:
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    # losses_epoch.append(loss.reshape(-1))
                
                elif self.old_model is not None:
                    T = 2
                    with torch.no_grad():
                        old_outputs = self.old_model(inputs)
                        old_outputs = self.bias_forward(old_outputs, train=False)
                        old_task_size = old_outputs.shape[1]
                    
                        old_logits = old_outputs.detach()

                    hat_pai_k = F.softmax(old_logits/T, dim=1)
                    log_pai_k = F.log_softmax(outputs[..., :old_task_size]/T, dim=1)

                    loss_soft_target = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                    loss_hard_target = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
                    
                    """
                    old_outputs = F.softmax(old_outputs/T, dim=1)
                    old_task_size = old_outputs.shape[1]
                
                    log_outputs = F.log_softmax(outputs[..., :old_task_size]/T, dim=1)

                    loss_soft_target = -torch.mean(torch.sum(old_outputs * log_outputs, dim=1))
                    loss_hard_target = nn.CrossEntropyLoss()(outputs, targets)
                    """

                    
                    if self.get_loss == True:
                        get_loss = loss_hard_target.clone().detach()
                        replay_idxs = (idxs < self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        stream_idxs = (idxs >= self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        stream_loss.append(get_loss[stream_idxs].mean(-1).item())
                        if get_loss[replay_idxs].size(0) > 0:
                            replay_loss.append(get_loss[replay_idxs].mean(-1).item())
                        
                    loss_hard_target = loss_hard_target.mean()

                    if self.distill == False:
                        self.alpha = 0
                        #print(f"No distill ... alpha will be {self.alpha}")
                        
                    loss = (self.alpha * loss_soft_target) + ((1-self.alpha) * loss_hard_target)
                    #loss = (loss_soft_target * T * T) + ((1-self.alpha) * loss_hard_target)
                    # losses_epoch.append(loss.reshape(-1))
                    
                else:
                    loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
                    # losses_epoch.append(loss.reshape(-1))
                    #print(loss.shape)
                    if self.get_loss == True:
                        get_loss = loss.clone().detach()
                        #get_loss = loss_ext.view(loss_ext.size(0), -1)
                        #get_loss = loss_ext.mean(-1)
                        replay_idxs = (idxs < self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        stream_idxs = (idxs >= self.replay_size).squeeze().nonzero(as_tuple=True)[0]
                        #print(get_loss)
                        #print(stream_idxs)
                        stream_loss.append(get_loss[stream_idxs].mean(-1).item())
                        replay_loss.append(get_loss[replay_idxs].mean(-1).item())
                        
                    loss = loss.mean()
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
            # inputs_epoch.append(((torch.round(inputs.reshape(-1))).type('torch.CharTensor')).reshape(-1))
            # targets_epoch.append(targets.reshape(-1))
            # outputs_epoch.append(((torch.round(outputs.reshape(-1))).type('torch.CharTensor')).reshape(-1))  
            # losses_batch.append(torch.cat(losses_epoch).reshape(-1))
            
            # if self.swap == True and self.tasks_so_far > 1:
            #     # swapout_batch.append(torch.cat(swapout_epoch).reshape(-1))
            #     if self.ckpt:
            #         rand_nl.append(sum(rand_nl_e))
            #         sele_nl.append(sum(sele_nl_e))
            #0830 print every epoch 
            # history_counter = Counter(update_target.history.values())            
            # history_score_counter = Counter(self.forgetting_score.values())
            if self.swap and self.total_balance: 
                
                replay_len_per_cls =self.replay_dataset.len_per_cls
                for i in range(len(replay_len_per_cls)):
                    # idxs = [i*replay_len_per_cls[i] + x for x in random.sample(range(replay_len_per_cls[i]),int(replay_len_per_cls[i]*self.swap_manager.threshold))]
                    # targets = [i] * replay_len_per_cls[i]
                    # self.swap_manager.swap(idxs,targets)
                    _, sub_label,sub_filename,sub_index = self.replay_dataset.get_sub_data(i)
                    if i%2 == 0 : mode = 'default'
                    else: mode = 'other' 
                    print('Swap-out: ' + self.swap_base)    
                    swap_idx, swap_targets = self.swap_manager.swap_determine(torch.tensor(sub_index),None,torch.tensor(sub_label,device=self.device),mode=mode)
                    epoch_expected_num_swap += len(swap_idx)
                    # print(f'total balance swap, swapping {swap_idx}')
                    print("Swap-in: random")                        
                    self.swap_manager.swap(swap_idx.tolist(),swap_targets.tolist())                    
            # if (self.get_history):
            #     #load parameters
            #     #fgt_epoch = torch.reshape(self.online_storage.dataset_history.fgt, (-1,)).tolist()
            #     #fgt_score_epoch = torch.reshape(self.online_storage.dataset_history.fgt_score, (-1,)).tolist()
            #     #fgt_len_epoch = torch.reshape(self.online_storage.dataset_history.fgt_len, (-1,)).tolist()
            #     #fgt_output_epoch = torch.reshape(self.online_storage.dataset_history.fgt_output, (-1,)).tolist()
            #     #fgt_grad_epoch = torch.reshape(self.online_storage.dataset_history.fgt_grad, (-1,)).tolist()
            #     #fgt_ent_epoch = torch.reshape(self.online_storage.dataset_history.fgt_ent, (-1,)).tolist()
            #     fgt_epoch = self.online_storage.dataset_history.fgt.tolist()
            #     fgt_score_epoch = self.online_storage.dataset_history.fgt_score.tolist()
            #     fgt_len_epoch = self.online_storage.dataset_history.fgt_len.tolist()
            #     fgt_output_epoch = self.online_storage.dataset_history.fgt_output.tolist()
            #     fgt_grad_epoch = self.online_storage.dataset_history.fgt_grad.tolist()
            #     fgt_ent_epoch = self.online_storage.dataset_history.fgt_ent.tolist()
                
            #     if (epoch == 0): 
            #         fgt_task, fgt_score_task, fgt_len_task = [fgt_epoch], [fgt_score_epoch], [fgt_len_epoch]
            #         fgt_output_task, fgt_grad_task, fgt_ent_task = [fgt_output_epoch], [fgt_grad_epoch], [fgt_ent_epoch]
            #     else: 
            #         fgt_task, fgt_score_task, fgt_len_task = fgt_task + [fgt_epoch], fgt_score_task + [fgt_score_epoch], fgt_len_task + [fgt_len_epoch]
            #         fgt_output_task, fgt_grad_task, fgt_ent_task = fgt_output_task + [fgt_output_epoch], fgt_grad_task + [fgt_grad_epoch], fgt_ent_task + [fgt_ent_epoch]
            #     if (epoch==self.num_epochs-1):
            #         #text file
            #         f = open(self.result_save_path+ 'dataset_history.txt', 'a')
            #         if ((self.tasks_so_far%10) == 0 and epoch==self.num_epochs-1):
            #             f.write("Task " +str(self.tasks_so_far) +"\n")
            #             f.write("   "+str(self.online_storage.dataset_history.pred_his) +"\n")
            #         f.close()  

            #         if (epoch == self.num_epochs-1):
            #             f = open(self.result_save_path +'fgt.txt', 'a')
            #             f.write("Task " +str(self.tasks_so_far) +"\n")
            #             f.write(" FGT:\n\n  "+str(fgt_task) +"\n.\n")
            #             f.write(" Score:\n\n  "+str(fgt_score_task) +"\n.\n")
            #             f.write(" Length:\n\n  "+str(fgt_len_task) +"\n.\n")
            #             #f.write(" Output:\n\n  "+str(fgt_output_task) +"\n.\n")
            #             f.close()  
            # self.rp_loc = self.online_storage.dataset_history.update_rp_loc(self.replay_dataset)
            # os.makedirs(f'{self.result_save_path}rp_loc/Task{self.tasks_so_far}/',mode = 0o777, exist_ok = True)
            # utils.rp_to_csv([self.rp_loc,self.replay_dataset.targets[:],self.replay_dataset.filename[:]],f'{self.result_save_path}rp_loc/Task{self.tasks_so_far}',f'T{self.tasks_so_far}_epoch{epoch+1}')
                   
            # if self.ckpt == False:
            #     print(stream_loss, replay_loss)
            #     self.stream_losses.append(np.mean(np.array(stream_loss)))
            #     self.replay_losses.append(np.mean(np.array(replay_loss)))
            #     print(self.stream_losses, self.replay_losses)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #epoch_accuracy = self.eval(1)
            print("lr {}".format(self.opt.param_groups[0]['lr']))
            self.loss_item.append(loss.item())

            if epoch > 0:
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))
                self.curr_task_swap_iter_time.append(np.mean(np.array(swap_iter_times)))

            if self.swap == True and self.tasks_so_far > 1 and condition:
                self.expected_num_swap.append(epoch_expected_num_swap)
                actual_num_swap = self.swap_manager.get_num_swap()
                f = open(self.result_save_path + self.filename + '_swap_num.txt', 'a')
                f.write(f'task {self.tasks_so_far} epoch {epoch+1} expected_num_swap {self.expected_num_swap[-1]} actual_num_swap {actual_num_swap} {self.expected_num_swap[-1]==actual_num_swap}\n')
                f.close()
                f = open(self.result_save_path + self.filename + '_cls_swap_num.txt', 'a')
                f.write(f'task {self.tasks_so_far} epoch {epoch+1}\n cls_num_swap {self.swap_manager.get_cls_num_swap()} \n')
                f.close()
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss.item(),actual_num_swap))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            else:
                print("epoch {}, loss {}".format(epoch, loss.item()))
            
            """   
            #----------------------------------------------------------------
            if epoch % 50 == 0 and epoch > 0:
                
                avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc = self.eval_task(get_entropy=False, num_tasks=task_id)
                print("\n\n============ ACC - INTERMEDIATE ============")
                print("task_accuracy : ", task_top1_acc)
                print("task_top5_accuracy : ", task_top5_acc)
                print("-")
                print("class_accuracy : ", class_top1_acc)
                print("class_top5_accuracy : ", class_top5_acc)
                print("-")
                print("current_accuracy : ", avg_top1_acc)
                print("current_top5_accuracy : ", avg_top5_acc)
                print("============================================\n\n")
                
                curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
                print("\n\n============ ACC - INTERMEDIATE ============")
                print("soft_class_accuracy : ", class_accuracy)
                print("soft_task_accuracy : ", task_accuracy)
                print("soft_current_top1_accuracy : ", curr_top1_accuracy.item())
                print("soft_current_top5_accuracy : ", curr_top5_accuracy.item())
                print("============================================\n\n")
            """
            #self.num_swap.append(self.swap_manager.get_num_swap())
            #self.swap_manager.reset_num_swap()
            
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #stage 2
        if self.old_model is not None:

            print("Training bias layers....")

            print("\nBEFORE BIAS TRAINING : BIAS LAYER PARAM!!!!")

            
            for _ in range(len(self.bias_layers)-1):
                self.bias_layers[_].eval()
                #self.bias_layers[_].train()
                self.bias_layers[_].printParam(_)
            
            self.bias_layers[len(self.bias_layers)-1].train()
            self.bias_layers[len(self.bias_layers)-1].printParam(len(self.bias_layers)-1)
            print("\n")
            
            #for epoch in range(2 * self.num_epochs):
            for epoch in range(self.num_epochs):    
                print(f"Training bias layers....epoch {epoch}")

                for i, (idxs, inputs, targets) in enumerate(self.cl_val_dataloader):
                    self.model.eval()

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    outputs = self.bias_forward(outputs, train=True)

                    loss = self.criterion(outputs, targets)
                    self.bias_opt.zero_grad()
                    loss.backward()
                    self.bias_opt.step()
                if self.bias_opt_scheduler is not None:
                    self.bias_opt_scheduler.step()
                print("lr {}".format(self.bias_opt.param_groups[0]['lr']))
                """
                if epoch % 100 == 0 and epoch > 0:
                    avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc = self.eval_task(get_entropy=False, num_tasks=task_id)
                    print("\n\n============ ACC - INTERMEDIATE ============")
                    print("task_accuracy : ", task_top1_acc)
                    print("task_top5_accuracy : ", task_top5_acc)
                    print("-")
                    print("class_accuracy : ", class_top1_acc)
                    print("class_top5_accuracy : ", class_top5_acc)
                    print("-")
                    print("current_accuracy : ", avg_top1_acc)
                    print("current_top5_accuracy : ", avg_top5_acc)
                    print("============================================\n\n")
                        
                    curr_top1_accuracy, curr_top5_accuracy, task_accuracy, class_accuracy = self.eval(1)    
                    print("\n\n============ ACC - INTERMEDIATE ============")
                    print("class_accuracy : ", class_accuracy)
                    print("task_accuracy : ", task_accuracy)
                    print("current_top1_accuracy : ", curr_top1_accuracy.item())
                    print("current_top5_accuracy : ", curr_top5_accuracy.item())
                    print("============================================\n\n")
                    
                    for _ in range(len(self.bias_layers)-1):
                        self.bias_layers[_].eval()
                        #self.bias_layers[_].train()
                        self.bias_layers[_].printParam(_)
                    
                    self.bias_layers[len(self.bias_layers)-1].train()
                    self.bias_layers[len(self.bias_layers)-1].printParam(len(self.bias_layers)-1)
                    print("\n")
                """

            
            print("\nAFTER BIAS TRAINING : BIAS LAYER PARAM!!!!")
            for _ in range(len(self.bias_layers)-1):
                self.bias_layers[_].printParam(_)
            self.bias_layers[len(self.bias_layers)-1].printParam(len(self.bias_layers)-1)
            print("\n")
            if self.swap == True:
                print("epoch {}, loss {}, num_swap {}".format(epoch, loss.item(), self.swap_manager.get_num_swap()))
                self.num_swap.append(self.swap_manager.get_num_swap())
                self.swap_manager.reset_num_swap()
            else:
                print("epoch {}, loss {}".format(epoch, loss.item()))
   
    def eval_task(self, get_entropy=False,update_history=True,num_tasks=None):
        if num_tasks == None: 
            num_tasks = self.tasks_so_far
        self.model.eval()
        
        for _ in range(len(self.bias_layers)):
           self.bias_layers[_].eval()
        
        avg_top1_acc, task_top1_acc, class_top1_acc = {},{},{}
        avg_top5_acc, task_top5_acc, class_top5_acc = {},{},{}
        task_size =len(self.stream_dataset.classes_in_dataset)
        for task in range(num_tasks+1):
            self.test_dataset.append_task_dataset(task)
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        ypreds, ytrue = self.compute_accuracy(test_dataloader, get_entropy)    
        avg_top1_acc, task_top1_acc, class_top1_acc = self.accuracy_per_task(ypreds, ytrue, task_size=task_size, class_size=task_size, topk=1)
        avg_top5_acc, task_top5_acc, class_top5_acc = self.accuracy_per_task(ypreds, ytrue, task_size=task_size, class_size=task_size, topk=5)
        
        #self.test_dataset.clean_task_dataset()
        del test_dataloader
        gc.collect()
        return avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc
        
    
    def compute_accuracy(self, loader, get_entropy=False):
        ypred, ytrue = [], []

        if self.swap==True and get_entropy == True:
            w_entropy_test = []
            r_entropy_test = []

            logits_list = []
            labels_list = []

        for i, (vecs, labels) in enumerate(loader):
            vecs = vecs.to(self.device)
            with torch.no_grad():
                outputs = self.model(vecs)
                outputs = self.bias_forward(outputs)
            outputs = outputs.detach()
            #get entropy of testset
            if self.swap==True and get_entropy == True:
                r, w = self.get_entropy(outputs, labels)
                r_entropy_test.extend(r)
                w_entropy_test.extend(w)
                    
                logits_list.append(outputs)
                labels_list.append(labels)


            ytrue.append(labels.numpy())
            ypred.append(torch.softmax(outputs, dim=1).cpu().numpy())

        if self.swap==True and get_entropy == True:
            print("RECORD TEST ENTROPY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            f = open(self.result_save_path +  '_correct_test_entropy.txt', 'a')
            f.write(str(r_entropy_test)+"\n")
            f.close()
            
            f = open(self.result_save_path + '_wrong_test_entropy.txt', 'a')
            f.write(str(r_entropy_test)+"\n")
            f.close()

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

            ece = self.ece_loss(logits, labels).item()
            f = open(self.result_save_path +  '_ece_test.txt', 'a')
            f.write(str(ece)+"\n")
            f.close()

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
        all_acc = {}

        avg_acc = self.accuracy(ypreds, ytrue, topk=topk) * 100
        
        task_acc = {}
        class_acc = {}

        if task_size is not None:
            for task_id, class_id in enumerate(range(0, np.max(ytrue) + task_size, task_size)):
                if class_id > np.max(ytrue):
                    break
                idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

                # label = "{}-{}".format(
                #     str(class_id).rjust(2, "0"),
                #     str(class_id + task_size - 1).rjust(2, "0")
                # )
                #all_acc[label] = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk)
                task_acc[task_id] = self.accuracy(ypreds[idxes], ytrue[idxes], topk=topk) * 100

                for class_idx in range(class_id, class_id + class_size):
                    idxes_c = np.where(ytrue == class_idx)[0]
                    class_acc[class_idx] = self.accuracy(ypreds[idxes_c], ytrue[idxes_c], topk=topk) * 100

        return avg_acc, task_acc, class_acc                

    def accuracy(self,output, targets, topk=1):
        """Computes the precision@k for the specified values of k"""
        output, targets = torch.tensor(output), torch.tensor(targets)

        batch_size = targets.shape[0]
        if batch_size == 0:
            return 0.
        nb_classes = len(np.unique(targets))
        topk = min(topk, nb_classes)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].reshape(-1).float().sum(0).item()
        return round(correct_k / batch_size, 4)

    """
    def eval(self, mode=1):
        
        self.model.eval()
        
        test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
        
        if mode==0:
            print("compute NMS")
        self.model.eval()
        for _ in range(len(self.bias_layers)):
           self.bias_layers[_].eval()
        
        correct, total = 0, 0
        class_correct = list(0. for i in range(self.classes_so_far))
        class_total = list(0. for i in range(self.classes_so_far))
        class_accuracy = list()
        top5_accuracy = list()
        task_accuracy = dict()

        for setp, ( imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
                outputs = self.bias_forward(outputs)

            #top5 acc
            top5_acc = self.top5_acc(outputs, labels)
            top5_accuracy.append(top5_acc.item())

            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            c = (predicts.cpu() == labels.cpu()).squeeze()

            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
        for i in range(len(class_correct)):
            if class_correct[i]==0 and class_total[i] == 0:
                continue
            class_acc = 100 * class_correct[i] / class_total[i]
            print('[%2d] Accuracy of %2d : %2d %%' % (
            i, i, class_acc))
            class_accuracy.append(class_acc)
        
        for task_id, task_classes in self.data_manager.classes_per_task.items():
            task_acc = np.mean(np.array(list(map(lambda x : class_accuracy[x] ,task_classes))))
            task_accuracy[task_id] = task_acc

        total_top1_accuracy = 100 * correct / total
        total_top5_accuracy = np.mean(np.array(top5_accuracy))
        
        self.model.train()
        return total_top1_accuracy, total_top5_accuracy, task_accuracy, class_accuracy
    
    def top5_acc(self, output, target):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
            res = correct_k.mul_(100.0 / batch_size)
            
            return res

    def classify(self, test_image):
        result = []
        test_image = F.normalize(self.model.feature_extractor(test_image).detach()).cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        #print(class_mean_set)

        for target in test_image:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
    """