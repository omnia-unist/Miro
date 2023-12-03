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
from agents.pure_er import PureER
from _utils.sampling import multi_task_sample_update_to_RB
from lib.swap_manager import fetch_from_storage
import power_check as pc # For Jetson experiments

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

class PureER_TINY(PureER):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename,  **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)

        if 'optimizer' in kwargs: 
            kwargs['optimizer']['log_file'] = f'{self.result_save_path}_{self.filename}_optimizer.csv'
            kwargs['optimizer']['test_set'] = self.test_set
            self.optimizer = Optimizer(self.observe,kwargs['optimizer'],self.device,self.optimizer_create_ckpt)
            print('[Miro Profiler] Enabled')
        else: self.optimizer=None

    def optimizer_create_ckpt(self, trail_duration=30,**kwargs):
        if 'pretrain' in kwargs: 
            pretrain = kwargs['pretrain']
        else: pretrain = self.load_from_history
        for param in self.model.parameters(): 
            param.requires_grad = False
        self.ckpt_model.Incremental_learning(self.classes_so_far-10,self.device)
        self.ckpt_model.load_state_dict(torch.load(f'{self.checkpoint_path}/Task_{self.tasks_so_far-1}.pt',map_location = self.device))
        self.ckpt_model.Incremental_learning(self.classes_so_far,self.device)
        if trail_duration == self.num_epochs: 
            ckpt_path =  f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt'
        else: 
            ckpt_path = f'{self.checkpoint_path}/Task_{self.tasks_so_far}_additional_ckpt_epoch{trail_duration}.pt'

        self.ckpt_model.to(self.device)
        self.cl_dataloader.update_loader()
        log_path = f'{self.result_save_path}/{self.filename}_optimizer.txt'
        f = open(log_path, 'a')
        f.write(f'Task {self.tasks_so_far} CHECKPOINT\n')
        f.write(f'train set size: {len(self.stream_dataset) + len(self.replay_dataset)}\n')
        
        for i, (_, param) in enumerate(self.ckpt_model.named_parameters()):
            param.requires_grad=True
        ckpt_opt, ckpt_lr_scheduler =  set_opt_for_profiler(self.test_set,self.ckpt_model,trail_duration)
        if self.swap:
            self.swap_manager.resume()
            self.swap_manager.update_meta(self.replay_dataset.get_meta())
            
        for epoch in range(trail_duration):  
            for i, (idxs, inputs, targets,filenames) in enumerate(self.cl_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.ckpt_model.train()
                outputs = self.ckpt_model(inputs)
                if self.swap and self.tasks_so_far >1: 
                    self.swap_manager.swap_pt(idxs.tolist(),targets.tolist(),filenames,data_ids=None)
                targets = self.to_onehot(targets,self.classes_so_far).to(self.device)
                loss_value = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
                loss_value = loss_value.mean()
                ckpt_opt.zero_grad()
                loss_value.backward()
                ckpt_opt.step()
            self.overhead += i+1
            if ckpt_lr_scheduler is not None:
                ckpt_lr_scheduler.step()
            f.write("lr {}".format(ckpt_opt.param_groups[0]['lr']))
            f.write("epoch {}, loss {}\n".format(epoch, loss_value.item()))
            f.flush()
        if self.swap:
            self.swap_manager.pause()

        avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc= self.eval_task(num_tasks=self.tasks_so_far,model=self.ckpt_model)
        print(f'T{self.tasks_so_far} Profilier Checkpoint, Accuracy:{avg_top1_acc}')
        self.ckpt_model_exist = True
        os.makedirs(f'{self.checkpoint_path}/',mode = 0o777, exist_ok = True)
        torch.save(self.ckpt_model.state_dict(), ckpt_path)
        return [avg_top1_acc, class_top1_acc], ckpt_path
    
    def observe(self,config:tuple, ckpt_model=None, ckpt_opt=None, ckpt_lr_scheduler=None, trail_duration=None,**kwargs):
        if 'pretrain' in kwargs: 
            pretrain = kwargs['pretrain']
        else: pretrain = self.load_from_history

        if config[0] > self.samples_seen:  
            print('EM larger than needed')
            return {'acc':0, 'energy':0,'energy_estim':0,'cls_acc':{0:0}}
        for param in self.model.parameters(): 
            param.requires_grad = False

        if trail_duration == self.num_epochs and not self.optimizer.use_ckpt and len(self.optimizer.layer_freeze) == 0 and self.optimizer.data_ratio ==1: 
            ckpt_path = f'{self.checkpoint_path}/{config[0]}_{config[1]}/Task_{self.tasks_so_far}.pt'
            self.return_path = f'{self.checkpoint_path}/Task_{self.tasks_so_far-1}_ckpt_ver.pt'
        else: 
            if not self.optimizer.use_ckpt: 
                ckpt_size = 'ckpt0'
                self.return_path = f'{self.checkpoint_path}/Task_{self.tasks_so_far-1}_ckpt_ver.pt'
            else: 
                ckpt_size = f'ckpt{self.optimizer.ckpt_size}'
                if not self.ckpt_model_exist: self.return_path = None
            ckpt_path = f'{self.checkpoint_path}/{config[0]}_{config[1]}/Task_{self.tasks_so_far}_ckpt{ckpt_size}_epoch{trail_duration}_dr{self.optimizer.data_ratio}.pt'

        if  os.path.exists(ckpt_path) and pretrain==True and self.load_from_history:
            ckpt_model = copy.deepcopy(self.model)
            ckpt_model.load_state_dict(torch.load(ckpt_path, map_location = self.device))
            print(f'PROFILE LOAD FROM {ckpt_path}')
        else:
            if self.optimizer.use_ckpt:
                if self.ckpt_model_exist == False: 
                    self.stream_dataset.resize(int(config[1]*self.optimizer.data_ratio))               
                    _,self.return_path = self.optimizer_create_ckpt(self.optimizer.ckpt_size,pretrain=False)
            else: 
                self.ckpt_model=copy.deepcopy(self.model)
                self.ckpt_model.Incremental_learning(self.classes_so_far,self.device)
                self.return_path = f'{self.checkpoint_path}/Task_{self.tasks_so_far-1}_ckpt_ver.pt'
                torch.save(self.ckpt_model.state_dict(), self.return_path)

            ckpt_model = copy.deepcopy(self.ckpt_model)
            ckpt_model.load_state_dict(torch.load(self.return_path, map_location = self.device))
            ckpt_model.Incremental_learning(self.classes_so_far,self.device)
            for param in self.ckpt_model.parameters(): 
                param.requires_grad = False
            res = self.replay_dataset.shm_resize(int(config[0]*self.optimizer.data_ratio))
            if res in [1,2]: # need to fill replay from files 
                memory_per_cls = min(int(self.replay_dataset.rb_size / len(self.replay_dataset.offset)), int(self.samples_seen / len(self.replay_dataset.offset)))
                fetch_targets = []
                for i in self.replay_dataset.offset: 
                    if self.swap and self.swap_manager.saver:
                        n_samples = min(memory_per_cls,self.swap_manager.saver.get_num_file_for_label_for_swap(i))
                        fetch_targets.extend([i for k in range(n_samples)])
                        self.replay_dataset.len_per_cls[i]=n_samples
                    else: 
                        fetch_targets.extend([i for k in range(memory_per_cls)])
                for i in range(1, len(self.replay_dataset.offset)): self.replay_dataset.offset[i] = self.replay_dataset.offset[i-1] + self.replay_dataset.len_per_cls[i-1]
                fetch_idx = [i for i in range(len(fetch_targets))]
                fetch_files = []
                for label in range(len(self.replay_dataset.offset)):
                    if self.swap_manager.saver:
                        label_list= [f'{label}_{count+1}'  for count in range(min(memory_per_cls,self.swap_manager.saver.get_num_file_for_label_for_swap(label))) ]
                    else: label_list= [f'{label}_{count+1}'  for count in range(memory_per_cls)]
                    fetch_files.extend(label_list)
                for j in range(len(fetch_targets)): self.replay_dataset.targets[j] = fetch_targets[j]
                fetch_from_storage(self.replay_dataset, fetch_idx,fetch_targets,self.task_id, filenames=fetch_files, testset=self.test_set)
                self.replay_dataset.filled[0] = len(fetch_idx)
            self.stream_dataset.resize(int(config[1]*self.optimizer.data_ratio)) 
            ckpt_model.to(self.device)
            self.cl_dataloader.update_loader()
            log_path = f'{self.result_save_path}/{self.filename}_optimizer.txt'
            f = open(log_path, 'a')
            f.write(f'Task {self.tasks_so_far}\n')
            f.write(f'stream_size: {len(self.stream_dataset)}   replay_size: {len(self.replay_dataset)}\n')
            f.write(f'train set size: {len(self.stream_dataset) + len(self.replay_dataset)}\n')
            
            for _, (_, param) in enumerate(ckpt_model.named_parameters()):
                param.requires_grad=True
            ckpt_opt, ckpt_lr_scheduler =  set_opt_for_profiler(self.test_set,ckpt_model,trail_duration)
            if self.swap:
                self.swap_manager.resume()
                self.swap_manager.update_meta(self.replay_dataset.get_meta())
            for epoch in range(trail_duration):  
                for i, (idxs, inputs, targets,filenames) in enumerate(self.cl_dataloader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    ckpt_model.train()
                    outputs = ckpt_model(inputs)
                    if self.swap and self.tasks_so_far >1: 
                        self.swap_manager.swap_pt(idxs.tolist(),targets.tolist(),filenames,data_ids=None)
                    targets = self.to_onehot(targets,self.classes_so_far).to(self.device)
                    loss_value = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
                    loss_value = loss_value.mean()
                    ckpt_opt.zero_grad()
                    loss_value.backward()
                    ckpt_opt.step()
                self.overhead += i+1 
                if ckpt_lr_scheduler is not None:
                    ckpt_lr_scheduler.step()
                f.write("lr {}".format(ckpt_opt.param_groups[0]['lr']))
                f.write("epoch {}, loss {}\n".format(epoch, loss_value.item()))
                f.flush()

            os.makedirs(f'{self.checkpoint_path}/{config[0]}_{config[1]}/',mode = 0o777, exist_ok = True)
          
            # # saving mini-checkpoints
            # if not os.path.exists(ckpt_path) or pretrain==False and self.save_ckpts: 
            #     os.makedirs(f'{self.checkpoint_path}/{config[0]}_{config[1]}/',mode = 0o777, exist_ok = True)
            #     torch.save(ckpt_model.state_dict(), ckpt_path)
            #     print(f'Saved to {ckpt_path} ')
            if self.swap:
                self.swap_manager.pause()

        energy = 0
        energy_estim = config[0] + config[1]
        if os.path.exists(ckpt_path): ckpt_model.load_state_dict(torch.load(ckpt_path, map_location = self.device))
        avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc= self.eval_task(num_tasks=self.tasks_so_far,model=ckpt_model)
        res = {'acc':avg_top5_acc, 'energy':energy,'energy_estim':energy_estim,'cls_acc':class_top5_acc}
        return res
    def before_train(self, task_id):
        self.task_id = task_id
        self.curr_task_iter_time = []
        self.curr_task_swap_iter_time = []
 
        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        
        # Upload data loader for training
        self.cl_dataloader._update_loader(self.replay_dataset) 
        # Update meta-data        
        self.classes_seen = self.classes_seen + [e for e in self.stream_dataset.classes_in_dataset if e not in self.classes_seen]
        self.classes_so_far = len(self.classes_seen)
        print("Classes in Storage : ", self.classes_seen)
        self.tasks_so_far = self.task_id +1
        print("Tasks So Far : ", self.tasks_so_far)
        self.samples_in_stream = len(self.stream_dataset)
        self.model.Incremental_learning(self.classes_so_far,self.device)
        self.replay_size = len(self.replay_dataset)
        print("EM Size : ", self.replay_size)
        self.stream_losses, self.replay_losses = list(), list()
        # Prepare swap manager
        if self.swap is True:
            self.swap_manager.before_train(self.replay_dataset.get_meta(),self.task_id)
            if self.swap_manager.saver: 
                st = 0
                print("SAVING ALL STREAM SAMPLES...WORKER")
                while True:
                    en = st + 500
                    self.swap_manager.saver.save(self.stream_dataset.data[st:en], self.stream_dataset.targets[st:en], self.stream_dataset.filename[st:en])
                    st = en
                    if st > len(self.stream_dataset):
                        break
        if self.optimizer:
            self.return_path=None
            if self.swap and self.tasks_so_far>1:
                self.swap_manager.pause()
            if self.tasks_so_far<self.optimizer.start_point  :# min(len(self.manual_config['rb_size']),self.optimizer.start_point-1):
                best_config = {'rb_size':self.rb_size,'st_size':self.st_size}
            else:
                base_config = {config:getattr(self,config) for config in self.optimizer.get_params()}
                self.optimizer.set_base_config(base_config)
                if self.jetson: 
                    self.power_log.recordEvent(name=f'Mini-profiler, start, T{self.tasks_so_far}')
                print(f'\n----Finding Best Config for Task {self.tasks_so_far}----')
                best_config, _ = self.optimizer.find_best_config(self.tasks_so_far)
                if self.jetson: 
                    self.power_log.recordEvent(name=f'Mini-profiler, end, T{self.tasks_so_far}')
                self.ckpt_model_exist=False
            for config in best_config: 
                if config == 'rb_size': 
                    self.rb_size = best_config['rb_size']
                    if self.replay_dataset.shm_resize(best_config['rb_size'],delete=True) in [1,2]:
                        memory_per_cls = min(int(self.replay_dataset.rb_size / len(self.replay_dataset.offset)), int(self.samples_seen / len(self.replay_dataset.offset)))
                        swap_targets = []
                        for i in self.replay_dataset.offset: swap_targets.extend([i for j in range(memory_per_cls)])
                        swap_idx = [i for i in range(len(swap_targets))]
                        for j in range(len(swap_targets)): self.replay_dataset.targets[j] = swap_targets[j]
                        swap_in_files = []
                        for label in range(len(self.replay_dataset.offset)):
                            if self.swap_manager.saver:
                                label_list= [f'{label}_{count+1}'  for count in range(min(memory_per_cls,self.swap_manager.saver.get_num_file_for_label_for_swap(label))) ]
                            else: label_list= [f'{label}_{count+1}'  for count in range(memory_per_cls)]
                            swap_in_files.extend(label_list)
                        self.replay_dataset.filled[0] = len(swap_idx)
                        fetch_from_storage(self.replay_dataset, swap_idx,swap_targets,self.task_id,filenames=swap_in_files,testset=self.test_set)
                elif config == 'st_size': 
                    self.stream_dataset.resize(best_config['st_size'],evict=True)
                if hasattr(self, config): 
                    setattr(self,config,best_config[config])
            if self.swap and self.tasks_so_far>1:
                self.swap_manager.resume()
                self.swap_manager.after_train()
                self.swap_manager.before_train(self.replay_dataset.get_meta(),self.task_id)
        if not self.optimizer and self.st_size != len(self.stream_dataset.data):
            self.stream_dataset.resize(self.st_size,evict=True)
        if self.manual_config and self.tasks_so_far <= len(self.manual_config['rb_size']):# and self.optimizer:
            self.checkpoint_path += f'{self.manual_config["rb_size"][self.tasks_so_far-1]}_{self.manual_config["st_size"][self.tasks_so_far-1]}/'
        else:
            self.checkpoint_path += f'{self.rb_size}_{self.st_size}/'
        self.model.eval()

    def after_train(self,task_id):
        self.model.eval()
        if self.swap == True: 
            swap_finished, num_swapped= self.swap_manager.after_train()
            if not swap_finished: print('SWAP_MANAGER: SWAPPING DIDNT FINISH ON TIME!')

        # Add stream to replay, make sure each class has the same number of seats 
        multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)

        # Clear stream to take new class 
        self.num_files_per_label.update(self.stream_dataset.num_files_per_label)
        self.samples_seen += self.samples_in_stream
        print("Samples in storage : ", self.samples_seen)
        self.stream_dataset.clean_stream_dataset()
        gc.collect()

        #Collect accuracies 
        avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc= self.eval_task(num_tasks=self.tasks_so_far)
        if self.save_tasks == True :
            path = f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt'
            if not os.path.exists(path): 
                os.makedirs(f'{self.checkpoint_path}/',mode = 0o777, exist_ok = True)
                torch.save(self.model.state_dict(), path)
                print(f'SAVING TASK{task_id+1} at {path}',flush=True)
        print("Task_accuracy : ", task_top1_acc)
        print("Task_top5_accuracy : ", task_top5_acc)
        print("Class_accuracy : ", class_top1_acc)
        print("Class_top5_accuracy : ", class_top5_acc)
        print("Current_accuracy : ", avg_top1_acc)
        print("Current_top5_accuracy : ", avg_top5_acc)

        self.soft_incremental_top1_acc.append(avg_top1_acc)
        self.soft_incremental_top5_acc.append(avg_top5_acc)

        print("Incremental_top1_accuracy : ", self.soft_incremental_top1_acc)
        print("Incremental_top5_accuracy : ", self.soft_incremental_top5_acc)
        print("\n")
        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        f.write(str(datetime.datetime.now())+'\n'+'Task '+str(self.tasks_so_far)+'\n')
        f.write("Task_accuracy : "+str(task_top1_acc)+"\n")
        f.write("Task_top5_accuracy : "+str(task_top5_acc)+"\n")
        f.write("Class_accuracy : "+str(class_top1_acc)+"\n")
        f.write("Class_top5_accuracy : "+str(class_top5_acc)+"\n")
        f.write("Incremental_accuracy : "+str(self.soft_incremental_top1_acc)+"\n")
        f.write("Incremental_top5_accuracy : "+str(self.soft_incremental_top5_acc)+"\n")
        if self.swap: f.write(f'Num_swapped : {num_swapped}\n')
        if self.optimizer: f.write(f'(Iter Trained, Iter Profiled)=({self.train_iter},{self.overhead})\n')
        f.close()
    def train(self):
        if self.swap is True:
            self.swap_manager.resume()
        if self.tasks_so_far == 1: 
            f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
            f.write(f'START TIME {datetime.datetime.now()}\n')
            f.close()
        # print( (self.optimizer and self.tasks_so_far <self.optimizer.start_point) and self.load_from_history==True)
        # print(os.path.exists(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt'))
        # print(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt')
        if  os.path.exists(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt') and self.load_from_history==True:
            self.model.load_state_dict(torch.load(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt', map_location = self.device))
            print(f'TRAIN LOAD FROM {self.checkpoint_path}/Task_{self.tasks_so_far}.pt')
            return
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = True
        self.reset_opt(self.tasks_so_far)
        print(f'\n----[TRAINING TASK{self.tasks_so_far}], (EM, SB) = ({len(self.replay_dataset)}, {len(self.stream_dataset)})----')
        self.model.train()
        if self.jetson: 
            self.power_log.recordEvent(name=f'Training, start, T{self.tasks_so_far}')
        for epoch in range(self.num_epochs):
            if self.swap_skip: condition = ((epoch+1)%self.swap_period!=0) #and self.rb_size < self.samples_seen
            else: condition = ((epoch+1)%self.swap_period==0) #and self.rb_size < self.samples_seen
            
            for i, (idxs, inputs, targets,filenames) in enumerate(self.cl_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)

                # Determine Swap Targets
                if self.swap and self.tasks_so_far > 1 and  condition:
                    self.swap_manager.swap_pt(idxs.tolist(),targets.tolist(),filenames,data_ids=None)
                #Compute Loss
                targets = self.to_onehot(targets, self.classes_so_far).to(self.device)
                loss_value = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
                loss_value = loss_value.mean()
                self.opt.zero_grad()
                loss_value.backward()
                self.opt.step()
            self.train_iter += i+1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.loss_item.append(loss_value.item())

            print("epoch {}, lr {}, loss {}".format(epoch,self.opt.param_groups[0]['lr'], loss_value.item()))
        if self.jetson: 
            self.power_log.recordEvent(name=f'Training, end, T{self.tasks_so_far}')
        self.model.eval()
    def compute_accuracy(self, loader, get_entropy=False,update_history=False,model=None):
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