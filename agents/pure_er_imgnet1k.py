from torch.nn import functional as F
import torch
import torch.nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T
import numpy as np 
import datetime 
from optim.optimizer import Optimizer, set_opt_for_profiler
import copy
import os
from agents.base import Base
# from lib.graph import Graph
from _utils.sampling import multi_task_sample_update_to_RB

from lib.save import DataSaver
from lib.swap_manager import fetch_from_storage_imagenet
import time
import gc
import power_check as pc

# CKPT_PATH = '../saved_checkpoints/'
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

class PureER_IMGNET1K(Base):
    def __init__(self, model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                test_dataset, filename,  **kwargs):
        super().__init__(model, opt_name, lr, lr_schedule, lr_decay, device, num_epochs, swap,
                        transform, data_manager, stream_dataset, replay_dataset, cl_dataloader, test_set,
                        test_dataset, filename, **kwargs)
        self.classes_so_far = 0                         # Number of classes in the model
        self.classes_seen=[]                            # List of classes which are model have seen
        self.tasks_so_far = 0                           # Number of tasks in the model 
        self.samples_seen = 0
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"        # cmt  
        self.soft_incremental_top1_acc = list()         # Top 1 accuracy
        self.soft_incremental_top5_acc = list()         # Top 5 accuracy
        # self.num_swap = list()                          # number of samples swapped in each epoch 
        self.criterion = torch.nn.CrossEntropyLoss()    # deprecated
        self.softmax = torch.nn.Softmax(dim=1)
        # self.swap_manager.swap_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.ckpt_model_exist = False
        self.how_long_stay = list()                     # deprecated, how much iter a sample in replay stayed before swapped out 
        self.how_much_reused = list()                   # deprecated, how much iter a sample in replay stayed before swapped out
        self.num_files_per_label = dict()
        self.total_iter = 0
        self.opt_iter = 0
        #debug
        # self.base_ckpt_path =f'{CKPT_PATH}/{self.test_set}/'
        self.checkpoint_path = self.base_ckpt_path
        self.rb_size = self.replay_dataset.rb_size
        # For half of the passes, use random; The other half, use 'swap-base'
        if 'dynamic' in self.swap_options:
            self.dynamic = True
        else:
            self.dynamic = False
            
        # cmt
        if 'ckpt_rand' in kwargs:
            self.ckpt_rand = kwargs['ckpt_rand']
        else:
            self.ckpt_rand = 0
        
        # smartness of the pretrained model;
        if 'p_epoch' in kwargs:
            self.p_epoch = kwargs['p_epoch']
        else:
            self.p_epoch = 150
        
        # How ratio of the samples to be blacklisted
        if 'bl_percent' in kwargs:
            self.bl_percent = kwargs['bl_percent']
        else:
            self.bl_percent = None
            
        # cmt
        if 'bl_percent_e' in kwargs: 
            self.bl_percent_e = kwargs['bl_percent_e']
        else:
            self.bl_percent_e = 1
        
        # deprecated; partial blacklisting
        if 'partial' in kwargs:
            self.partial = kwargs['partial']
        else:
            self.partial = 0
    
        if 'manual_config' in kwargs:
            self.manual_config = kwargs['manual_config']
        else:
            self.manual_config = None
        # Drawing graph
        if 'graph' in kwargs:
            self.draw_graph = kwargs['graph']
        else:
            self.draw_graph = False

        if 'optimizer' in kwargs: 
            kwargs['optimizer']['log_file'] = f'{self.result_save_path}_{self.filename}_optimizer.csv'
            kwargs['optimizer']['test_set'] = self.test_set
            self.optimizer = Optimizer(self.observe_2,kwargs['optimizer'],self.device,self.make_additional_ckpt)
            print('OPTIMIZER ACTIVATED')
        else: self.optimizer=None

        # Make a folder for each run 
        # os.makedirs(f'{self.result_save_path}/{self.filename}/', mode = 0o777, exist_ok = True) 
        # self.result_save_path = f'{self.result_save_path}/{self.filename}/'

        if self.save_tasks:
            os.makedirs(f'{self.result_save_path}/checkpoints/', mode = 0o777, exist_ok = True)


    def before_train(self, task_id):
        self.task_id = task_id
        self.curr_task_iter_time = []
        self.curr_task_swap_iter_time = []
 
        # Add NEW TASK to stream, and test dataset
        self.model.eval()
        self.stream_dataset.create_task_dataset(task_id)
        print("len(self.test_dataset : " + str(len(self.test_dataset)))
        # Upload data loader for training
        if(self.swap_mode != 'one_level'):
            print(f'USED DEFAULT update_loader')
            self.cl_dataloader.update_loader()

        else:
            self.cl_dataloader._update_loader(self.replay_dataset) 
        replay_classes = self.classes_so_far
            
        # Update meta-data        
        self.classes_seen = self.classes_seen + [e for e in self.stream_dataset.classes_in_dataset if e not in self.classes_seen]
        self.classes_so_far = len(self.classes_seen)
        print("classes_seen : ", self.classes_seen)
        print("classes_so_far : ", self.classes_so_far)
        self.tasks_so_far = self.task_id +1
        print("tasks_so_far : ", self.tasks_so_far)
        self.samples_in_stream = len(self.stream_dataset)
        
        self.model.Incremental_learning(self.classes_so_far,self.device)
            
        self.replay_size = len(self.replay_dataset)
        print("length of replay dataset : ", self.replay_size)

        self.stream_losses, self.replay_losses = list(), list()


        # Prepare swap manager
        if self.swap is True:
            self.swap_manager.before_train(self.replay_dataset,self.task_id)
            if self.swap_manager.saver is not None: 
                st = 0
                while True:
                    en = st + 500
                    self.swap_manager.saver.save(self.stream_dataset.data[st:en], self.stream_dataset.targets[st:en], self.stream_dataset.filename[st:en])
                    print("SAVING ALL STREAM SAMPLES...WORKER")
                    st = en
                    if st > len(self.stream_dataset):
                        break
        if self.optimizer:  
            # self.swap_manager.after_train(now=True)
            if self.swap and self.tasks_so_far>1:
                print('swap_manager paused')
                self.swap_manager.pause()
                print('swap_manager queue cleared')
            if self.manual_config and self.tasks_so_far<=len(self.manual_config['rb_size'])  :# min(len(self.manual_config['rb_size']),self.optimizer.start_point-1):
                best_config = {'rb_size':self.manual_config['rb_size'][task_id],'st_size':self.manual_config['st_size'][task_id]}           
            else:
                base_config = {config:getattr(self,config) for config in self.optimizer.get_params()}
                self.optimizer.set_base_config(base_config)
                os.makedirs(f'{self.result_save_path}/checkpoints/', mode = 0o777, exist_ok = True)
                
                print(f'FIND_BEST_CONFIG:')
                best_config, _ = self.optimizer.find_best_config(self.tasks_so_far)
                self.ckpt_model_exist=False
            # switch configuration
            print(f'best_config: (EM, SB) = ({best_config["rb_size"]}, {best_config["st_size"]})')
            for config in best_config: 
                if config == 'rb_size': 
                    self.rb_size = best_config['rb_size']
                    num_classes = self.classes_so_far-len(self.stream_dataset.classes_in_dataset)
                    if self.tasks_so_far == 1: 
                        self.replay_dataset.offset = {x:0 for x in range(num_classes)}
                        self.replay_dataset.len_per_cls = {x:0 for x in range(num_classes)}
                    if self.replay_dataset.resize(best_config['rb_size'],self.samples_seen, self.swap_manager,
                                fetch_from_storage_imagenet, self.test_set, self.task_id, delete=True) in [1,2]: # need to fill replay from files 
                        memory_per_cls = min(int(self.replay_dataset.rb_size / num_classes), int(self.samples_seen / num_classes))
                        swap_targets = []
                        for i in range(num_classes): swap_targets.extend([i for j in range(memory_per_cls)])
                        swap_idx = [i for i in range(len(swap_targets))]
                        for j in range(len(swap_targets)): self.replay_dataset.targets[j] = swap_targets[j]
                        swap_in_files = []
                        for label in range(num_classes):
                            if self.swap and self.swap_manager.saver:
                                label_list= [f'{label}_{count+1}'  for count in range(min(memory_per_cls,self.swap_manager.saver.get_num_file_for_label_for_swap(label))) ]
                            else: label_list= [f'{label}_{count+1}'  for count in range(memory_per_cls)]
                            swap_in_files.extend(label_list)
                        print('FETCH FROM STORAGE') 
                        self.replay_dataset.filled[0] = len(swap_idx)
                        fetch_from_storage_imagenet(self.replay_dataset, swap_idx,swap_targets,self.task_id,filenames=swap_in_files,testset=self.test_set,transform=T.RandomResizedCrop((400,500)))
                    print(self.replay_dataset.offset)
                elif config == 'st_size': 
                    self.stream_dataset.resize(best_config['st_size'],evict=True)
                if hasattr(self, config): 
                    setattr(self,config,best_config[config])
            if self.swap and self.tasks_so_far>1:
                self.swap_manager.resume()
                self.swap_manager.after_train()
                self.swap_manager.before_train(self.replay_dataset,self.task_id)
        if not self.optimizer and self.st_size != len(self.stream_dataset.data):
            self.stream_dataset.resize(self.st_size,evict=True)
        if self.manual_config and self.tasks_so_far <= len(self.manual_config['rb_size']):# and self.optimizer:
            self.checkpoint_path += f'{self.manual_config["rb_size"][self.tasks_so_far-1]}_{self.manual_config["st_size"][self.tasks_so_far-1]}/'
        else:
            self.checkpoint_path += f'{self.rb_size}_{self.st_size}/'

    def after_train(self,task_id):
        self.model.eval()
        if self.swap == True: 
            swap_finished, num_swapped= self.swap_manager.after_train()
            if not swap_finished: print('SWAP_MANAGER: SWAPPING DIDNT FINISH ON TIME!')
        multi_task_sample_update_to_RB(self.replay_dataset, self.stream_dataset)

        # Clear stream to take new class 
        self.num_files_per_label.update(self.stream_dataset.num_files_per_label)
        self.samples_seen += self.samples_in_stream
        print("samples_so_far : ", self.samples_seen)
        self.stream_dataset.clean_stream_dataset()
        gc.collect()
        f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
        f.write(str(datetime.datetime.now())+'\n'+'Task '+str(self.tasks_so_far)+'\n')
        avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc= self.eval_task(get_entropy=self.get_test_entropy,num_tasks=self.tasks_so_far)

        if self.save_tasks == True :
            # path = f'{self.result_save_path}/checkpoints/Task_{self.tasks_so_far}.pt'
            path = f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt'
            if not os.path.exists(path): 
                os.makedirs(f'{self.checkpoint_path}/',mode = 0o777, exist_ok = True)
                torch.save(self.model.state_dict(), path)
                print(f'SAVING TASK{task_id+1} at {path}',flush=True)

        print("task_accuracy : ", task_top1_acc)
        print("task_top5_accuracy : ", task_top5_acc)

        print("class_accuracy : ", class_top1_acc)
        print("class_top5_accuracy : ", class_top5_acc)

        print("current_accuracy : ", avg_top1_acc)
        print("current_top5_accuracy : ", avg_top5_acc)

        self.soft_incremental_top1_acc.append(avg_top1_acc)
        self.soft_incremental_top5_acc.append(avg_top5_acc)

        print("incremental_top1_accuracy : ", self.soft_incremental_top1_acc)
        print("incremental_top5_accuracy : ", self.soft_incremental_top5_acc)



        f.write("task_accuracy : "+str(task_top1_acc)+"\n")
        f.write("task_top5_accuracy : "+str(task_top5_acc)+"\n")

        f.write("class_accuracy : "+str(class_top1_acc)+"\n")
        f.write("class_top5_accuracy : "+str(class_top5_acc)+"\n")

        f.write("incremental_accuracy : "+str(self.soft_incremental_top1_acc)+"\n")
        f.write("incremental_top5_accuracy : "+str(self.soft_incremental_top5_acc)+"\n")
        f.write("training iteration : "+str(self.total_iter)+"\n")
        f.write("profiling overhead : "+str(self.opt_iter)+"\n")
        f.close()

        f = open(self.result_save_path + 'time.txt','a')
        self.avg_iter_time.append(np.mean(np.array(self.curr_task_iter_time)))
        self.avg_swap_iter_time.append(np.mean(np.array(self.curr_task_swap_iter_time)))
        f.write("avg_iter_time : "+str(self.avg_iter_time)+"\n")
        f.write("avg_swap_iter_time : "+str(self.avg_swap_iter_time)+"\n")
        f.close()
    def train(self):     
        if self.tasks_so_far == 1: 
            f = open(self.result_save_path + self.filename + '_accuracy.txt', 'a')
            f.write(f'START TIME {datetime.datetime.now()}\n')
            f.close()
        
        print( (self.optimizer and self.tasks_so_far <self.optimizer.start_point) and self.load_from_history==True)
        print(os.path.exists(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt'))
        print(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt')
        if  os.path.exists(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt') and self.load_from_history==True: 
            self.model.load_state_dict(torch.load(f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt', map_location = self.device))
            print(f'TRAIN LOAD FROM {self.checkpoint_path}/Task_{self.tasks_so_far}.pt')
            return
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = True
        self.reset_opt(self.tasks_so_far)

        print(f'DEF TRAIN:  {len(self.replay_dataset)}, {len(self.stream_dataset)}')
        self.model.train()
        forward_time, backward_time = 0,0

        
        for epoch in range(self.num_epochs):
            # Profiling           
            iter_times = []
            swap_iter_times = []
            iter_st = None
            if self.swap_skip: condition = ((epoch+1)%self.swap_period!=0) #and self.rb_size < self.samples_seen
            else: condition = ((epoch+1)%self.swap_period==0) #and self.rb_size < self.samples_seen
            for i, (idxs, inputs, targets,filenames) in enumerate(self.cl_dataloader):
                # print(inputs[0].shape)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                
                #eval_batch needs this
                iter_st = time.perf_counter()
                outputs = self.model(inputs)
                forward_time += time.perf_counter() - iter_st

                # Determine Swap Targets
                if self.swap and self.tasks_so_far > 1 and  condition and self.total_balance==False:
                    self.swap_manager.swap_pt(idxs.tolist(),targets.tolist(),filenames,data_ids=None)
                targets = self.to_onehot(targets, self.classes_so_far).to(self.device)
                loss_value = F.binary_cross_entropy_with_logits(outputs, targets, reduction="mean")
                self.opt.zero_grad()
                loss_value.backward()
                self.opt.step()
            self.total_iter += i+1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #epoch_accuracy = self.eval(1)
            print("lr {}".format(self.opt.param_groups[0]['lr']))
            self.loss_item.append(loss_value.item())

            if epoch > 0:
                self.curr_task_iter_time.append(np.mean(np.array(iter_times)))
                self.curr_task_swap_iter_time.append(np.mean(np.array(swap_iter_times)))

            print("epoch {}, loss {}".format(epoch, loss_value.item()))

        self.model.eval()
    def eval_task(self, get_entropy=False,update_history=False,num_tasks=None,model=None):
        print(f'EVAL_TASK, origin model = {model==None}')
        if num_tasks == None: 
            num_tasks = self.tasks_so_far
        avg_top1_acc, task_top1_acc, class_top1_acc = {},{},{}
        avg_top5_acc, task_top5_acc, class_top5_acc = {},{},{}
        task_size =len(self.stream_dataset.classes_in_dataset)
        #### To reduce memory usage
        if num_tasks >= 10:   
            half = int(num_tasks/2)
            for task in range(half):
                self.test_dataset.append_task_dataset(task)
            print(len(self.test_dataset))
            test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
            ypreds_1, ytrue_1 = self.compute_accuracy(test_dataloader,model=model)    
            self.test_dataset.clean_task_dataset()
            
            for task in range(half,num_tasks):
                self.test_dataset.append_task_dataset(task)
            print(len(self.test_dataset))
            test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
            ypreds_2, ytrue_2 = self.compute_accuracy(test_dataloader,model=model)    
            
            ypreds = np.concatenate([ypreds_1,ypreds_2])
            ytrue = np.concatenate([ytrue_1,ytrue_2])
        else: 
            for task in range(num_tasks):
                self.test_dataset.append_task_dataset(task)
            test_dataloader = DataLoader(self.test_dataset, batch_size = 128, shuffle=False)
            ypreds, ytrue = self.compute_accuracy(test_dataloader,model=model)    
        avg_top1_acc, task_top1_acc, class_top1_acc = self.accuracy_per_task(ypreds, ytrue, task_size=task_size, class_size=task_size, topk=1)
        if self.classes_so_far>=5:
            avg_top5_acc, task_top5_acc, class_top5_acc = self.accuracy_per_task(ypreds, ytrue, task_size=task_size, class_size=task_size, topk=5)
        
        self.test_dataset.clean_task_dataset()
        del test_dataloader
        gc.collect()
        return avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc
        
    def compute_accuracy(self, loader, get_entropy=False,update_history=False,model=None):
        ypred, ytrue = [], []
        if self.swap==True and get_entropy == True:
            w_entropy_test = []
            r_entropy_test = []

            logits_list = []
            labels_list = []
        if update_history == False: 
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
                    #get entropy of testset
                    if self.swap==True and get_entropy == True:
                        r, w = self.get_entropy(outputs, labels)
                        r_entropy_test.extend(r)
                        w_entropy_test.extend(w)
                        
                        logits_list.append(outputs)
                        labels_list.append(labels)


                    ytrue.append(labels.numpy())
                    ypred.append(torch.softmax(outputs, dim=1).cpu().numpy())
        else:
            for i, (idx,imgs, labels,_) in enumerate(loader):
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    outputs = self.model(imgs) if model == None else model(imgs)

                    outputs = outputs.clone().detach()
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
    def make_additional_ckpt(self, trail_duration=30,**kwargs):
        if 'pretrain' in kwargs: 
            pretrain = kwargs['pretrain']
        else: pretrain = self.load_from_history
        for param in self.model.parameters(): 
            param.requires_grad = False
        if self.test_set == 'cifar100': num_cls = 10 
        elif self.test_set == 'urbansound8k': num_cls = 1
        elif self.test_set == 'imagenet1000': num_cls = 20
        elif self.test_set == 'imagenet_r': num_cls = 20
        else: 
            num_cls = 20
        self.ckpt_model.Incremental_learning(self.classes_so_far-num_cls,self.device)
        self.ckpt_model.load_state_dict(torch.load(f'{self.checkpoint_path}/Task_{self.tasks_so_far-1}.pt',map_location = self.device))
        self.ckpt_model.Incremental_learning(self.classes_so_far,self.device)
        # self.ckpt_model = copy.deepcopy(self.model)
        # ckpt_path
        if trail_duration == self.num_epochs: 
            ckpt_path =  f'{self.checkpoint_path}/Task_{self.tasks_so_far}.pt'
        else: 
            ckpt_path = f'{self.checkpoint_path}/Task_{self.tasks_so_far}_additional_ckpt_epoch{trail_duration}.pt'

        # self.reset_opt()
        self.ckpt_model.to(self.device)
        self.cl_dataloader.update_loader()
        log_path = f'{self.result_save_path}/{self.filename}_optimizer.txt'
        f = open(log_path, 'a')
        
        print(f'stream_size: {len(self.stream_dataset)}   replay_size: {len(self.replay_dataset)}')
        print(f'train set size: {len(self.stream_dataset) + len(self.replay_dataset)}')
        f.write(f'Task {self.tasks_so_far} CHECKPOINT\n')
        f.write(f'stream_size: {len(self.stream_dataset)}   replay_size: {len(self.replay_dataset)}\n')
        f.write(f'train set size: {len(self.stream_dataset) + len(self.replay_dataset)}\n')
        
        for i, (name, param) in enumerate(self.ckpt_model.named_parameters()):
            param.requires_grad=True
        ckpt_opt, ckpt_lr_scheduler =  set_opt_for_profiler(self.test_set,self.ckpt_model,trail_duration)
        if self.swap:
            self.swap_manager.resume()
            self.swap_manager.update_meta(len(self.replay_dataset))
            
        time_ = 0
        for epoch in range(trail_duration):  
            for i, (idxs, inputs, targets,filenames) in enumerate(self.cl_dataloader):
                # if len(self.replay_dataset)>0:
                #     with open(f'debug.txt','w') as debug_file:
                #         debug_file.write(f'{self.replay_dataset[0][1][0]}')
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
                f.write(f"EPOCH {epoch}, ITER {i}, loss {loss_value.item()}\n")                  
            if ckpt_lr_scheduler is not None:
                ckpt_lr_scheduler.step()
            f.write("lr {}".format(ckpt_opt.param_groups[0]['lr']))
            f.write("epoch {}, loss {}\n".format(epoch, loss_value.item()))
            f.flush()
        if ((not os.path.exists(ckpt_path)) or pretrain==False )and self.save_tasks: 
            os.makedirs(f'{self.checkpoint_path}/',mode = 0o777, exist_ok = True)
            torch.save(self.ckpt_model.state_dict(), ckpt_path)
            print(f'Saved CHECKPOINT to {ckpt_path} ')
        if self.swap:
            print('swap_manager paused')
            self.swap_manager.pause()
            print('swap_manager queue cleared')

        if (self.test_set not in ["urbansound8k", "twentynews", "dailynsports", "shakespeare"]):
            avg_top1_acc, task_top1_acc, class_top1_acc, avg_top5_acc, task_top5_acc, class_top5_acc= self.eval_task(get_entropy=self.get_test_entropy,num_tasks=self.tasks_so_far,model=self.ckpt_model)
        else: 
            avg_top1_acc, class_top1_acc, avg_top5_acc, class_top5_acc= self.eval_task_blurry(model=self.ckpt_model)
                
        if 'imagenet' in self.test_set: 
            print(f'MAKE CHECKPOINT, acc={avg_top5_acc}')
        else:
            print(f'MAKE CHECKPOINT, acc={avg_top1_acc}')
        print(f'rb_size:{len(self.replay_dataset)}, st_size:{len(self.stream_dataset)}')
        self.ckpt_model_exist = True
        return [avg_top1_acc, class_top1_acc], ckpt_path
    
    def observe_2(self,config:tuple, ckpt_model=None, ckpt_opt=None, ckpt_lr_scheduler=None, trail_duration=None,**kwargs):
        if 'pretrain' in kwargs: 
            pretrain = kwargs['pretrain']
        else: pretrain = self.load_from_history
        if 'mk_ckpt' in kwargs: 
            mk_ckpt = kwargs['mk_ckpt']
        else: mk_ckpt = False
        


        print('OBSERVE_2', end=' ')
        print(config)      
        if config[0] > self.samples_seen:  
            print('Replay larger than needed')
            return {'acc':0, 'energy':0,'energy_estim':0,'cls_acc':{0:0}}
        for param in self.model.parameters(): 
            param.requires_grad = False

        # 0309
        if trail_duration == self.num_epochs and not self.optimizer.use_ckpt and len(self.optimizer.layer_freeze) == 0 and self.optimizer.data_ratio ==1: 
            ckpt_path = f'{self.checkpoint_path}/{config[0]}_{config[1]}/Task_{self.tasks_so_far}.pt'
        else: 
            if not self.optimizer.use_ckpt: ckpt_size = 'ckpt0'
            else: ckpt_size = f'ckpt{self.optimizer.ckpt_size}'
            ckpt_path = f'{self.checkpoint_path}/{config[0]}_{config[1]}/Task_{self.tasks_so_far}_ckpt{ckpt_size}_epoch{trail_duration}_dr{self.optimizer.data_ratio}.pt'

        if  os.path.exists(ckpt_path) and pretrain==True and self.load_from_history:
            ckpt_model = copy.deepcopy(self.model)
            ckpt_model.load_state_dict(torch.load(ckpt_path, map_location = self.device))
            print(f'PROFILE LOAD FROM {ckpt_path}')
        else:
            if self.optimizer.use_ckpt:
                if self.ckpt_model_exist == False: 
                    ori_config = config
                    config=[25000,15600] # 50 tasks
                    # int(config[0]*self.optimizer.data_ratio
                    res = self.replay_dataset.resize(int(config[0]*self.optimizer.data_ratio),self.samples_seen, self.swap_manager,
                            fetch_from_storage_imagenet, self.test_set, self.task_id, delete=False)
                    print(f'resize returned {res}')
                    if res in [1,2]: # need to fill replay from files 
                        memory_per_cls = min(int(self.replay_dataset.rb_size / len(self.replay_dataset.offset)), int(self.samples_seen / len(self.replay_dataset.offset)))
                        fetch_targets = []
                        for i in self.replay_dataset.offset: 
                            if self.swap_manager.saver:
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
                        print('FETCH FROM STORAGE')

                        fetch_from_storage_imagenet(self.replay_dataset, fetch_idx,fetch_targets,self.task_id, filenames=fetch_files, testset=self.test_set,transform=T.RandomResizedCrop((400,500)))
                        self.replay_dataset.filled[0] = len(fetch_idx)
                    self.stream_dataset.resize(int(config[1]*self.optimizer.data_ratio))               
                    self.make_additional_ckpt(self.optimizer.ckpt_size,pretrain=False)
                    for param in self.ckpt_model.parameters(): 
                        param.requires_grad = False
                    ckpt_model = copy.deepcopy(self.ckpt_model)  
                    config = ori_config
                else: 
                    ckpt_model = copy.deepcopy(self.ckpt_model)
                    for param in self.ckpt_model.parameters(): 
                        param.requires_grad = False
            else: 
                ckpt_model = copy.deepcopy(self.model)
                for param in self.ckpt_model.parameters(): 
                    param.requires_grad = False
            res = self.replay_dataset.resize(int(config[0]*self.optimizer.data_ratio),self.samples_seen, self.swap_manager,
                            fetch_from_storage_imagenet, self.test_set, self.task_id, delete=False)
            if res in [1,2]: # need to fill replay from files 
                memory_per_cls = min(int(self.replay_dataset.rb_size / len(self.replay_dataset.offset)), int(self.samples_seen / len(self.replay_dataset.offset)))
                fetch_targets = []
                for i in self.replay_dataset.offset: 
                    if self.swap_manager.saver:
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
                print('FETCH FROM STORAGE')

                fetch_from_storage_imagenet(self.replay_dataset, fetch_idx,fetch_targets,self.task_id, filenames=fetch_files, testset=self.test_set,transform=T.RandomResizedCrop((400,500)))
                self.replay_dataset.filled[0] = len(fetch_idx)
            self.stream_dataset.resize(int(config[1]*self.optimizer.data_ratio)) 
            # self.reset_opt()
            ckpt_model.to(self.device)
            self.cl_dataloader.update_loader()
            log_path = f'{self.result_save_path}/{self.filename}_optimizer.txt'
            f = open(log_path, 'a')
            
            print(f'stream_size: {len(self.stream_dataset)}   replay_size: {len(self.replay_dataset)}')
            print(f'train set size: {len(self.stream_dataset) + len(self.replay_dataset)}')
            f.write(f'Task {self.tasks_so_far}\n')
            f.write(f'stream_size: {len(self.stream_dataset)}   replay_size: {len(self.replay_dataset)}\n')
            f.write(f'train set size: {len(self.stream_dataset) + len(self.replay_dataset)}\n')
            
            # if data_ratio == 1: no subsamling
            task_size = (len(self.replay_dataset)+len(self.stream_dataset))
            total_param, frozen_param=0,0            
            for i, (name, param) in enumerate(ckpt_model.named_parameters()):
                param.requires_grad_(False)
                total_param+=1
                condition =[layer in name for layer in self.optimizer.layer_freeze] 
                if True not in condition: 
                    param.requires_grad=True
                else: 
                    param.requires_grad=False
                    frozen_param+=1
            print(f'Froze {frozen_param}/{total_param} parameters')
            ckpt_opt, ckpt_lr_scheduler =  set_opt_for_profiler(self.test_set,ckpt_model,trail_duration)
            if self.swap:
                self.swap_manager.resume()
                self.swap_manager.update_meta(len(self.replay_dataset))
                
            print(f'start...',end='')
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
                self.opt_iter += i+1
                if ckpt_lr_scheduler is not None:
                    ckpt_lr_scheduler.step()
                f.write("lr {}".format(ckpt_opt.param_groups[0]['lr']))
                f.write("epoch {}, loss {}\n".format(epoch, loss_value.item()))
                f.flush()
            print(f'...end',end='')

            # saving mini-checkpoints
            if not os.path.exists(ckpt_path) or pretrain==False: 
                os.makedirs(f'{self.checkpoint_path}/{config[0]}_{config[1]}/',mode = 0o777, exist_ok = True)
                torch.save(ckpt_model.state_dict(), ckpt_path)
                print(f'Saved to {ckpt_path} ')
            if self.swap:
                print('swap_manager paused')
                self.swap_manager.pause()
                print('swap_manager queue cleared')

        # energy_estim = (self.num_epochs/trail_duration)*((len(self.replay_dataset)+len(self.stream_dataset))/(self.cl_dataloader.batch_size*num_batch))*energy
        energy_estim = config[0] + config[1]
        
        # load before test for consisitency 
        avg_top1_acc, _, class_top1_acc, avg_top5_acc, _, class_top5_acc= self.eval_task(get_entropy=self.get_test_entropy,num_tasks=self.tasks_so_far,model=ckpt_model)
        if self.test_set in ['cifar100', 'urbansound8k']:
            result = {'acc':avg_top1_acc, 'energy':energy_estim,'energy_estim':energy_estim,'cls_acc':class_top1_acc}
        else: 
            result = {'acc':avg_top5_acc, 'energy':energy_estim,'energy_estim':energy_estim,'cls_acc':class_top5_acc}
        return result