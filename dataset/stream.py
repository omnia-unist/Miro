from pydoc import source_synopsis
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from collections import defaultdict, deque
import time
from queue import Queue
import math
import random
#from bitmap import BitMap
TIMEOUT = 1
SLEEP = 0.001


if __debug__:
    pass
"""
    import logging
    import time
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
"""
# Hard coded
def filename_decode(filename):
    cls_no= int(filename.split('_')[0])

    sample = int(filename.split('_')[1])-1

    return cls_no,sample

class StreamDataset(IterableDataset):
    def __init__(self, batch, transform):
        self.batch = batch
        self.transform = transform
        self.raw_data_dq = deque(list(), self.batch)

        self.data = []
        self.targets = []
        self.tasks = []

        self._recv_data = Queue()

    def append_stream_data(self, vec, label, task_id):
        self.put_queue(vec,label,task_id)

        self.data.append(vec)
        self.targets.append(label)
        self.tasks.append(task_id)

    def put_queue(self, vec, label, task_id):
        self._recv_data.put((vec,label,task_id))
    
    def __iter__(self):
        while True:
            try:
                vec, label, task_id = self._recv_data.get(timeout=TIMEOUT)
            except Exception: 
                time.sleep(SLEEP)
                continue

            self.raw_data_dq.append(vec)
            if self.transform is not None:
                vec = self.transform(vec)
            
            yield (vec, label, task_id)

class DatasetHistory():
    def __init__(self,grad=True,entropy=True,pred=True,online_storage=False,his_len=20,samples_per_cls=500):
        # self.num_tasks = 0 
        # self.cls_per_task = 0 
        self.grad = grad
        self.entropy = entropy
        self.pred = pred
        self.tasks_so_far = 0
        # hard coded 
        self.sample_per_cls = samples_per_cls
        if self.pred==True: 
            self.pred_his = []
            self.fgt = []
            self.fgt_len = []
            self.fgt_score = [] #None
            # gradient & entropy
            if self.grad==True: self.fgt_grad = []
            if self.entropy==True: self.fgt_ent = []
            self.window = False
            
            self.swap_in_targets = None
            self.rp_loc=[]
            self.window = True
            self.his_len = his_len

    def update_swap_in_targets(self,bl_percent):
        if self.pred == False:
            return
        if self.fgt_score == None:
            return
        # for imagenet
        if bl_percent == None: 
            self.swap_in_targets = None
            return
        self.swap_in_targets = torch.ones((self.fgt_score.shape),dtype=int)
        thr = int(self.fgt_score.shape[1] * abs(bl_percent))
        for i in range(self.fgt_score.shape[0]):
            arg_rank = torch.argsort(self.fgt_score[i],descending=(bl_percent>0))
            arg_rank = arg_rank[:thr]
            self.swap_in_targets[i,arg_rank] = 0

    def update_rp_loc(self,replay_dataset):
        self.rp_loc = []
        if not replay_dataset.is_filled(): return self.rp_loc
        filenames = [filename for filename in replay_dataset.filenames] 
        filenames = filenames[:len(replay_dataset)]
        for filename in filenames: 
            cls_no,sample = filename_decode(filename)
            idx = int(cls_no*self.sample_per_cls)+ sample
            self.rp_loc.append(idx)   
        return self.rp_loc 
        
    def add_task(self,num_cls):
        task_his = torch.full((num_cls,self.sample_per_cls,1),-1)
        task_his = task_his.tolist()
        task_fgt = torch.full((num_cls,self.sample_per_cls),-1)
        task_score = torch.full((num_cls,self.sample_per_cls),-1) #-100
        task_len = torch.full((num_cls,self.sample_per_cls),0)
        task_output = torch.full((num_cls,self.sample_per_cls),-1)
        
        if len(self.fgt) ==0: 
            self.pred_his = task_his
            self.fgt = task_fgt
            self.fgt_score = task_score
            self.fgt_len = task_len
            self.fgt_output = task_output
            # gradient & entropy
            if self.grad==True: 
                task_grad = torch.full((num_cls,self.sample_per_cls),-1)
                self.fgt_grad = task_grad
            if self.entropy==True:
                task_ent = torch.full((num_cls,self.sample_per_cls),-1)
                self.fgt_ent = task_ent
        else:
            # assert(len(self.pred_his)==task_no-1)
            self.pred_his.extend(task_his)
            self.fgt = torch.cat((self.fgt,task_fgt),dim=0)
            self.fgt_score = torch.cat((self.fgt_score,task_score),dim=0)
            self.fgt_len = torch.cat((self.fgt_len,task_len),dim=0)
            self.fgt_output = torch.cat((self.fgt_output,task_output),dim=0)
            # gradient & entropy
            if self.grad==True: 
                task_grad = torch.full((num_cls,self.sample_per_cls),-1)
                self.fgt_grad = torch.cat((self.fgt_grad,task_grad),dim=0)
            if self.entropy==True:
                task_ent = torch.full((num_cls,self.sample_per_cls),-1)
                self.fgt_ent = torch.cat((self.fgt_ent,task_ent),dim=0)
        self.tasks_so_far +=1

    def get_sub_score(self,filenames):
        sub_fgt, sub_fgt_score, sub_fgt_output, sub_fgt_len = [],[],[],[]
        if self.grad==True: sub_fgt_grad = []
        if self.entropy==True: sub_fgt_ent = []
        for filename in filenames:
            c,s = filename_decode(filename)
            sub_fgt.append(self.fgt[c][s])
            sub_fgt_score.append(self.fgt_score[c][s])
            sub_fgt_len.append(self.fgt_len[c][s])
            sub_fgt_output.append(self.fgt_output[c][s])
            # gradient & entropy
            if self.grad==True: sub_fgt_grad.append(self.fgt_grad[c][s])
            if self.entropy==True: sub_fgt_ent.append(self.fgt_ent[c][s])
            
        if self.grad==True: 
            if self.entropy==True: return sub_fgt,sub_fgt_score,sub_fgt_len,sub_fgt_output,sub_fgt_grad,sub_fgt_ent
            else: return sub_fgt,sub_fgt_score,sub_fgt_len,sub_fgt_output,sub_fgt_grad,None
        elif self.entropy==True: return sub_fgt,sub_fgt_score,sub_fgt_len,sub_fgt_output,None,sub_fgt_ent
        else: return sub_fgt,sub_fgt_score,sub_fgt_len,sub_fgt_output,None,None

    def eval_batch(self,ypreds,ytrue,filenames,loss_batch,soft_output,bl_percent=None): #outputs,targets,filenames

        newly_learned, newly_forgotten = {x:0 for x in range(self.tasks_so_far)},{x:0 for x in range(self.tasks_so_far)}    
        ypreds,ytrue = ypreds.clone().detach(), ytrue.clone().detach()
        _,ypreds = ypreds.topk(1,1,True,True)
        ypreds = ypreds.t()
        self.ypreds, self.loss_batch, self.soft_output = ypreds, loss_batch, soft_output

        assert(len(filenames)==len(ypreds[0])==len(ytrue))
        correct = ypreds.eq(ytrue.view(1, -1).expand_as(ypreds)).reshape(-1)
        # num_correct = correct.eq(True).sum()
        
        if self.grad==True: 
            self.avg_grad_y, self.avg_grad_l = torch.mean(ypreds.float()), torch.mean(loss_batch.float())
            #self.avg_grad = (avg_grad) #/ np.linalg.norm(avg_grad)
        #if self.entropy==True:
            #MSEloss = nn.MSELoss()
            #self.avg_e = torch.mean(loss_batch.float())
            #self.mse_e = MSEloss(ypreds, loss_batch)
            
        for i in range(len(filenames)):
            c,s = filename_decode(filenames[i])
            
            if correct[i] == True: 
                if self.pred_his[c][s][0] == -1:
                    self.pred_his[c][s][0] = 1
                else:
                    self.pred_his[c][s].append(1)
                #fgt
                # if self.fgt[c][s]==-1: #learned
                #     self.fgt[c][s]=0
                #     newly_learned[t]+=1
            else:
                if self.pred_his[c][s][0] == -1:
                    self.pred_his[c][s][0] = 0
                else:
                    self.pred_his[c][s].append(0)
                #fgt
                # if self.fgt[c][s] >=0: # forgetting
                #     if(self.fgt[c][s]==0):
                #         newly_forgotten[t] +=1
                #     self.fgt[c][s] +=1

            #fgt score
            # if self.fgt[c][s]==-1: self.fgt_score[c][s]= -1
            self.fgt[c][s], self.fgt_score[c][s]= self.cal_score(c,s)     
            #fgt len
            self.fgt_len[c][s]=len(self.pred_his[c][s])

            #fgt output
            self.ypreds_i = [round((ypreds[j][i]).item(), 3) for j in range(len(ypreds))]
            self.ypreds_i = sum(self.ypreds_i) / len(self.ypreds_i)
            self.fgt_output[c][s] =np.round(self.ypreds_i,2)
            #fgt gradient & entropy
            if self.grad==True: self.fgt_grad[c][s]=self.cal_grad(i)
            if self.entropy==True: self.fgt_ent[c][s]=self.cal_entropy(i)
            if bl_percent is not None:
                self.update_swap_in_targets(bl_percent)
            else:
                self.update_swap_in_targets(None)

        return newly_learned, newly_forgotten

    def cal_score(self,c,s): #forgetting score
        fgt_len = len(self.pred_his[c][s])
        if (fgt_len <= self.his_len) or (self.window==False):
            # Consider all 
            pred_his = self.pred_his[c][s]
        else: 
            # Consider only the last HIS_LEN records 
            pred_his = self.pred_his[c][s][-self.his_len:]
            fgt_len = self.his_len
        if 1 not in pred_his: 
            fgt, fgt_score = -1,-1 
        else: 
            st = pred_his.index(1)
            pred_his = pred_his[st:]
            fgt = pred_his.count(0)
            fgt_score = int((fgt/fgt_len) * 100)
        return fgt,fgt_score
        
    def cal_grad(self,i): #gradient
        sample_grad = (self.avg_grad_y*self.ypreds_i + self.avg_grad_l*self.loss_batch[i])*100
        #sample_grad = (sample_grad) #/ np.linalg.norm(sample_grad)
        #sample_dot = np.nan_to_num(np.dot(self.avg_grad,sample_grad)*100)
        #sample_cos = np.nan_to_num(np.arccos(sample_dot))
        fgt_g = round(sample_grad.item(), 3) #np.degrees(sample_cos)
        return fgt_g
        
    def cal_entropy(self,i): #entropy
        #fgt_e = int(((self.avg_e - self.loss_batch[i]))) + int(((self.mse_e - self.loss_batch[i])))
        #fgt_e = self.loss_batch_f[i]
        sample_ent = (torch.distributions.categorical.Categorical(probs=self.soft_output*100).entropy()).tolist()[i]
        fgt_e = round(sample_ent, 3)
        return fgt_e


class MultiTaskStreamDataset(Dataset):
    def __init__(self, batch, samples_per_task, transform=None,samples_per_cls=500,device='cpu',test_set='cifar100'):
        self.batch = batch
        self.samples_per_task = samples_per_task
        self.transform = transform
        self.data_queue = dict()

        self.device = device
        self.classes_seen = list()
        self.classes_in_dataset = list()
        self.samples_per_cls = samples_per_cls
        self.data = list()
        self.targets = list()
        self.filename = list()
        self.test_set = test_set
        self.dataset_history = DatasetHistory()
        self.num_files_per_label = dict()
        self.offset = dict()
    def make_mask(self,mask:list=[]):
        if len(mask) == 0: self.mask = [i for i in range(len(self.targets))]
        else: self.mask = mask
        self.available = len(self.mask)
    def resize(self, new_size, evict=False):
        if self.mask: self_len = self.available
        else: self_len = len(self.targets)
        if new_size == self_len :
            return 
        if new_size >= len(self.targets):
            self.mask = None
            return
        if abs(new_size-self_len ) <=5:
            return
        old_size = len(self.targets)
        ratio =  float((old_size-new_size)/(old_size))
        samples_to_evict_per_class = {i:int(self.task_num_files_per_label[i]*ratio) for i in self.task_num_files_per_label}
        offset = self.offset
        new_offset, new_len_per_cls = {},{}
        mask = []
        
        for i in self.offset: 
            mask.extend(list(range(self.offset[i],self.offset[i]+self.task_num_files_per_label[i]-samples_to_evict_per_class[i])))
            st = sum(list(new_len_per_cls.values()))
            length = self.task_num_files_per_label[i]-samples_to_evict_per_class[i]
            new_offset[i] = st 
            new_len_per_cls[i] = length
        if evict: 
            eviction_list = [i for i in range(len(self.targets)) if i not in mask]
            self.evict(eviction_list)
            self.mask=None
            self.offset = new_offset
            self.task_num_files_per_label = new_len_per_cls
            for i in self.num_files_per_label:
                if i in self.task_num_files_per_label: self.num_files_per_label[i] -= samples_to_evict_per_class[i]
            # df = open('temp.txt','a')
            # df.write(f'STREAM_RESIZE, {self_len }--> {new_size}, evict = {evict}\n')
            # df.write(str(self.mask)+'\n')
            # df.write(str(self.filename)+'\n')
            # df.write(str(self.targets)+'\n')
            # df.close()
        else:
            self.make_mask(mask)
            # df = open('temp.txt','a')
            # df.write(f'STREAM_RESIZE, {self_len }--> {new_size}, evict = {evict}\n')
            # df.write(str(self.mask)+'\n')
            # df.close()
            
    def evict(self, eviction_list):
        for idx in sorted(eviction_list, reverse=True): 
            del self.data[idx]
            del self.targets[idx]
            del self.filename[idx]    
    def append_stream_data(self, vec, label, task_id, is_train):
        if task_id not in self.data_queue:
            self.data_queue[task_id] = list()
        self.data_queue[task_id].append((vec, label))
        return (None, False)
    
    def clean_stream_dataset(self):
        del self.data[:]
        del self.targets[:]
        del self.filename[:]
        del self.dataset_history

        self.data = list()
        self.targets = list()
        #str
        self.filename = list()
        self.samples_per_task.pop(0)

    def create_task_dataset(self, task_id):
        self.task_num_files_per_label = dict()
        self.offset, self.mask, self.available = {},[],None
        self.classes_seen = list(set(self.classes_seen+self.classes_in_dataset))
        self.classes_in_dataset = list()
        self.dataset_history = DatasetHistory()
        i = 0
        while True:
            if not task_id in self.data_queue: # Key error
                print("KeyError: " + str(task_id))
                break
            elif len(self.data_queue[task_id]) == 0: #or len(self.data) >= self.samples_per_task:
                break
            vec, label = self.data_queue[task_id][i]
            self.data.append(vec)
            self.targets.append(label)
            if label not in self.classes_in_dataset:
                self.classes_in_dataset.append(label)
            if label not in self.num_files_per_label:
                self.num_files_per_label[label] =0
            self.num_files_per_label[label] += 1
            if label not in self.task_num_files_per_label:
                self.task_num_files_per_label[label] =0
            self.task_num_files_per_label[label] += 1

            self.filename.append(f'{label}_{self.num_files_per_label[label]}')
            
            del self.data_queue[task_id][i]
        self.offset[min(self.task_num_files_per_label)]=0
        cls_list = sorted(list(self.task_num_files_per_label.keys()))
        for i in range (len(cls_list)):
            label = cls_list[i]
            if label in self.offset: continue     
            self.offset[label] = self.offset[cls_list[i-1]] + self.task_num_files_per_label[cls_list[i-1]] 
        """
        del self.data_queue[task_id][:self.samples_per_task]
        """
    def append_task_dataset(self,new_stream):
        if self.data is False and self.targets is False:
            self.data = list()
            self.targets = list()
            self.filename = list()
            self.history = dict()

        i = 0
        cnt_label = defaultdict(int) # str
        self.targets.extend(new_stream.targets)
        self.filename.extend(new_stream.filename)
        self.classes_in_dataset.extend(new_stream.classes_in_dataset)
        self.dataset_history.add_task(len(new_stream.classes_in_dataset))    
        self.samples_per_cls = new_stream.samples_per_cls
    def split(self, ratio=0.1):
        stream_val_data, stream_val_target, stream_rep_data, stream_rep_target = [],[],[],[]
        stream_val_ac_index, stream_rep_ac_index = [],[]
        
        for new_label in self.classes_in_dataset:
            sub_data, sub_label,_, actual_index = self.get_sub_data(new_label)
            num_for_val_data = math.ceil(len(sub_data) * ratio)

            stream_val_data.extend(sub_data[:num_for_val_data])
            stream_val_target.extend(sub_label[:num_for_val_data])
            stream_val_ac_index.extend(actual_index[:num_for_val_data])

            stream_rep_data.extend(sub_data[num_for_val_data:])
            stream_rep_target.extend(sub_label[num_for_val_data:])
            stream_rep_ac_index.extend(actual_index[num_for_val_data:])
        
        return stream_val_data, stream_val_target, stream_val_ac_index, stream_rep_data, stream_rep_target, stream_rep_ac_index
    
    def get_sub_data(self, label):
        sub_data = []
        sub_label = []
        actual_index = []
        sub_filenames = []
        for idx in range(len(self.data)):
            if self.targets[idx] == label:
                sub_data.append(self.data[idx])
                sub_label.append(label)
                actual_index.append(idx)
                sub_filenames.append(self.filename[idx])

        return sub_data, sub_label, sub_filenames,actual_index
    def get_sub_data_idx(self, label):
        actual_index = []
        for idx in range(len(self.data)):
            if self.targets[idx] == label:
                actual_index.append(idx)

        return actual_index
    def __len__(self):
        if self.mask: return self.available 
        else: return len(self.targets)
    def __getitem__(self, idx):
        if self.mask: idx = self.mask[idx]
        vec = self.data[idx]
        if self.transform is not None:
            vec = self.transform(vec)
        label = self.targets[idx]
        filename = self.filename[idx]

        return idx, vec, label,filename

class OnlineStorage(MultiTaskStreamDataset):
    def __init__(self, batch, samples_per_task, transform,samples_per_cls,test_set):
        super().__init__(batch, samples_per_task, transform,samples_per_cls,test_set)
        self.batch = batch
        self.samples_per_task = samples_per_task
        self.transform = transform
        self.data_queue = dict()

        self.classes_in_dataset = list()
        self.samples_per_cls = 500
        self.data = list()
        self.targets = list()
        #str
        self.filename = list()
        self.history = dict()
        self.micro_history = dict()
        self.dataset_history = DatasetHistory()
        self.bitmap_init()
        
    def bitmap_init(self):

        if self.test_set in ['tiny_imagenet']: num_cls = 200 
        elif  self.test_set in ['imagenet1000']: num_cls = 1000
        else: num_cls = 100
        size = self.samples_per_cls*num_cls # For tiny imagenet it's 200
        # self.bitmap = torch.zeros((3,size))
        self.bitmap = torch.ones((3,size))
        self.bitmap_size = size
        self.frequency = torch.zeros(size)
        print('bitmap created!')
    def bitmap_clear(self): 
        self.bitmap = torch.zeros((3,self.bitmap_size))

    def bitmap_set_gray(self, pos,cnt):
        cls_no = int(pos/self.samples_per_cls)
        sample_no = int(pos%self.samples_per_cls)

        if self.dataset_history.pred==True:
            if self.dataset_history.swap_in_targets==None:
                t = torch.tensor([0,0,0])
            elif self.dataset_history.swap_in_targets.shape[0]>cls_no and self.dataset_history.swap_in_targets[cls_no][sample_no].item() == 0:
                t = torch.tensor([0,0.5,1])
                cnt+=1
            else: 
                t = torch.tensor([0,0,0])
        else:
            t = torch.tensor([0,0,0])
        self.bitmap[:,pos] = t
        return cnt
    def bitmap_set_by_score(self, pos):
        cls_no = int(pos/self.samples_per_cls)
        if cls_no not in self.classes_in_dataset:
            self.bitmap[:,pos] = torch.tensor([0,0,0])
            return
        else:
            sample_no = int(pos%self.samples_per_cls)
            score = self.dataset_history.fgt_score[cls_no][sample_no]
            tone = (score)/100
            tone = tone.item()
            if self.dataset_history.pred == True and self.dataset_history.swap_in_targets is not None and len(self.dataset_history.swap_in_targets) > cls_no and self.dataset_history.swap_in_targets[cls_no][sample_no].item() == 0:
                t = torch.tensor([0,0.5,1])
            elif tone < 0: # negative score
                t = torch.tensor([0.85,0.85,0.85])
            elif tone <=0.33:
                t = torch.tensor([0,1,0])
            elif tone <=0.67:
                t = torch.tensor([1,0.5,0])
            else:
                t = torch.tensor([1,0,0])
            self.bitmap[:,pos] = t
    def bitmap_update(self,rp_loc=None):
        self.bitmap_init()
        cnt = 0
        if rp_loc == None:
            rp_loc = self.dataset_history.rp_loc
        for i in range(self.bitmap.shape[1]):
            if i in rp_loc:
                if self.dataset_history.pred==False:
                    continue
                self.bitmap_set_by_score(i)
            else:
                cnt = self.bitmap_set_gray(i,cnt)
        return self.bitmap

    def frequency_bitmap_update(self, rp_loc): 
        increment = 0.2
        self.frequency[rp_loc] += 1
        for pos in rp_loc:
            red_n = self.frequency[pos]*increment
            if red_n >=1: red_n = 1
            non_red = float(1 - red_n)
            self.bitmap[:,pos] = torch.tensor([1,non_red,non_red])
        return self.bitmap
        
        pass
    
    def get_subset(self,size,rb_size,bl=False):
        subset = MultiTaskStreamDataset(self.batch,self.samples_per_task,self.transform,rb_size)
        target_size = size 
        source_size = len(self.data) - self.samples_per_task[0]

        # get old data 
        source_data, source_targets, source_filename = self.data, self.targets, self.filename
        if target_size >= source_size:
            subset.data, subset.targets,subset.filename = source_data, source_targets, source_filename
            return subset
        else: 
            num_classes = int(len(self.data)/self.samples_per_cls- self.samples_per_task[0]/self.samples_per_cls)
            idxs = []
            class_target_size = int(target_size/num_classes)
            target_size = class_target_size*num_classes
            for cls_no in range(num_classes):
                if self.dataset_history.swap_in_targets==None or bl==False: 
                    possible_idxs = list(range(self.samples_per_cls))
                else:
                    possible_idxs = (( self.dataset_history.swap_in_targets[cls_no]).nonzero(as_tuple=True)[0])
                if len(possible_idxs) < class_target_size:
                    residue_size = class_target_size - len(possible_idxs)
                    residue_idxs =  (( self.dataset_history.swap_in_targets[cls_no]-1).nonzero(as_tuple=True)[0])
                    cls_idxs =  random.choices(residue_idxs,k=residue_size)
                    cls_idxs.extend(possible_idxs)
                else: 
                    cls_idxs = random.choices(possible_idxs,k=class_target_size )
                actual_idxs = [cls_no*self.samples_per_cls + x for x in cls_idxs]
                idxs.extend(actual_idxs)
        subset.data = [self.data[i] for i in idxs]
        subset.targets = [self.targets[i] for i in idxs]
        subset.filename = [self.filename[i] for i in idxs]
        assert(len(subset) == target_size)
        return subset

