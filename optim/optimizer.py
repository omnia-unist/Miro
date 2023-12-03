
import os
import random
import torch
import power_check as pc
import math
import itertools
import numpy as np
import copy
import gc
import random 

import time
 
pre_result_path_us8k = '/home/xinyuema/carm-0117/results_test/profile/er_us8k_r_100_100_optim_full_/_er_us8k_r_100_100_optim_full_run0_optimizer.csv'
pre_result_path_cifar = '/home/xinyuema/carm-0117/results_test/profile/cifar_use_csv.csv'
class Optimizer(object):
	def __init__(self, observer, param_list, device, func=None):
		self.baseline = []
		self.scores = []
		self.tested = []
		self.observe = observer
		self.ckpt_model = None
		self.ckpt_opt = None
		self.ckpt_model_path, self.ckpt_opt_path = None, None
		self.device = device
		self.observations=None
		self.set_params(param_list)
		if func: 
			self.create_ckpt = func
		

	def construct_grid(self, configs):
		if isinstance(configs, dict):
			self.params = list(configs.keys())
   
			# For dailynsports dataset only, don't try reversing rb_size order
			if self.test_set == "dailynsports":
				configs['rb_size'] = sorted(configs['rb_size'],reverse=False)
			else:
				configs['rb_size'] = sorted(configs['rb_size'],reverse=True)
				configs['rb_size'] = sorted(configs['rb_size'],reverse=True)
			values = list(configs.values())
			self.base_grid = list(itertools.product(*values))
		else:
			# pseudo-balanced sampling
			if self.test_set=='cifar100':			
				self.base_grid=[(20000, 3000), (15000, 2000), (12000, 2000), (10000, 2000), (8000, 1000), (6000, 1000), (5000, 3000), (4000, 4000), (3000, 5000), (2500, 1000), (2000, 4000), (1500, 5000), (1000, 5000), (500, 4000)]
			elif self.test_set == 'imagenet1000': 
				if isinstance(configs,str):
					self.base_grid = eval(configs)
				else:
					self.base_grid = [(50000, 10400), (45000, 20800), (40000, 10400), (35000, 15600),(30000, 5200),(25000, 5200),(20000, 26000),(15000,26000), (10000, 15600),(5000,20800)]	
			if self.test_set == 'urbansound8k':
				self.base_grid=[(800, 600), (1000, 600), (600, 400), (1200, 800), (200, 200), (400, 800), (300, 1000), (500, 400), (100, 1000),(100,100)]
			elif self.test_set == 'dailynsports': 
				if configs == 'SMALL':
					self.base_grid=[(400,384),(200,576),(800,192),(200,768),(600,768),(400,576),(1000,100),(900,384),(700,192)]
				elif configs == 'MEDIUM': 
					self.base_grid=[(400,384),(200,576),(800,192),(600,768),(1200,192),(700,576),(500,768),(300,384),(1200,768),(1600,192)]
				else: 
					self.base_grid=[(400,384),(200,576),(800,192),(600,768),(1200,192),(1500,768),(700,576),(2000,192),(500,768),(300,384)]
		
		self.params = ['rb_size','st_size']
		self.grid=[]
		print(f'Profiler Search Space {self.base_grid}')
		# self.observations=np.full((len(self.params),len(self.grid)),-1,dtype=float)
	def score_functions(self, score_policy, navi_policy):
		scorers = {'most_efficient':self.most_efficient,
					'lowest_energy': self.lowest_energy,
					'highest_accuracy':self.highest_accuracy,
					'highest_ETA': self.highest_utility
                 	}
		navigators = {'random': self.random_navigator,
					'weighted_random': self.weighted_random_navigator,
                    'grid': self.grid_navigator
                }
		self.navi_param = navi_policy[1]
		return scorers[score_policy], navigators[navi_policy[0]]

	def set_params(self, param_list: dict):
		if "trail_duration" in param_list:
			self.trail_duration = param_list['trail_duration']
		self.log_file = param_list['log_file']
		self.result_save_path = os.path.dirname(self.log_file)
		self.test_set = param_list['test_set']
		if "start_point" in param_list: 
			self.start_point = param_list['start_point']
		else: self.start_point = 2
		if "acc_coeff" in param_list: 
			self.acc_coeff = float(param_list['acc_coeff'])
		else: self.acc_coeff = 0.5
		if "energy_coeff" in param_list: 
			self.energy_coeff = float(param_list['energy_coeff'])
		else: self.energy_coeff = 1 - self.acc_coeff
		if "cutline" in param_list: 
			self.cutline = float(param_list['cutline'])
		else: self.cutline = 0.2
		if "time_log_file" in param_list: 
			self.time_log_file =param_list['time_log_file']
		else: self.time_log_file = None
		if "data_ratio" in param_list: 
			self.data_ratio =float(param_list['data_ratio'])
		else: self.data_ratio = 1
		if "layer_freeze" in param_list: 
			self.layer_freeze =param_list['layer_freeze']
		else: self.layer_freeze = []
		if "pretrain" in param_list:
			self.pretrain = param_list['pretrain']
		else: 
			self.pretrain = True
		if "use_ckpt" in param_list:
			self.use_ckpt = param_list['use_ckpt']
			if "ckpt_size" in param_list: self.ckpt_size = param_list['ckpt_size']
			else: self.ckpt_size = 20
		else: 
			self.use_ckpt = False	
		if "seed" in param_list:
			self.seed = param_list['seed']
		else: 
			self.seed = 0
		os.makedirs(f'{self.result_save_path}/optimizer/', mode=0o777, exist_ok=True)

		self.construct_grid(param_list['configs'])
		self.scorer, self.navigator = self.score_functions(
			param_list['score_policy'], param_list['navi_policy'])
		f = open(self.log_file, 'a')
		f.write('task_id,rb_size,st_size,acc_term,energy_term,score,accuracy,energy_estimate,replay_acc,stream_acc,cls_accuracy\n')
		f.close()

	def get_params(self):
		return self.params

	def set_ckpt_model(self, ckpt_model,ckpt_opt,model_path=None, opt_path=None):
		if self.ckpt_model:
			del self.ckpt_model
		if self.ckpt_opt: del self.ckpt_opt
		self.ckpt_model = ckpt_model 
		self.ckpt_opt = ckpt_opt
	def set_base_config(self, base_config):
		self.base_config = base_config

	def lowest_energy(self, observation,cutline=0.2):
     
		# adjusting the grid
		skipped_configs = self.tested[-1].count(-1) + np.count_nonzero(observation[0]==0)
		if skipped_configs != 0: 
			cutline = self.cutline * ((len(self.scores[-1])-skipped_configs)/len(self.scores[-1]))
		else: cutline = self.cutline

		thres = max(int(len(observation[0])*cutline),1)
		f = open(self.log_file, 'a')
		filtered_configs_idxs = np.argsort(-observation[0,:])[:thres]
		filtered_configs = observation[:,filtered_configs_idxs]
		best_config_idx = filtered_configs_idxs[np.argsort(filtered_configs[1])[0]]

		scores = self.scores[-1]
		for i in range(len(observation[0])):
			if i in filtered_configs_idxs: scores[i] +=1 
			if i == best_config_idx: scores[i] +=1
		print(f'Raw Scores: {scores}')

		for i in range(len(observation[0])):
			f.write(f'{self.task_id}, {self.grid[-1][i][0]}, {self.grid[-1][i][1]}, - ,,, {scores[i]}, {observation[0,i]}, {observation[1,i]}\n')
		f.close()
		return scores
	def highest_accuracy(self, observation,cutline=None):
		
		best_config_idx = np.argsort(observation[0,:])[-1]
		scores = [1 if i == best_config_idx else 0 for i in range(len(observation[0]))]
		f = open(self.log_file, 'a')
		for i in range(len(observation[0])):
		
			f.write(f'{self.task_id},{self.grid[-1][i][0]},{self.grid[-1][i][1]},-,-,{scores[i]},{observation[0,i]},{observation[1,i]}\n')
		f.close()
		return scores
	def most_efficient(self, observation,cutline=0.2):
		cutline = self.cutline
		f = open(self.log_file, 'a')
		filtered_configs_idxs = np.argsort(-observation[0,:])[:int(len(observation[0])*cutline)]
		filtered_configs = observation[:,filtered_configs_idxs]
		med_idx = np.argsort(-filtered_configs[0,:])[int(len(filtered_configs[0])/2)]
		[med_acc,med_energy] = filtered_configs[:,med_idx]
		cutline = observation[0,filtered_configs_idxs[0:-1]] 
		acc_terms = self.acc_coeff *np.exp((observation[0]/med_acc))
		
		energy_terms = self.energy_coeff*(1-np.log(observation[1]/med_energy))
		scores = [acc_terms[i]+energy_terms[i] if i in filtered_configs_idxs else 0 for i in range(len(observation[0]))]
		for i in range(len(observation[0])):
			f.write(f'{self.task_id}, {self.grid[-1][i][0]}, {self.grid[-1][i][1]}, {acc_terms[i]}, {energy_terms[i]} , {scores[i]}, {observation[0,i]}, {observation[1,i]}\n')
		f.close()
		return scores
	def highest_utility(self, observation,cutline=0.5):
		# adjusting the grid
		skipped_configs = self.tested[-1].count(-1) + np.count_nonzero(observation[0]==0)
		if skipped_configs != 0: 
			cutline = self.cutline * ((len(self.scores[-1])-skipped_configs)/len(self.scores[-1]))
		else: cutline = self.cutline

		thres = max(int(len(observation[0])*cutline),1)
		f = open(self.log_file, 'a')
		filtered_configs_idxs = np.argsort(-observation[0,:])[:thres]
		filtered_configs = observation[:,filtered_configs_idxs]

		scores = self.scores[-1]
		for i in range(len(observation[0])):
			if i in filtered_configs_idxs: scores[i] = observation[0,i]/observation[1,i]
		print(np.argsort(scores)[0])
		best_config_idx =  np.argsort(scores)[-1]

		for i in range(len(observation[0])):
			f.write(f'{self.task_id}, {self.grid[-1][i][0]}, {self.grid[-1][i][1]}, - ,,, {scores[i]}, {observation[0,i]}, {observation[1,i]}\n')
		f.close()
		return scores
     
	def random_navigator(self):
		ratio = self.navi_param

		# set up mask
		if self.tested[-1].count(-1) == 0:
			thres = len(self.tested[-1]) - int(len(self.tested[-1])*ratio)
			self.tested[-1][:thres] = [-1]*thres
			for i in range(self.task_id): random.shuffle(self.tested[-1])
			print('RANDOM GRID')
			print(self.tested)
		terminate_cond = self.tested[-1].count(0) == 0

		if terminate_cond:
			return None, None
		while True:
			next_idx = self.tested[-1].index(0)
			if self.scores[-1][next_idx] == -1:
				break
		return next_idx, self.grid[-1][next_idx]
	def weighted_random_navigator(self):
		print(f'weighted random_navigator, ratio={self.navi_param}')
		ratio = self.navi_param
		# Initial possibility and step size at first profiling
		if not hasattr(self, 'grid_possibility'):
			self.grid_possibility = []
		# Calibrate possibility
		if self.tested[-1].count(-1) == 0:
			if len(self.grid_possibility)==0: 
				self.grid_possibility.append([1/len(self.base_grid)]*len(self.base_grid))
			else: 
				print('Grid Calibration')
				if self.cutline == 1: 
					self.grid_possibility.append([1/len(self.base_grid)]*len(self.base_grid))
				# obtain tested configs
				tested_idxs = []
				for i in range(len(self.last_observations[0])):
					if self.last_observations[0, i] != -1: tested_idxs.append(i) 
				tested_idxs = np.array(tested_idxs)
				# 1. lower 1/4 accuracy 
				acc_thres = max(int(len(tested_idxs)*0.25),1)
				acc_victim_idxs = np.argsort(self.last_observations[0,tested_idxs])[:acc_thres]
				acc_victim_idxs = tested_idxs[acc_victim_idxs]
    
				# 2. lower 1/2 utility from filtered 
				# lower scores
				thres = max(int(len(tested_idxs)/2),1)
				scores = self.scores[-2]

				scores = np.array(scores)
				score_victim_idxs =  np.argsort(-scores)[int(thres/2)+1:thres]
				print(f'{[self.last_observations[0, i] for i in score_victim_idxs]}')
				
				# combine victims 
				victim_idxs =  list(set(np.concatenate((score_victim_idxs,acc_victim_idxs)))) # union of the two
				reward_idxs = [i for i in tested_idxs if i not in victim_idxs]
				# reduce probalibility of the victims 
				n_possibilities = [i for i in self.grid_possibility[-1]]
				returned_tickets = 0
				for idx in victim_idxs: 	
					# Make Togglable
					ticket = n_possibilities[idx]*0.8
					returned_tickets += ticket
					n_possibilities[idx] -= ticket
				reward_ticket = returned_tickets/len(reward_idxs)
				for idx in reward_idxs: 	
					n_possibilities[idx] += reward_ticket
				# weighted sampling from the basegrid 
				self.grid_possibility.append(n_possibilities)
				
			# mask unselected idxs 
			sel_idxs = np.random.choice(range(len(self.base_grid)), size=int(len(self.base_grid)*ratio), p=self.grid_possibility[-1],replace=False)
			for i in range(len(self.tested[-1])):
				if i not in sel_idxs: self.tested[-1][i] = -1
			print('WEIGHTED RANDOM GRID')
			print(self.tested[-1])
		terminate_cond = self.tested[-1].count(0) == 0
		if terminate_cond:
			return None, None
		while True:
			next_idx = self.tested[-1].index(0)
			if self.scores[-1][next_idx] == -1:
				break
		return next_idx, self.grid[-1][next_idx]

	def grid_navigator(self):
		terminate_cond = (0) not in self.tested[-1]  #exhaustive search

		if terminate_cond:
			return None, None
		next_idx = self.tested[-1].index(0)+self.tested[-1].count(0)-1
  
		return next_idx, self.grid[-1][next_idx]

	def prep_profiler(self):
		# assume classes_so_far is known
		if self.test_set == 'cifar100':
			cls_so_far = int(10*(self.task_id-1))
			max_replay_size = cls_so_far * 5000
			max_stream_size = 5000
		elif self.test_set == 'imagenet1000': 
			max_replay_size, max_stream_size = 50000, 26000
		elif self.test_set == 'urbansound8k':
			stream_sizes = [374, 1000, 1000, 429, 1000, 1000, 1000, 1000, 929, 1000]
			max_replay_size = 3803 + 1000 * (self.task_id-6)
			max_stream_size = stream_sizes[self.task_id-1]
	
		elif self.test_set == 'dailynsports':
			cls_so_far = int(self.task_id-1)
			stream_sizes = [768,768,768,768,768,768,768,768,768,384]
			max_replay_size = sum(stream_sizes[:self.task_id-1])
			max_stream_size = stream_sizes[self.task_id-1]
		elif self.test_set == 'tiny_imagenet':
			cls_so_far = int(10*(self.task_id-1))
			max_replay_size = cls_so_far * 5000
			max_stream_size = 5000
		
  		# For other datasets -> Not implemented
		else:
			print("optim/optimizer.py prep_profiler is not implemented for this dataset")
			import sys
			sys.exit()
  
		grid = []
		for (r,s) in self.base_grid: 
			if r >max_replay_size:
				r = max_replay_size
			elif s > max_stream_size: 
				s = max_stream_size
			if (r,s) not in grid:
				grid.append((r,s))
		self.grid.append(grid)
		score = [-1]*len(grid)
		tested = [0]*len(grid)
		self.scores.append(score)
		self.tested.append(tested)
		self.last_observations=self.observations
		self.observations=np.full((len(self.params),len(self.grid[-1])),-1,dtype=float)
		print(f'TASK {self.task_id}, grid size={len(grid)}: {grid}')	
  
	def find_best_config(self, task_id): 
				
		self.task_id = task_id    		
		self.prep_profiler()
	
		best_config, best_score = None, 0
		cnt=0
		while True:
			next_idx, next_config = self.navigator()
			if next_config == None:
				break
			if next_config == self.base_config:
				self.scores[-1][next_idx] = 1
				continue
			test_model= None
			test_opt, test_lr_scheduler = None,None
			print(f'[{cnt+1}/{len(self.grid[-1])}] Profile', end=' ')
			print(next_config)      
			observation = self.observe(next_config, test_model,test_opt,test_lr_scheduler, self.trail_duration,pretrain=self.pretrain)  #debug
       
			del test_model, test_opt, test_lr_scheduler
			gc.collect()
			if use_csv:
				if self.test_set == 'urbansound8k':
					self.observations[:,next_idx] = [observation['acc'],list(self.pre_results.loc[(self.pre_results['task_id']==task_id) & (self.pre_results[" rb_size"]==next_config[0]) & (self.pre_results[" st_size"]==next_config[1])][" energy_estimate"])[0]]
				else: 
					self.observations[:,next_idx] = [observation['acc'], next_config[0]+next_config[1]]
			else:
				self.observations[:,next_idx] = [observation['acc'],observation['energy_estim']]
			cnt+=1
			print(f'Profiler accuracy: {next_config} -> {list(self.observations[:,next_idx])}')

			self.tested[-1][next_idx] = 1
		if self.time_log_file: time_st = time.perf_counter()
		self.scores[-1] = self.scorer(self.observations)
		best_idx = np.argsort(self.scores[-1])[-1]
		best_config = self.grid[-1][best_idx]
		best_score = self.scores[-1][best_idx]
		print('BEST CONFIG: ', end='')
		print(best_config,end='\n\n\n')
		best_config_as_dict = {self.params[i]: best_config[i]
                         for i in range(len(self.params))}
		if self.time_log_file: 
			with open(self.time_log_file,'a') as f: f.write(f'{self.task_id},-,Optimizer,Scorer,{time.perf_counter()-time_st}\n')
		return best_config_as_dict, best_score

def set_opt_for_profiler(test_set,model,num_epochs):
	if test_set == "cifar100":
		opt = torch.optim.SGD(model.parameters(), lr=2.0, momentum=0.9, weight_decay=0.00001)
		
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
								opt, [49,63], gamma=0.2
							)
		if num_epochs > 80:
			lr_change_point = list(range(0,num_epochs,40))
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, lr_change_point, gamma=0.25)
	elif test_set == 'urbansound8k':
		opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001,momentum=0.9,weight_decay=0.001)
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
								opt, [30,60], gamma=2
							)
	elif test_set in ["tiny_imagenet", "imagenet100", "imagenet1000"]:
		if num_epochs <=5:
			opt = torch.optim.SGD(model.parameters(), lr=2.0, momentum=0.9,  weight_decay=0.00001) #imagenet
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [20,30,40,50], gamma=0.20) #imagenet
		else:
			opt = torch.optim.SGD(model.parameters(), lr=2.0, momentum=0.9,  weight_decay=0.00001) #imagenet
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [20,30,40,50], gamma=0.20) #imagenet
	elif test_set == 'audioset':
		opt= torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-08,amsgrad=True,
									weight_decay=0)
		lr_scheduler = None
	elif test_set == "dailynsports":
		opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
		lr_scheduler = None
	else:
		raise f"No pre-set optimizer for {test_set}"
	return opt, lr_scheduler
