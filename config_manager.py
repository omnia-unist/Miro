import numpy as np
import yaml
import argparse
import itertools,json
import pprint
pp = pprint.PrettyPrinter(sort_dicts=True)
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--make', action='store_true')
parser.add_argument('--p', dest='preset', default=None)
parser.add_argument('-r', '--run', action='store_true')
parser.add_argument('--basefile', dest='basefile', default=None)
parser.add_argument('--modify', type=json.loads)
parser.add_argument('--filename', type=json.loads, default=['agent_name','test_set','swap','epochs','data_order'])
parser.add_argument('--path', type=str, default='/home/xinyuema/carm-extra/config')

parser.add_argument('--gpu_num', type=json.loads, default=[0,1])
parser.add_argument('--num_per_gpu', type=int, default=1)
parser.add_argument('--rank_by', type=json.loads, default=None)


args = parser.parse_args()

swap_grid_us8k = [(0.05, 5), (0.1, 10), (0.15, 15),(0.05, 2), (0.15, 6),(0.375, 15),(0.05, 1),(0.25, 5),(0.75, 15),(0.1, 1), (0.3, 3), (1, 10),(0.25, 2), (0.375, 3), (0.75, 6),(0.15, 1), (0.3, 2), (0.9, 6),(0.2, 1), (0.4, 2),(1, 5),(0.3, 1),  (0.6, 2), (0.9, 3),(0.4, 1),  (0.6, -3), (0.8, 2),(0.5, 1), (0.75, -3), (1, 2),(0.6, 1), (0.75, -5), (0.9, -3),(0.7, 1),(0.8, 1), (1, -5),(0.9,1),(1,1)]
swap_grid_dsads = [(0.1, 1), (0.3, 3), (1, 10), (0.1, 2), (0.25, 5), (0.75, 15), (0.1, 10), (0.15, 15), (0.15, 1), (0.75, 5), (0.9, 6), (0.15, 6), (0.25, 10), (0.375, 15), (0.2, 1),  (0.4, 2),  (1, 5), (0.25, 2), (0.375, 3), (0.75, 6), (0.3, 1),  (0.6, 2), (0.9, 3), (0.4, 1), (0.6, -3), (0.8, 2), (0.5, 1), (0.75, -3), (1, 2), (0.6, 1), (0.75, -5), (0.9, -3), (0.7, 1), (0.8, 1), (1, -5), (0.9, 1), (1, 1)]
cifar_sizes = [(200, 1800), (400, 1600), (600, 1400), (800, 1200), (1000, 1000), (1200, 800), (1400, 600), (1600, 400), (1800, 200)]
def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)[key]
        except yaml.YAMLError as exc:
            print(exc)

def save_yaml(path,configs):
    
    config_file = {'parameters': configs}
    with open(path,'w') as file: 
        yaml.dump(config_file,file,default_flow_style=False)

def make_file_name(configs, parameters):
    datasets = {'urbansound8k':'us8k','cifar100':'cifar', 'dailynsports':'dsads', 'shakespeare':'shake'}
    swap_base = {'random':'r', 'entropy_based':'e','gradient_based':'g','forgetting_based':'f'}
    elems = [] 
    filename = '_'
    for p in parameters: 
        if p == 'test_set' and configs[p] in datasets: 
            elems.append(datasets[configs[p]])
        elif p == 'swap': 
            if configs[p]==True:
                elems.append(swap_base[configs['swap_base']])
            else: elems.append('ns')
        elif configs[p]=='er_us8k': elems.append('er')
        elif p == 'threshold': elems.append(str(100*configs[p][0]))
        elif p == 'swap_period': elems.append(f'{(1/configs[p]*100):.1f}')
        else:
            elems.append(str(configs[p]))
    elems.append('')
    filename = filename.join(elems)
    configs['filename'] = filename

def modify_yaml(configs, keys, values):
    for i in range(len(keys)):
        if keys[i] == 'threshold':
            configs[keys[i]] = [values[i]]
        else: configs[keys[i]] = values[i]

def generate_tasks(path,gpu_list,num_per_line,rank_by=None):
    # base_command = f'python main.py --config={} --gpu_num={}'
    gpu_num =  itertools.cycle(gpu_list)
    commands = []
    counter =0    
    paths = list(os.scandir(path))
    if rank_by is not None:
        names = [path.name.split('.yml')[0].split('_') for path in paths]
        values = np.array([0]*len(names))
        print(names)
        for i in range(len(names)):
            values[i] = sum([eval(names[i][j]) for j in rank_by])
        idxs = np.argsort(values)
        sorted_paths = [paths[idxs[i]] for i in range(len(idxs))]
        print(sorted_paths)
        paths = sorted_paths
    for config in paths:
        print(config)
        if counter==0: 
            command = ''
            gpu = next(gpu_num)
        counter += 1 
        config_path = config.path 
        gpu = next(gpu_num)
        if counter < num_per_line:
            command += f'(python3 main.py --config="{config_path}" --gpu_num={gpu} &);'
        else:command += f'(python3 main.py --config="{config_path}" --gpu_num={gpu});'
        if counter == num_per_line: 
            counter = 0
            commands.append(command)
    if counter < num_per_line and counter>0: commands.append(command)
    return commands
if args.make: 
    # default values for us8k
    if args.basefile is None: 
        configs = dict()
        configs['agent_name'] = 'er_us8k'
        configs['batch_size'] = 128 
        configs['data_order'] = 'blurry2'
        configs['epochs'] = 10
        configs['mode'] = "disjoint"
        configs['model'] = "resnet18"
        configs['num_task_cls_per_task'] = [10,10]
        configs['num_workers'] = 2
        configs['rb_path'] =  'data/cl_saved_data/urbansound8k/fixed'
        configs['rb_size'] = 700 
        configs['result_save_path'] = 'results_test/us8k/'
        configs['run'] = 1 
        configs['sampling'] = 'ringbuffer' 
        configs['seed_start'] = 0
        configs['swap'] = False
        configs['swap_base'] = 'random'
        configs['test_set'] = 'urbansound8k' 
        configs['threshold'] = [1] 
        configs['swap_period'] = 1 
        configs['total_balance'] = False
        configs['test_set_path'] = '/home/xinyuema/dataset/UrbanSound8K_spec' 
        configs['filename'] = 'er_us8k_ns_10_blr2_'
    else: 
        configs = load_yaml(args.basefile)

    grid = args.modify
    fname = args.filename
    if grid or args.preset: 
        if  args.preset == 'us8k':
            key = ['threshold','swap_period']
            combinations = swap_grid_us8k
        elif  args.preset == 'dsads':
            key = ['threshold','swap_period']
            combinations = swap_grid_dsads
        elif args.preset == 'cifar':
            key = ['rb_size','st_size']
            combinations = cifar_sizes
        
        else:
            key = list(grid.keys())
            values = list(grid.values())
            combinations = list(itertools.product(*values))
        for combination in combinations: 
            modify_yaml(configs,key,combination)
            make_file_name(configs, fname)
            save_yaml(f"{args.path}/{configs['filename'][:len(configs['filename'])-1]}.yml",configs)
    else:
        make_file_name(configs, fname)
        save_yaml(f'{args.path}/sample_yaml.yml',configs)     
elif args.run: 
    import os
    parent_dir,gpu_num,num_per_gpu = args.path, args.gpu_num, args.num_per_gpu
    rank_by = args.rank_by
    configs = os.listdir(parent_dir)
    num_tasks = len(configs)
    num_per_line = len(gpu_num)*num_per_gpu
    commands = generate_tasks(parent_dir,gpu_num,num_per_line,rank_by)
    print(f'Total {num_tasks} tasks')
    for i in range(len(commands)): 
    # for i in range():
        print(f'Running command {i+1} out of {len(commands)}')
        print(commands[i])
        os.system(commands[i])
    
else:
    rb_size, total_size, batch_size, min_samples, epoch = 300, 1100, 128, 2, 30
    grid = {'threshold': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.375, 0.4, 0.5, 0.6, 0.66, 0.7,0.75,0.8, 0.9, 1], 'swap_period': [1, 2, 3, -3, -4, 4, 5,-5, 6, 8, 9, 10,15]}
    
    min_ratio = min_samples/(batch_size*(rb_size/total_size))
    print(min_ratio)
    possible_ratio = [i for i in grid['threshold'] if i >= min_ratio ]
    possible_periods = []
    for freq in grid['swap_period']:
        if (epoch)%(abs(freq)) == 0: 
            possible_periods.append(freq)
    values = [possible_ratio,possible_periods]

    combinations = list(itertools.product(*values))
    final_dict = dict()
    possible_percent = 0.01, 0.0125, 0.025, 0.05, 0.1, 0.125, 0.15, 0.2, 0,25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1
    
    final_combinations= [pair for pair in combinations if (pair[1]>0 and round(pair[0] * (1/pair[1]),4) in possible_percent) or (pair[1]<0 and round(pair[0] * (1-(1/abs(pair[1]))),4) in possible_percent)]
 
    for combination in final_combinations: 
        if combination[1]>0: percent = round(combination[0] * (1/combination[1]),4) 
        else:   percent = round(combination[0] * (1-(1/abs(combination[1]))),4)
        if percent not in final_dict: final_dict[percent] = []
        final_dict[percent].append(combination)

    pp.pprint(final_dict)
    print(len(final_combinations))




