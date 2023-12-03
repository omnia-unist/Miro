import sys
import numpy as np
import torch
import yaml
import argparse
import os

import numpy as np
import time
import tracemalloc


from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
from types import SimpleNamespace
import warnings

JTOP=False

from evaluation import  test_cifar100,  test_imagenet1000, test_tinyimagenet,  test_urbansound8k,  test_dailynsports

def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)[key]
        except Exception as exc:
            print(exc)

def save_yaml(path, config):
    with open(path,'w') as file:
        yaml.dump(config,file)

def main():
    
    start_t = time.time()
    tracemalloc.start()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='config/ori/exp_r.yaml')
    parser.add_argument('--gpu_num', type=int, default=0)

    args = parser.parse_args()
    print("\n==========================================================\n")
    print(str(args.config))

    params = load_yaml(args.config)
    final_params = SimpleNamespace(**params)
    if final_params.jetson: 
        import power_check as pc
        pc.printFullReport(pc.getDevice())
        pl = pc.PowerLogger(interval=2)
        final_params.power_log = pl
        pl.start()
        if JTOP:
            from jtop_logger import JtopLogger
            pathname = final_params.result_save_path + '/'+ final_params.filename+'_util.log'
            jl = JtopLogger(interval=1,path=pathname)
            jl.start()
        time.sleep(10)
        pl.recordEvent(name='Process Start')
        

    final_params.gpu_num = args.gpu_num
    # MAKE separate folder for each config
    os.makedirs(final_params.result_save_path + '/'+ final_params.filename+'/', mode = 0o777, exist_ok = True)
    final_params.result_save_path = str(final_params.result_save_path)+'/'+str(final_params.filename)+'/'
    save_yaml( final_params.result_save_path+'config.yaml',params)

    print(final_params.result_save_path + '/'+ final_params.filename + 'log.txt')
    print(final_params.result_save_path + '/'+ final_params.filename + 'run0_accuracy.txt')
    sys.stdout = open(final_params.result_save_path + '/'+ final_params.filename + 'log.txt','w')

    warnings.filterwarnings("ignore")

    print("\n==========================================================\n")
    print(final_params.result_save_path)
    print("\n==========================================================\n")

    exp_dataset = {
        'cifar100' : test_cifar100,
        'imagenet1000' : test_imagenet1000,
        'tiny_imagenet' : test_tinyimagenet,
        'urbansound8k' : test_urbansound8k,
        'dailynsports': test_dailynsports
    }

    exp_dataset[final_params.test_set].experiment(final_params)


    # time check
    print("---{}s seconds---\n".format(time.time()-start_t))

    if final_params.jetson:
        time.sleep(10)
        pl.stop()
        if JTOP: jl.stop()
        
        pl.showDataTraces(filename=final_params.filename)
        events = pl.evenLog
        f1 = open(f'{final_params.results_save_path}/test/event_log.csv','a')
        f1.write(str(events))
        f1.close()
        nodename =pl._nodes[0][0]
        pl.showMostCommonPowerValue(nodeName=nodename,filename=final_params.filename)
        print(str(pl.eventLog))
        pc.printFullReport(pc.getDevice())
    print('All End')

if __name__ == '__main__':
    # CUDA MULTIPROCESSING
    main()