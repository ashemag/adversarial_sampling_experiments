import numpy as np
import time
from tqdm import tqdm
from tqdm import trange


import os
import sys

from collections import OrderedDict

def main():

    # for e in range(10):
    #     for i in tqdm(range(10),file=sys.stdout):
    #         time.sleep(0.1)
    #
    #     print("something")

    # delta = np.array([1,2,3,4])
    # delta = np.reshape(delta,(1,-1))
    # out = np.mean(delta) * np.ones_like(delta)

    d = OrderedDict({"current_epoch":0,"train_acc":500,"test_acc":600,"fda":400,"fdafd":500,"fdafdafda":40,"lala":400})
    od = OrderedDict(d.items())

    print("type: ",type(od))

    if type(od)== OrderedDict:
        print("laala ")

    print(d.keys())
    print(d.values())
    #print(od.keys())


if __name__ == '__main__':

    main()


    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # saved_models_dir = os.path.join(ROOT_DIR, 'saved_models/test_simple/saved_models_train')
    # model_path = os.path.join(saved_models_dir, 'model_40')
    #
    # print("root: ",ROOT_DIR)
    # print(model_path)

    #main()


