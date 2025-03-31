"""Copied from original Muat-github

https://github.com/primasanjaya/muat-github/blob/master/preprocessing/create_PCAWG_class_info.py
Little modifications to fit this better to my data
"""

import numpy as np
import pandas as pd
import os
import pdb
from tqdm import tqdm

def create_class_info(pcawg_dir, output_dir):

    tumour_types = os.listdir(pcawg_dir)
    tumour_types = [i for i in tumour_types if len(i.split('.'))==1]
    tumour_types.sort()

    #scan all samples
    pd_allsamples = []
    for i in range(len(tumour_types)):
        all_samples = os.listdir(os.path.join(pcawg_dir, tumour_types[i]))
        #filter
        print(all_samples)
        #all_samples = [i[5:] for i in all_samples if i[0:4]=='new_']
        one_tuple = (tumour_types[i],i,len(all_samples))
        pd_allsamples.append(one_tuple)
    pd_allsamples = pd.DataFrame(pd_allsamples)
    pd_allsamples.columns = ['class_name','class_index','n_samples']
    pd_allsamples.to_csv(output_dir + 'classinfo_pcawg.csv', index=False)

def main(pcawg_dir, output_dir, choose_classes = None, preprocessed = False):
    #requirements:
    total_fold = 5

    '''
    #pcawg dir --> new pcawg directory
    pcawg_dir = 'pathto/data/'

    #output dir --> to projectdir/dataset_utils
    output_dir = '../dataset_utils/'
    '''
    #scan all samples per tumour types
    if choose_classes is None:
        tumour_types = os.listdir(pcawg_dir)
        tumour_types = [i for i in tumour_types if len(i.split('.'))==1]
        tumour_types.sort()
    else:
        tumour_types = choose_classes
        tumour_types.sort()

    #scan all samples
    print("Scanning all the samples...")
    pd_allsamples = pd.DataFrame()
    for i in tqdm(range(len(tumour_types))):
        if not preprocessed:
            all_samples = os.listdir(pcawg_dir + tumour_types[i])
            #filter
            if len(all_samples) < 20:
                continue

        elif preprocessed:
            all_samples = []
            #filter
            for sample in os.listdir(os.path.join(pcawg_dir, tumour_types[i])):
                npz = 0
                for file in os.listdir(os.path.join(pcawg_dir, tumour_types[i], sample)):
                    #print(file)
                    if file[-3:] == 'npz':
                        npz +=1
                if npz > 0:
                    all_samples.append(sample)
        pd_samp = pd.DataFrame({'samples':all_samples})
        pd_samp['nm_class'] = tumour_types[i]
        
        pd_allsamples = pd.concat([pd_allsamples, pd_samp])
    pd_allsamples.columns = ['samples','nm_class']
    print(f'rows: {len(pd_allsamples)}')

    #slicing data
    print('Slicing the data...')
    get_10slices = []
    startslice=0
    for i in tqdm(range(0,len(pd_allsamples))):
        startslice = startslice + 1    
        if startslice > total_fold:
            startslice = 1
            get_10slices.append(startslice)
        else:
            get_10slices.append(startslice)
    pd_allsamples['slices'] = get_10slices


    #create_train_val_test split,
    print('Splitting to train and val sets...')
    trainAll = pd.DataFrame()
    valAll = pd.DataFrame()

    for valfold in tqdm(range(1,total_fold+1)):
        val = pd_allsamples.loc[pd_allsamples['slices']==valfold]
        train = pd_allsamples.loc[pd_allsamples['slices']!=valfold]
        
        train['fold'] = valfold
        val['fold'] = valfold
        
        trainAll = pd.concat([trainAll,train])
        valAll = pd.concat([valAll, val])
    trainAll.to_csv('../extfiles/' + 'pcawg_train_.csv')
    valAll.to_csv('../extfiles/'+ 'pcawg_val_.csv')

    
    #create class_info
    pd_allsamples = []
    print('Creating classinfo...')
    for i in tqdm(range(len(tumour_types))):
        all_samples = os.listdir(os.path.join(pcawg_dir,  tumour_types[i]))
        #filter
        if len(all_samples) < 20:
            continue
        one_tuple = (tumour_types[i],i,len(all_samples))
        pd_allsamples.append(one_tuple)
    pd_allsamples = pd.DataFrame(pd_allsamples)
    pd_allsamples.columns = ['class_name','class_index','n_samples']
    pd_allsamples.to_csv(output_dir + 'classinfo_pcawg_.csv')

if __name__ == '__main__':
    class_info_muat_orig = pd.read_csv('../data/utils/classinfo_pcawg_.csv')
    tumour_classes = class_info_muat_orig['class_name'].values.tolist()
    path = "/csc/epitkane/projects/PCAWG20191001/data/modified_data/raw/all2patient/"
    path2 = "/csc/epitkane/projects/multimodal/data/temp/motif3"
    main(path2, "../data/utils/", tumour_classes, preprocessed=True)

