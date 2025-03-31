import sys
import os
import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def status(msg, verbose=True, lf=True, time=True) -> None:
    """writes status message to stream

    Args:
        msg: status as a string
        verbose: boolean value which is utilised to tell if status messages are printed 
        lf: boolean value which is utilised to tell if newline char is added to the end of message 
        time: boolean value telling if the time stamp is added to the statusmessage
    """
    if verbose:
        if time:
            tstr = '[{}] '.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            tstr = ''
        sys.stdout.write('{}{}'.format(tstr, msg))
        if lf:
            sys.stdout.write('\n')
        sys.stdout.flush()

def list_files_in_dir(dir, results):
    for path in os.listdir(dir):
        full_path = os.path.join(dir, path)
        if os.path.isdir(full_path):
            list_files_in_dir(full_path, results)
        else:
            results.append(full_path)
    return results

def register_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, name=name: print(f'Gradient for {name}: {grad}'))

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param.grad).any():
                print(f'NaN gradient in {name}')
            elif torch.isinf(param.grad).any():
                print(f'Inf gradient in {name}')
            else:
                print(f'{name} gradient is OK')

def load_array(file_path, array_name):
    data = np.load(file_path)
    return data[array_name]

def load_txt_array(file_path):
    data = np.loadtxt(file_path)
    return data

def load_ckpt(ckpt_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device) 
        model_config = ckpt['model_config']
        model_state_dict= ckpt['model']#model.load_state_dict(ckpt['model'])
        #model = model.to(device)
        config = ckpt['config']
        epoch = ckpt['epoch']
        optimizer_state_dict = ckpt['optimizer']
        #config['ckpt_path'] = '/mnt/ahuttun/multimodal/models'
        return model_config, model_state_dict, config, epoch +1, optimizer_state_dict

def find_duplicates(dict_path):
    dict_ =pd.read_csv(dict_path)
    triplets = dict_.loc[:,'triplet'].values.tolist()
    exists = []

    for t in triplets:
        if t not in exists:
            exists.append(t)
        else:
            print(t)

def fix_pos_dict(dict_path, temp_dir):
    paths = []
    status('Listing all the files...', True)
    list_files_in_dir(temp_dir, paths)
    files = []
    for f in paths:
        if f.split(os.sep)[-1][:5] != 'count' and f.split(os.sep)[-1][-7:] == '.tsv.gz':
            files.append(f) 
    status(f'{len(files)} files found', True)
    motif_dict = pd.read_csv(dict_path, low_memory=False)
    seqs = motif_dict.loc[:,'chrompos'].tolist()
    token_count = len(seqs)
    new_motifs = {'chrompos':[], 'token':[]}
    for f in files:
        status(f'Processing file {f}', True)
        mutations = pd.read_csv(f, low_memory=False, sep='\t', header=0, compression='gzip')
        
        pos = mutations.loc[:,'pos'].values.tolist()
        chr = mutations.loc[:,'chrom'].values.tolist()

        pos = [np.floor(int(p)/1000000) for p in pos]
        ps = map(str, map(int,pos))
        chrom = map(str, chr)

        chrom_pos = [ch + '_' + p for ch, p in zip(chrom, ps)]

        for s in chrom_pos:
            if str(s) not in seqs:
                status(f'triplet {s} was not found', True)
                if str(s) not in new_motifs['chrompos']:
                    token_count +=1
                    new_motifs['chrompos'].append(s)
                    new_motifs['token'].append(token_count) 


    new_dict = pd.DataFrame(new_motifs)
    new_dict.to_csv(os.path.join(os.pardir, 'extfiles', 'dictChpos_missing_triplets.csv'))

def one_hot(arr, token_size):
    encoded_arr = np.zeros((arr.size, token_size), dtype=int)
    encoded_arr[np.arange(arr.size),arr] = 1
    return encoded_arr

def create_preprocess_data(data_dir, file_name ='data_preprocessing.tsv.gz', choose_classes=None, verbose=False, report_interval=200):
    if not os.path.exists(data_dir):
        sys.exit(1)

    else:
        if choose_classes is None:
            dirs = os.listdir(data_dir)
        else:
            classinfo_df = pd.read_csv(choose_classes)
            dirs = classinfo_df['class_name'].values.tolist()
        dirs = [dir for dir in dirs if os.path.isdir(os.path.join(data_dir, dir))]
        files_list = []
        n_files = 0
        for i, dir in enumerate(dirs):
            files = os.listdir(os.path.join(data_dir, dir))
            for j, file in enumerate(files):
                files_list.append(os.path.join(data_dir, dir, file))
                n_files += 1
                if n_files % report_interval == 0:
                    status(f'Listed {j+1}/{len(files)} from dir {i+1}/{len(dirs)}. Total files listed: {n_files}', verbose=True)
        df = pd.DataFrame({'path': files_list})
        df.to_csv(os.path.join(os.pardir, 'data', 'utils', file_name),index=False, sep= '\t', compression='gzip')

def fix_motif_dict(dict_path, temp_dir):
    paths = []
    status('Listing all the files...', True)
    list_files_in_dir(temp_dir, paths)
    files = []
    for f in paths:
        if f.split(os.sep)[-1][:5] != 'count' and f.split(os.sep)[-1][-7:] == '.tsv.gz':
            files.append(f) 
    status(f'{len(files)} files found', True)
    motif_dict = pd.read_csv(dict_path, low_memory=False)
    seqs = motif_dict.loc[:,'triplet'].tolist()
    token_count = len(seqs)
    new_motifs = {'triplet':[], 'triplettoken':[], 'typ': [], 'mut_type':[]}
    for f in files:
        status(f'Processing file {f}', True)
        mutations = pd.read_csv(f, low_memory=False, sep='\t', header=0, compression='gzip')
        mutations_seq = mutations.loc[:,'seq'].tolist()
        for s in mutations_seq:
            if str(s) not in seqs:
                status(f'triplet {s} was not found', True)
                if str(s) not in new_motifs['triplet']:
                    token_count +=1
                    new_motifs['triplet'].append(s)
                    new_motifs['triplettoken'].append(token_count) 
                    new_motifs['typ'].append(None)
                    new_motifs['mut_type'].append(None)

    new_dict = pd.DataFrame(new_motifs)
    new_dict.to_csv(os.path.join(os.pardir, 'extfiles', 'dictMotif_missing_triplets.csv'))

def find_faulty_files(folder):
    paths = []
    paths = list_files_in_dir(folder, paths)
    #paths =["/csc/epitkane/projects/multimodal/data/temp/DNABERT_motif1001/Breast-AdenoCA/f7fdda4f-7bf7-ede7-e040-11ac0c486e57/MNV_f7fdda4f-7bf7-ede7-e040-11ac0c486e57.tsv.gz"]
    for i, f in enumerate(paths):
        df = pd.read_csv(f, compression='gzip', sep='\t', index_col=0)
        if "Unnamed: 0" in list(df.columns):
            print(f)
        status(f'{i+1}/{len(paths)} files handled')
    print("finished")    

def generate_count_files(data_dir, use_tqdm=False):
    paths = []
    paths = list_files_in_dir(data_dir, paths)
    paths.sort()

    numpy_files = []
    count_files = []
    for i in paths:        
        if i.split('/')[-1].split('.')[-1] == 'npz':
            numpy_files.append(i)
            count_files.append('/'.join(i.split('/')[:-1]) + '/count_' + i.split('/')[-1].split('_')[-1][:-4]+ '.tsv.gz')
    
    for np_path, count in tqdm(zip(numpy_files, count_files), total=len(numpy_files), disable = not use_tqdm):
        array = np.load(np_path)['motif']
        tumour = np_path.split("/")[-3]
        if not os.path.exists(count):
            count_df = pd.DataFrame.from_dict({'SNV':[0], 'MNV':[0], 'indel':[0], 'SV/MEI':[0], 'Normal':[0]})
        else:
            count_df = pd.read_csv(count, compression='gzip', sep = '\t', index_col=0)
        mutation_type= np_path.split("/")[-1].split("_")[0]
        count_df.loc[0,mutation_type] = int(len(array))
        count_df.to_csv(count, compression='gzip', sep='\t' )
    print("done")

if __name__ =='__main__':
    #find_duplicates('/csc/epitkane/projects/multimodal/extfiles/dictMotif.csv')
    #fix_pos_dict('/csc/epitkane/projects/multimodal/extfiles/dictChpos.csv', '/csc/epitkane/projects/multimodal/data/temp/muat_orig')
    #find_faulty_files('/csc/epitkane/projects/multimodal/data/temp/DNABERT_motif1001')
    generate_count_files('/csc/epitkane/projects/multimodal/data/train_new/DNABERT_motif3_diff')