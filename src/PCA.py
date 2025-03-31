"""
Module utilised to create PCA model from data
"""
import argparse
import numpy as np
import gc
import os
import pandas as pd
import umap
from sklearn.decomposition import IncrementalPCA
import pickle
import sys

from utils import list_files_in_dir, status

def handle_ueneven_batch(batch_size, unfinished_batch, embeddings, i, IPCA, embed_dim):
    needed = batch_size - len(unfinished_batch)
    #print(f'needed: {needed}')
    if len(embeddings[i*batch_size:,:]) < needed:
        unfinished_batch = np.concatenate([unfinished_batch,embeddings[i*batch_size:,:]], axis= 0)
    elif len(embeddings[i*batch_size:,:]) == needed:
        unfinished_batch = np.concatenate([unfinished_batch,embeddings[i*batch_size:,:]], axis= 0)
        IPCA.partial_fit(unfinished_batch)
        del unfinished_batch
        gc.collect()
        unfinished_batch = np.array([])
        unfinished_batch = unfinished_batch.reshape(0,embed_dim)
    elif len(embeddings[i*batch_size:,:]) > needed:
        unfinished_batch = np.concatenate([unfinished_batch,embeddings[i*batch_size:i*batch_size+needed,:]], axis= 0)
        IPCA.partial_fit(unfinished_batch)
        del unfinished_batch
        gc.collect()
        unfinished_batch = np.array([])
        unfinished_batch = unfinished_batch.reshape(0,embed_dim)
        unfinished_batch = np.concatenate([unfinished_batch,embeddings[i*batch_size+needed:,:]], axis= 0)

    return unfinished_batch, IPCA

def train_IPCA(paths, output_file, n_components=50, batch_size=227, embed_dim = 1536, save_interval=1, continue_from=-1, sample=0):
    IPCA = IncrementalPCA(n_components=n_components)
    unfinished_batch = np.array([])
    unfinished_batch = unfinished_batch.reshape(0, embed_dim)
    for idx, path in enumerate(paths):
        
        embeddings = np.load(path, mmap_mode='r')['motif'].astype(np.float32)
        if sample >0:
            if len(embeddings) < sample:
                size = len(embeddings)
            else:
                size = sample
            idx = np.random.choice(len(embeddings), size = size, replace=False)
            embeddings = embeddings[idx,:]
            file = os.path.join(output_file.split('/')[:-1], path.split('/')[-3], path.split('/')[-2], 'idx_' + path.split('/')[-1].split('.')[0] + '.npy')    
            np.save(file, idx, allow_pickle=True)
        size = len(embeddings)
        #print(size)
        #data_set = []
        if size > batch_size:
            i = 0
            while size > 0:
                batch = embeddings[i*batch_size:(i+1)*batch_size,:]
                if len(batch) == batch_size:
                    #data_set.append(batch)
                    i += 1
                    size -= batch_size
                    IPCA.partial_fit(batch)
                    del batch
                    gc.collect()
                else:
                    unfinished_batch, IPCA = handle_ueneven_batch(batch_size,
                                                                  unfinished_batch,
                                                                  embeddings,
                                                                  i,
                                                                  IPCA,
                                                                  embed_dim)
                    size = 0
            #print(data_set[-1].shape)
            #for embd in data_set:
            #    IPCA.partial_fit(embd)
        else:
            unfinished_batch, IPCA = handle_ueneven_batch(batch_size,
                                                          unfinished_batch,
                                                          embeddings,
                                                          0,
                                                          IPCA, 
                                                          embed_dim)
        if idx%save_interval ==0:
            with open(output_file,'wb') as f:
                pickle.dump(IPCA,f)
            status(f'IPCA saved after file index {idx}', True)
        del embeddings
        gc.collect()

    if len(unfinished_batch) != 0:
        IPCA.partial_fit(unfinished_batch)
    
    with open(output_file,'wb') as f:
            pickle.dump(IPCA,f)
    status('IPCA saved and trained', True)

def PCA_transform(paths, IPCA_path, output_path, no_components=1, continue_from=-1):
    with open(IPCA_path,'rb') as f:
        IPCA = pickle.load(f)
    for i, path in enumerate(paths):
        if continue_from > -1 and i < continue_from:
            continue
        else:
            status(f'Handling file {path}', True)
            data = np.load(path, allow_pickle=True)['motif']
            transformed = IPCA.transform(data)
            transformed = transformed[:,:no_components]
            file = os.path.join(output_path, path.split('/')[-3], path.split('/')[-2])
            if not os.path.exists(file):
                os.makedirs(file, exist_ok=True)
            file = os.path.join(file, path.split('/')[-1].split('.')[0] + '.npy')    
            np.save(file, transformed, allow_pickle=True)
            status(f'{i+1}/{len(paths)} files handled')

def orig_MuAt_token_embed(model_path, paths, output_path ):
    from muat_models import get_model, ModelConfig
    from utils import load_ckpt
    import torch

    status('Loading the embedding layer', True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config, model_state_dict, config, start_epoch, optimizer_satate_dict = load_ckpt(model_path)
    modelConfig = ModelConfig(**model_config)
    model, pos, ges, one_hot = get_model('MuAtMotif', modelConfig)
    model.load_state_dict(model_state_dict)
    model.to(device)

    token_embedding = model.token_embedding

    folder = os.path.join(output_path, path.split('/')[-3], path.split('/')[-2])
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for i, path in enumerate(paths):
        status(f'Handling file {path}', True)
        data = np.load(path, allow_pickle=True)['motif']
        data = torch.from_numpy(data).to(torch.int)
        transformed = token_embedding(data)
        
        file = os.path.join(folder, path.split('/')[-1].split('.')[0] + '.npy')
        transformed = transformed.detach().cpu().numpy()     
        np.save(file, transformed, allow_pickle=True)
        status(f'{i+1}/{len(paths)} files handled')



def UMAP_fit(paths, output_file, base, df_save_path, random_sample=False, without_pca=False,size=30, data_shape=768, same_samples="/csc/epitkane/projects/multimodal/UMAP/df_train_data_PCA_motif3_UMAP_30_samples_each_file.tsv.gz"):
    status('Processing the data...', True)
    data = np.array([])
    data = data.reshape([0, data_shape])
    label = []
    mutation_types= []
    sequences = []
    references = []
    positions = []
    chromosomes = []
    sample_names = []
    idx = None
    if same_samples != None:
        ref_sample = pd.read_csv(same_samples, sep='\t', compression= 'gzip', index_col=0)
        ref_sample = ref_sample.loc[:, ['chr', 'pos', 'mut_type', 'sample']]

    for path in paths:
        tumour = path.split('/')[-3]
        mutation_type = path.split('/')[-1].split('_')[0]
        if without_pca:
            embeddings = np.load(path)['motif']
        else:
            embeddings = np.load(path)
        orig_data_path = os.path.join(base, path.split('/')[-3], path.split('/')[-2], path.split('/')[-1].split('.')[0] + '.tsv.gz')
        print(path)
        print(orig_data_path)
        orig_data_file = pd.read_csv(orig_data_path, sep='\t', compression='gzip', low_memory=False)

        if len(embeddings) < size:
            sample_size = len(embeddings)
        else:
            sample_size = size

        if random_sample:
            idx = np.random.choice(len(embeddings), size = sample_size, replace=False)
        else:
            sample = path.split('/')[-2]
            rows_to_choose = ref_sample[(ref_sample.loc[:, 'sample'] == sample) & (ref_sample.loc[:, 'mut_type'] == mutation_type)]
            filter = [(rows_to_choose.loc[:,'chr'].astype(str).values.tolist()[i], rows_to_choose.loc[:,'pos'].astype(str).values.tolist()[i]) for i in range(0, len(rows_to_choose))]
            orig_data_file = orig_data_file[orig_data_file[['chrom', 'pos']].astype(str).apply(tuple, axis=1).isin(filter)]
            indices_to_use = orig_data_file.index
            #indices_to_use = orig_data_file[(orig_data_file.loc[:, 'chrom'].astype(str).isin(rows_to_choose.loc[:,'chr'].astype(str).values.tolist())) & (orig_data_file.loc[:, 'pos'].astype(str).isin(rows_to_choose.loc[:,'pos'].astype(str).values.tolist()))].index
            #orig_data_file = orig_data_file[(orig_data_file.loc[:, 'chrom'].astype(str).isin(rows_to_choose.loc[:,'chr'].astype(str).values.tolist())) & (orig_data_file.loc[:, 'pos'].astype(str).isin(rows_to_choose.loc[:,'pos'].astype(str).values.tolist()))]

        seq = orig_data_file.loc[:, 'seq']
        ref = orig_data_file.loc[:, 'ref_seq']
        pos = orig_data_file.loc[:, 'pos']
        chrom = orig_data_file.loc[:, 'chrom']
        del orig_data_file
        gc.collect()

        print(embeddings.shape[0])
        print(len(seq))
        #print(len(seq.iloc[idx]))
        #sys.exit(-1)
        if not random_sample:
            assert len(seq) == len(rows_to_choose)
            embeddings = embeddings[indices_to_use,:]
            sequences = sequences + seq.values.tolist()
            references = references + ref.values.tolist()
            positions = positions + pos.values.tolist()
            chromosomes = chromosomes + chrom.values.tolist()
            
        else:
            embeddings = embeddings[idx,:]
            sequences = sequences + seq.iloc[idx].values.tolist()
            references = references + ref.iloc[idx].values.tolist()
            positions = positions + pos.iloc[idx].values.tolist()
            chromosomes = chromosomes + chrom.iloc[idx].values.tolist()

        label.extend([tumour for i in range(sample_size)])
        mutation_types.extend(mutation_type for i in range(sample_size))
        sample_names.extend(path.split('/')[-2] for i in range(sample_size))
        data = np.concatenate([data, embeddings], axis=0)
    df = pd.DataFrame(data= data)
    df['labels'] = label
    df['seq'] =sequences
    df['ref'] = references
    df['pos'] = positions
    df['chr'] = chromosomes
    df['mut_type'] = mutation_types
    df['sample'] = sample_names
    df.to_csv(df_save_path, sep='\t', compression='gzip')
    status('Data processed!', True)

    status('Starting training of UMAP...', True)
    reducer = umap.UMAP( verbose=True)
    reducer.fit(data)
    with open(output_file,'wb') as f:
        pickle.dump(reducer,f)
    status('UMAP saved!', True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("function", nargs="?", choices=['train_IPCA', 'transform_data', 'fit_UMAP', 'MuAt_embed'], default='train_IPCA')
    args, sub_args = parser.parse_known_args()

    if args.function == 'train_IPCA':
        status("Training the PCA model", True)
        parser = argparse.ArgumentParser()
        parser.add_argument("tmp_dir", type=str, help="directory where the files to handle are saved")
        parser.add_argument("save_path", type=str, help="path to location where to save IPCA model")
        parser.add_argument("-c", "--no_components", default=50, type=int, help="Number of components in principal component analysis")
        parser.add_argument("-b", "--batch_size", default= 2039, type= int, help="The size of the batch used in incremental PCA")
        parser.add_argument("-e", "--embed_dim", default=1536, type= int, help="the size of the last dimension of embedding")
        parser.add_argument("-s", "--sample", default=0, type= int, help="random sample to take every file, if zero whole file is utilised")
        func_args = parser.parse_args(sub_args)

        status('Reading the files...', True)
        paths = []
        paths = list_files_in_dir(func_args.tmp_dir, paths)
        filtered_paths = [i for i in paths if i[-3:] == 'npz' ]
        status(f'{len(filtered_paths)} file(s) found!', True)

        status("Starting to train IPCA...", True)
        train_IPCA(paths=filtered_paths,
                    output_file=func_args.save_path,
                    n_components=func_args.no_components,
                    batch_size=func_args.batch_size,
                    embed_dim=func_args.embed_dim,
                    sample=func_args.sample)
    
    elif args.function == 'transform_data':
        status("Transforming the data using PCA", True)
        parser = argparse.ArgumentParser()
        parser.add_argument("tmp_dir", type=str, help="directory where the files to handle are saved")
        parser.add_argument("IPCA_path", type=str, help="Path to IPCA model")
        parser.add_argument("save_path", type=str, help="path to location where to save save transformed data")
        parser.add_argument("-c", "--no_components", default=10, type=int, help="Number of principal components to include")
        func_args = parser.parse_args(sub_args)

        status('Reading the files...', True)
        paths = []
        paths = list_files_in_dir(func_args.tmp_dir, paths)
        filtered_paths = [i for i in paths if i[-3:] == 'npz' ]
        status(f'{len(filtered_paths)} file(s) found!', True)

        status("Starting the transform...", True)
        PCA_transform(paths=filtered_paths,
                      IPCA_path=func_args.IPCA_path,
                      output_path=func_args.save_path,
                      no_components=func_args.no_components)
    
    elif args.function == 'fit_UMAP':
        status("Training the UMAP model", True)
        parser = argparse.ArgumentParser()
        parser.add_argument("tmp_dir", type=str, help="directory where the files to handle are saved")
        parser.add_argument("model_save_path", type=str, help="path where to save the UMAP model")
        parser.add_argument("full_data", type=str, help="directory where the information of data is saved")
        parser.add_argument("df_save_path", type=str, help="path where to save the data frame of th data utilised for training the UMAP")
        parser.add_argument("-w", "--without_PCA", action="store_true", help="handle files without pca transform")
        parser.add_argument("-r", "--random_sample", action="store_true", help="Use random sample")
        func_args = parser.parse_args(sub_args)

        status('Reading the files...', True)
        paths = []
        paths = list_files_in_dir(func_args.tmp_dir, paths)
        if func_args.without_PCA:
            filtered_paths = [i for i in paths if i[-3:] == 'npz']
        else:
            filtered_paths = [i for i in paths if i[-3:] == 'npy']
        status(f'{len(filtered_paths)} file(s) found!', True)


        UMAP_fit(filtered_paths, func_args.model_save_path, func_args.full_data, func_args.df_save_path, random_sample=func_args.random_sample, without_pca=func_args.without_PCA)

    elif args.function == 'MuAt_embed':
        status("Embedding the data using MuAt model token embedding layer", True)
        parser = argparse.ArgumentParser()
        # model_path, paths, output_path 
        parser.add_argument("tmp_dir", type=str, help="directory where the files to handle are saved")
        parser.add_argument("model_path", type= str, help="path to MuAt model whose token embedding layer is utilised")
        parser.add_argument("save_path", type=str, help="path to location where to save save transformed data")
        func_args = parser.parse_args(sub_args)

        status("Reading the files...", True)
        paths = []
        paths = list_files_in_dir(func_args.tmp_dir, paths)
        filtered_paths = [i for i in paths if i[-3:] == 'npz' ]
        status(f'{len(filtered_paths)} file(s) found!', True)

        orig_MuAt_token_embed(func_args.model_path, 
                              filtered_paths,
                              func_args.save_path)
        
    status('DONE!', True)