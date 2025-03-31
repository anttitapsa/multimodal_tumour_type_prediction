"""
In this scrip, idea is to have same mutations in every data set . This needs to be done since e.g.,
there might be SVs and MEIs which are excluded in motif 101 but not motif 3 since the mutations exists addition to middle mutations 
"""
import os
import gzip
import sys
import pandas as pd
import numpy as np
from utils import list_files_in_dir, status

def filter_rows_mut(paths, paths101, continue_from=-1):
    for idx, (file, file101) in enumerate(zip(paths, paths101)):
        if idx < continue_from-1:
            continue
        print(file)
        print(file101)
        file_list = file.split('/')
        assert file_list[-1] == file101.split('/')[-1], f'Not the same file'
        df_data = pd.read_csv(file, sep='\t', compression='gzip', index_col=0, low_memory=False)
        df_101 = pd.read_csv(file101, sep='\t', compression='gzip', index_col=0, low_memory=False)
        df_data = df_data.astype({'chrom': str})
        df_101 = df_101.astype({'chrom': str})
        df_data = df_data.astype({'pos': str})
        df_101 = df_101.astype({'pos': str})

        df_data = df_data.drop_duplicates(subset=['chrom', 'pos'],ignore_index=True)
        df_101 = df_101.drop_duplicates(subset=['chrom', 'pos'],ignore_index=True)
        
        df_data['original_indices'] = df_data.index

        df_filtered = pd.merge(df_data, df_101, on = ['chrom', 'pos'], how='inner')
        df_filtered = df_filtered.drop_duplicates(subset=['original_indices'])
        removed_indices = df_data[~df_data.loc[:,'original_indices'].isin(df_filtered.loc[:,'original_indices'])].index.tolist()
        df_filtered = df_filtered.iloc[:,:12]
        df_filtered=df_filtered.loc[:, df_filtered.columns != 'original_indices']
        df_filtered.columns = ['chrom','pos','ref','alt','sample','seq','ref_seq','gc1kb','genic','exonic','strand','histology']

        assert df_101.loc[:,['chrom', 'pos']].equals(df_filtered.loc[:, ['chrom', 'pos']]), f"Something went wrong during handling files\n{file}\n{file101}"
       
        df_filtered.to_csv(file, sep='\t', compression='gzip')

        count_file = "/".join(file_list[:-1]) + '/count_' + file_list[-1].split('_')[-1]
        mut_type = file_list[-1].split('_')[0]
        count_df = pd.read_csv(count_file, sep='\t', compression='gzip', index_col = 0)
        count_df.loc[0, mut_type] = len(df_filtered)
        count_df.to_csv(count_file, sep='\t', compression='gzip')

        status(f'{idx+1}/{len(paths)} files filtered', True)

def filter_epipos(paths, ref, continue_from=-1):
    for idx, (file, ref_file) in enumerate(zip(paths, ref)):
        if idx < continue_from-1:
            continue
        print(file)
        print(ref_file)
        file_list = file.split('/')
        assert '_'.join([file.split('/')[-1][:-7].split('_')[0],  file.split('/')[-1][:-7].split('_')[-1]]) == ref_file.split('/')[-1][:-7], f'Not the same file'
        
        df_data = pd.read_csv(file, sep='\t', compression='gzip', index_col=0, low_memory=False)
        df_data = df_data.reset_index(drop=True)
        df_ref = pd.read_csv(ref_file, sep='\t', compression='gzip', index_col=0, low_memory=False)
        df_ref = df_ref.drop_duplicates()
        df_data = df_data.astype({'chrom': str, 'pos': str})
        df_ref = df_ref.astype({'chrom': str, 'pos': str})

        df_data['original_indices'] = df_data.index
        duplicate_indices = df_data.index.difference(df_data.drop_duplicates().index)
        df_data = df_data.drop_duplicates()
        
        array_path = '/' + os.path.join(*file_list[:-1], file_list[-1].split('_')[0] + '_' + file_list[-1].split('_')[-1].split('.')[0] + '.txt.gz')
        arr = np.loadtxt(array_path)

        df_filtered = pd.merge(df_data, df_ref, on = ['chrom', 'pos'], how='inner')
        df_filtered = df_filtered.drop_duplicates(subset=['original_indices'])
        removed_indices = df_data[~df_data.loc[:,'original_indices'].isin(df_filtered.loc[:,'original_indices'])].index.tolist()
        indices_to_remove = np.concatenate([duplicate_indices, removed_indices])
        
        df_filtered.loc[:, 'chrom'] = [int(i) if i != 'X' and i != 'Y' else i for i in df_filtered.loc[:, 'chrom']]
        df_filtered = df_filtered.astype({'pos': int})
        if not os.path.exists(os.path.join('../data/train_new/muat_orig_epipos', file_list[-3], file_list[-2])):
            os.makedirs(os.path.join('../data/train_new/muat_orig_epipos', file_list[-3], file_list[-2]), exist_ok=True)
        df_filtered.sort_values(by=['chrom', 'pos']).to_csv(os.path.join('../data/train_new/muat_orig_epipos', file_list[-3], file_list[-2], file_list[-1].split('_')[0] + '_' + file_list[-1].split('_')[-1]), sep='\t', compression='gzip')

        df_filtered = df_filtered.loc[:, ['chrom', 'pos']]

        df_filtered = df_filtered.sort_values(by=['chrom', 'pos'])
        order = df_filtered.index.to_numpy(dtype=int)

        count_path = os.path.join('../data/train_new/muat_orig_epipos', file_list[-3], file_list[-2], 'count' + '_' + file_list[-1].split('_')[-1])
        if not os.path.exists(count_path):
            df_mutationcount = pd.DataFrame({'SNV': 0,
                                     'MNV': 0,
                                     'indel': 0,
                                     'SV/MEI': 0,
                                     'Normal': 0}, index=[0])
        else:
            df_mutationcount = pd.read_csv(count_path, sep='\t', compression='gzip', header=0, index_col=0)
        
        df_mutationcount[ file_list[-1].split('_')[0]] = len(df_filtered)
        df_mutationcount.to_csv(count_path, sep='\t', compression='gzip')

        if len(indices_to_remove) > 0:
            arr = np.delete(arr, indices_to_remove, axis=0 )
        arr = arr[order]
        np.savetxt(array_path, arr)
        df_filtered.to_csv(file, sep='\t', compression='gzip')

        status(f'{idx+1}/{len(paths)} files filtered', True)
    
def filter_context(data_folder='/csc/epitkane/projects/multimodal/data/train_new/DNABERT_motif1001', save_folder='/csc/epitkane/projects/multimodal/data/train_new/DNABERT_1k_context'):
    paths = []
    paths = list_files_in_dir(data_folder, paths)
    paths.sort()
    arrays = [i for i in paths if i.split('/')[-1][-4:] == '.npz' ]

    base = save_folder
    for idx, array_path in enumerate(arrays):
        arr = np.load(array_path)
        A = arr['motif']
        A = A[:, :768]
        file = '1k_context_' + array_path.split('/')[-1].split('.')[0] + '.txt.gz'
        folder = os.path.join(base, array_path.split('/')[-3], array_path.split('/')[-2])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, file), A)
        status(f'{idx+1}/{len(arrays)} contex files handled', True)

def DNABERT_embed_diff(data_folder, save_folder):
    paths = []
    paths = list_files_in_dir(data_folder, paths)
    paths.sort()
    arrays = [i for i in paths if i.split('/')[-1][-4:] == '.npz' ]

    for idx, array_path in enumerate(arrays):
        arr = np.load(array_path)
        A = arr['motif']
        B = A[:, 768:]
        A = A[:, :768]

        assert B.shape[-1] == 768 and A.shape[-1] ==768
        motif = A -B
        file = array_path.split('/')[-1].split('.')[0] + '.npz'
        folder = os.path.join(save_folder, array_path.split('/')[-3], array_path.split('/')[-2])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        np.savez_compressed(os.path.join(folder, file), motif=motif, position=arr['position'], GES =arr['GES'])
        status(f'Difference of embeddings calculated for {idx+1}/{len(arrays)} file(s)', True)


if __name__ == '__main__':

    paths = []
    ref_paths = []
    
    status('Listing files...', True)
    paths = list_files_in_dir("/csc/epitkane/projects/multimodal/data/train_new/DNABERT_motif1001", paths)
    paths = [i for i in paths if (i[-3:] != 'npz' and i.split("/")[-1][:5] != 'count')]
    paths.sort()
    paths101 =[]
    paths101 = list_files_in_dir("/csc/epitkane/projects/multimodal/data/train_new/DNABERT_motif201", paths101)
    paths101 = [i for i in paths101 if (i[-3:] != 'npz' and i.split("/")[-1][:5] != 'count')]
    paths101.sort()
    status(f'{len(paths)} files listed!', True)
    filter_rows_mut(paths, paths101)
    
    '''
    ref_paths = list_files_in_dir("/csc/epitkane/projects/multimodal/data/train_new/muat_orig", ref_paths)
    ref_paths = [i for i in ref_paths if (i[-3:] != 'npz' and i.split("/")[-1][:5] != 'count')]
    samples = [i.split('/')[-1][:-7] for i in ref_paths]
    
    paths = list_files_in_dir("/csc/epitkane/projects/multimodal/data/train_new/test_epipos", paths)
    paths = [i for i in paths if (i[-7:] != '.txt.gz' and '_'.join([i.split('/')[-1][:-7].split('_')[0],  i.split('/')[-1][:-7].split('_')[-1]]) in samples)]
    samples_i = ['_'.join([i.split('/')[-1][:-7].split('_')[0],  i.split('/')[-1][:-7].split('_')[-1]]) for i in paths]

    final_ref = []
    for i in ref_paths:
        if i.split('/')[-1][:-7] not in samples_i:
            continue
        else:
            final_ref.append(i)
    paths.sort()
    final_ref.sort()
    assert len(final_ref) == len(paths), f"Difference: {len(final_ref)-len(paths)}"
    filter_epipos(paths, final_ref, continue_from=1484)
    '''
 
    status("DONE!", True)