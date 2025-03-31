"""
Script to order chromatin state latents resulted from epicVAE by samples and tumour types
"""

import os
import sys
import gzip
import pandas as pd    
import numpy as np
import time
from utils import status

def Order_epipos_data(latent_file,
                        epigenetic_file,
                        class_info='/csc/epitkane/projects/multimodal/data/utils/classinfo_pcawg_muat_orig.csv',
                        save_dir= '/csc/epitkane/projects/multimodal/data/test_epipos',
                        ref_dir= '/csc/epitkane/projects/multimodal/data/temp/muat_orig',
                        indel_file = False):

    status('Reading the data...', True)
    tumour_type_index = 14 if indel_file else 13
    epipos =  pd.read_csv(latent_file,
                            header= None)
    epipos_order = pd.read_csv(epigenetic_file,
                                compression='gzip',
                                sep ='\t',
                                low_memory=False,
                                skipinitialspace=True,
                                header=None,
                                usecols=[3,4,8,tumour_type_index],
                                chunksize=1000000)
    cancers = pd.read_csv(class_info, header=0, low_memory=False).loc[:, 'class_name'].values.tolist()

    print('Start ordering {}'.format(latent_file))
    start_total = time.time()

    for i, order in enumerate(epipos_order):
        status(f'Chunck number: {i+1}', True)
        order = order[order.loc[:,tumour_type_index].isin(cancers)]
        order.loc[:,3] = order.loc[:,3].apply(lambda x: str(x[3:]))

        for tumour_type in cancers:
            start = time.time()
            sample_names = [i for i in os.listdir(os.path.join(ref_dir, tumour_type)) if os.path.isdir(os.path.join(ref_dir, tumour_type, i))]

            for sample in sample_names:
                correct_rows = order[(order.loc[:,8]==sample) & (order.loc[:,tumour_type_index]==tumour_type)]
                correct_rows = correct_rows.astype(str)
                folder = os.path.join(save_dir, tumour_type, sample)
                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)

                if indel_file:
                    try:
                        # indel files 
                        if os.path.exists(os.path.join(ref_dir, tumour_type, sample, f'indel_{sample}.tsv.gz')):
                            indel = pd.read_csv(os.path.join(ref_dir, tumour_type, sample, f'indel_{sample}.tsv.gz'),
                                            compression='gzip',
                                            sep='\t',
                                            low_memory= False)
                            indel = indel.loc[:,['chrom', 'pos']]
                            indel = indel.astype(str)
                            indel_rows = correct_rows[(correct_rows.loc[:,3].isin(indel.loc[:,'chrom'])) & (correct_rows.loc[:,4].isin(indel.loc[:,'pos']))]
                            if indel_rows.shape[0] != 0:
                                indel_indices = indel_rows.index
                                indel_data = epipos.iloc[indel_indices, :]

                                save_data_and_order(path = os.path.join(folder, f'indel_{sample}.txt.gz'),
                                                    order_path = os.path.join(folder, f'indel_order_{sample}.tsv.gz'),
                                                    data_df = indel_data,
                                                    data_rows=indel_rows)
                    except Exception as E:
                        sys.stderr.write(f'Something happened on indel file with sample {sample}, and tumour type {tumour_type}\n')
                        raise E 
                
                
                # Filter SNV and MNV files based on the files in muat_orig folder 
                # SNV files
                #else:
                try:
                    if os.path.exists(os.path.join(ref_dir, tumour_type, sample, f'SNV_{sample}.tsv.gz')):
                        SNV = pd.read_csv(os.path.join(ref_dir, tumour_type, sample, f'SNV_{sample}.tsv.gz'),
                                        compression='gzip', 
                                        sep='\t',
                                        low_memory=False)
                        SNV = SNV.loc[:,['chrom', 'pos']]
                        SNV = SNV.astype(str)
                        SNV_rows = correct_rows[(correct_rows.loc[:,3].isin(SNV.loc[:,'chrom'].values.tolist())) & (correct_rows.loc[:,4].isin(SNV.loc[:,'pos']).values.tolist())]
                        
                        if SNV_rows.shape[0] != 0:
                            SNV_indices = SNV_rows.index
                            SNV_data = epipos.iloc[SNV_indices, :]
                            
                            save_data_and_order(path = os.path.join(folder, f'SNV_{sample}.txt.gz'),
                                                order_path = os.path.join(folder, f'SNV_order_{sample}.tsv.gz'),
                                                data_df = SNV_data,
                                                data_rows=SNV_rows)
                except Exception as E:
                    sys.stderr.write(f'Something happened on SNV file with sample {sample}, and tumour type {tumour_type}\n')
                    raise E
                
                try:
                    # MNV files 
                    if os.path.exists(os.path.join(ref_dir, tumour_type, sample, f'MNV_{sample}.tsv.gz')):
                        MNV = pd.read_csv(os.path.join(ref_dir, tumour_type, sample, f'MNV_{sample}.tsv.gz'),
                                        compression='gzip',
                                        sep='\t',
                                        low_memory= False)
                        MNV = MNV.loc[:,['chrom', 'pos']]
                        MNV = MNV.astype(str)
                        MNV_rows = correct_rows[(correct_rows.loc[:,3].isin(MNV.loc[:,'chrom'])) & (correct_rows.loc[:,4].isin(MNV.loc[:,'pos']))]
                        if MNV_rows.shape[0] != 0:
                            MNV_indices = MNV_rows.index
                            MNV_data = epipos.iloc[MNV_indices, :]

                            save_data_and_order(path = os.path.join(folder, f'MNV_{sample}.txt.gz'),
                                                order_path = os.path.join(folder, f'MNV_order_{sample}.tsv.gz'),
                                                data_df = MNV_data,
                                                data_rows=MNV_rows)
                except Exception as E:
                    sys.stderr.write(f'Something happened on MNV file with sample {sample}, and tumour type {tumour_type}\n')
                    raise E 
                
            end = time.time()
            status(f'Ordering tumour type {tumour_type} took {time.strftime("%H:%M:%S",time.gmtime(end-start))} hours, total utilised time: {time.strftime("%H:%M:%S",time.gmtime(end-start_total))} hours', True)

def save_data_and_order(path, order_path, data_df, data_rows) -> None:
    if not os.path.exists(path):
        np.savetxt(path, data_df.to_numpy())
        data_rows = data_rows.loc[:, [3,4]].rename(columns={3:'chrom',4:'pos'})
        data_rows.to_csv(order_path, compression='gzip', sep='\t')
    else:
        data_read = np.loadtxt(path)

        if data_read.ndim == 1:
            data_read = data_read.reshape((1, len(data_read)))
        
        data_read = np.concatenate((data_read, data_df.to_numpy()), axis=0)
        np.savetxt(path, data_read)

        data_read_order = pd.read_csv(order_path, compression = 'gzip', sep = '\t', index_col=0)
        data_rows = data_rows.loc[:, [3,4]].rename(columns={3:'chrom',4:'pos'})
        data_read_order = pd.concat([data_read_order, data_rows], axis=0)
        data_read_order.to_csv(order_path, compression='gzip', sep='\t')



if __name__ == '__main__':
    indel = False
    if indel:
        Order_epipos_data(latent_file="/csc/epitkane/projects/multimodal/data/raw_epipos/new_series_3__ChromHMM_epigenome_for_indels_problematic_excluded_exclude_missing.bed.gz_latent.csv",
                            epigenetic_file='/csc/epitkane/projects/EpicSV/data/VAE_inputs/chromhmm/Indel_files/ChromHMM_epigenome_for_indels_problematic_excluded__full_annotation_final_missing_excluded.bed.gz',
                            class_info='/csc/epitkane/projects/multimodal/data/utils/classinfo_pcawg_muat_orig.csv',
                            save_dir= '/csc/epitkane/projects/multimodal/data/test_epipos',
                            ref_dir= '/csc/epitkane/projects/multimodal/data/temp/muat_orig',
                            indel_file=True)
        for i in range(1,4):
            n = i
            Order_epipos_data(latent_file="/csc/epitkane/projects/multimodal/data/raw_epipos/new_series_3__ChromHMM_epigenome_for_indels_problematic_excluded_{}_exclude_missing.bed.gz_latent.csv".format(n),
                                epigenetic_file='/csc/epitkane/projects/EpicSV/data/VAE_inputs/chromhmm/Indel_files/ChromHMM_epigenome_for_indels_problematic_excluded_{}_full_annotation_final_missing_excluded.bed.gz'.format(n),
                                class_info='/csc/epitkane/projects/multimodal/data/utils/classinfo_pcawg_muat_orig.csv',
                                save_dir= '/csc/epitkane/projects/multimodal/data/test_epipos',
                                ref_dir= '/csc/epitkane/projects/multimodal/data/temp/muat_orig',
                                indel_file=True)
        
    else:
        Order_epipos_data(latent_file="/csc/epitkane/projects/multimodal/data/raw_epipos/new_series_3__ChromHMM_epigenome_for_SNV_v2_problematic_excluded_exclude_missing.bed.gz_latent.csv",
                            epigenetic_file='/csc/epitkane/projects/EpicSV/data/VAE_inputs/chromhmm/SNV_files/ChromHMM_epigenome_for_SNV_v2_problematic_excluded__full_annotation_final_missing_excluded.bed.gz',
                            class_info='/csc/epitkane/projects/multimodal/data/utils/classinfo_pcawg_muat_orig.csv',
                            save_dir= '/csc/epitkane/projects/multimodal/data/test_epipos_fix',
                            ref_dir= '/csc/epitkane/projects/multimodal/data/temp/muat_orig')
        for i in range(1,9):
            n = i
            Order_epipos_data(latent_file="/csc/epitkane/projects/multimodal/data/raw_epipos/new_series_3__ChromHMM_epigenome_for_SNV_v2_problematic_excluded_{}_exclude_missing.bed.gz_latent.csv".format(n),
                                epigenetic_file='/csc/epitkane/projects/EpicSV/data/VAE_inputs/chromhmm/SNV_files/ChromHMM_epigenome_for_SNV_v2_problematic_excluded_{}_full_annotation_final_missing_excluded.bed.gz'.format(n),
                                class_info='/csc/epitkane/projects/multimodal/data/utils/classinfo_pcawg_muat_orig.csv',
                                save_dir= '/csc/epitkane/projects/multimodal/data/test_epipos_fix',
                                ref_dir= '/csc/epitkane/projects/multimodal/data/temp/muat_orig')
    status('Finished!', True)