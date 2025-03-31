import os
import sys
import dask
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
from utils import load_array, load_txt_array

class PCAWG_DNABERT_Dataset(Dataset):
    
    def __init__(self, data_dir,
                block_size=5000,
                mutation_ratio='0.4-0.3-0.3-0-0',
                tumour_info_file_name = 'classinfo_pcawg_muat_orig.csv',
                split = None,
                split_file_names = ('pcawg_train_muat_orig.csv', 'pcawg_val_muat_orig.csv'),
                fold=None,
                pos = False,
                ges = False,
                muat_orig = False,
                one_hot = False,
                nembed = 1536,
                one_hot_length = 3,
                epipos = False,
                context = False,
                context_length= 1000):
        
        if not os.path.exists(data_dir):
            print(f'directory {data_dir} not exists, cannot create dataset') 
            sys.exit(1)
        self.pd_class_info = pd.read_csv(os.path.join(os.getcwd(), os.pardir, 'data', 'utils', tumour_info_file_name))
        self.one_hot = one_hot
        self.muat_orig = muat_orig
        self.fold = fold
        self.data_dir = data_dir
        self.pos = pos
        self.ges = ges
        self.one_hot_length = one_hot_length
        self.epipos = epipos
        self.context = context
        self.context_len = context_length
        self.SNV = False
        self.MNV = False
        self.indel = False
        self.SVMEI = False
        self.Normal = False 
        self.nembed =nembed

        if mutation_ratio is not None:
            self.mutation_ratio = mutation_ratio.split('-')
            self.mutation_ratio = [float(i) for i in self.mutation_ratio]

            if self.mutation_ratio[0]>0:
                self.SNV = True 
            if self.mutation_ratio[1]>0:
                self.MNV = True
            if self.mutation_ratio[2]>0:
                self.indel = True

            # following types not in use yet
            if self.mutation_ratio[3]>0:
                self.SVMEI = True
            if self.mutation_ratio[4]>0:
                self.Normal = True

        self.block_size = block_size
        #self.Embedder = Embedder()
        
        if split == None:
            data_files = os.listdir(data_dir)
        else:
            split_file = {split == 'train': split_file_names[0],  split == 'val':split_file_names[1]}.get(True, -1)
            if split == -1:
                sys.exits(f'Split {split} is incorrect. Split needs to be \'train\' or \'val\'.')

            fold_dir = os.path.join(os.getcwd(), os.pardir, 'extfiles', split_file)
            if not os.path.exists(fold_dir):
                print(f'Cannot find directory for {fold_dir}. Run the preprocess code first.')
                sys.exit(1)
            
            self.files_pd = pd.read_csv(fold_dir)

            if fold is not None:
                self.files_pd = self.files_pd[self.files_pd['fold'] == fold]
        
        path_to_tumour_info = (os.path.join(os.pardir, 'data', 'utils', tumour_info_file_name))
        
        if not os.path.exists(path_to_tumour_info):
            print(f'Cannot find {path_to_tumour_info}. Please run MuAt/create_PCAWG_classinfo.py')
            sys.exit(1)
        
        self.tumour_dict = {}
        self.tumour_dict_reverse = {}
        with open(path_to_tumour_info) as tumour_file:
            for i, line in enumerate(tumour_file):
                #header
                if i==0:
                    continue
                line = line.split(sep=',')
                self.tumour_dict[line[1]] = int(line[2]) 
                self.tumour_dict_reverse[int(line[2])] = line[1] 

    def adjust_mutation_ratio(self, row_count):
        avail_count = np.asarray(self.mutation_ratio, dtype= 'float64') * self.block_size      
            
        #Check if there are more avail_count than rows some parts of ratio
        diff = avail_count -row_count
        pos = diff>0
        avail_count1 = row_count * pos
        diff = row_count > avail_count

        # Check if there are more rows than avail count for some parts of ratio
        avail_count2 = avail_count * diff
        
        # ration with available row amounts
        avail_count3 = avail_count1 + avail_count2

        # Use as big amount as possible SNVs for block
        # and that the block is correct size 
        shadowavail_count3 = avail_count3
        shadowavail_count3[0] = row_count[0]

        # Check that block size is ok, if not fix it
        if sum(shadowavail_count3) > self.block_size:
            diff = self.block_size - sum(avail_count3) 
            shadowavail_count3[0] = diff + avail_count3[0]
            
        avail_count2 = shadowavail_count3.astype(int)

        #Handle negative values 
        if avail_count2[0]<0:

            secondmax = avail_count2[np.argmax(avail_count2)]
            avail_count2 = avail_count2 * 0.7

            avail_count = avail_count2

            diff = avail_count - row_count
            pos = diff>0
            avail_count1 = row_count * pos
            diff = row_count > avail_count

            avail_count2 = avail_count * diff
            avail_count3 = avail_count1 + avail_count2
            shadowavail_count3 = avail_count3
            shadowavail_count3[0] = row_count[0]

            if sum(shadowavail_count3) > self.block_size:
                diff = self.block_size - sum(avail_count3) 
                shadowavail_count3[0] = diff + avail_count3[0]
                
            avail_count2 = shadowavail_count3.astype(int)

        return avail_count2
    

    def __len__(self):
        if self.fold == None:
            return self.files_pd.shape[0]
        else:
            return self.files_pd[self.files_pd['fold']== self.fold].shape[0]

    def __getitem__(self, index):
        # Pick the file 
        # file needed to list the samples with histology and sample name
        row = self.files_pd.iloc[index]
        sample = row['samples'].split('.')[0]
        histology = row['nm_class']
        # Maybe file needed to tell the counts
        file_path = os.path.join(self.data_dir, histology, sample)

        count_file = pd.read_csv(os.path.join(file_path, 'count_' + sample + '.tsv.gz'), sep='\t', compression='gzip', index_col=0)
        row_count = count_file.values[0]

        if self.muat_orig:
            data_motif = np.array([]).reshape(0,1)
        elif self.one_hot:
            data_motif = np.array([]).reshape(0,self.one_hot_length,25)
        else:
            data_motif = np.array([], dtype='float32').reshape(0,self.nembed)

        if self.pos:
            data_pos = np.array([]).reshape(0, 1)
        if self.ges:
            data_ges = np.array([]).reshape(0, 1)
        if self.epipos:
            epipos_path = os.path.join(self.data_dir, "epipos", histology, sample)
            data_epipos = np.array([]).reshape(0, 50)
        if self.context:
            context_path = os.path.join(self.data_dir, "context", histology, sample)
            data_context = np.array([]).reshape(0, self.context_len)
        # mutation ratio
        if self.mutation_ratio is not None:
            avail_count = self.adjust_mutation_ratio(row_count)
        
        #print(avail_count)
        try:
            if self.SNV and avail_count[0] > 0:
                #print(os.path.join(file_path, 'SNV_' + sample + '.npz'))
                SNV_motif = dask.delayed(load_array)(os.path.join(file_path, 'SNV_' + sample + '.npz'), 'motif')
                if self.pos:
                    SNV_pos = dask.delayed(load_array)(os.path.join(file_path, 'SNV_' + sample + '.npz'), 'position')
                if self.ges:
                    SNV_ges = dask.delayed(load_array)(os.path.join(file_path, 'SNV_' + sample + '.npz'), 'GES')
                if self.epipos:
                    SNV_epipos = dask.delayed(load_txt_array)(os.path.join(epipos_path, 'SNV_' + sample + '.txt.gz'))
                if self.context:
                    SNV_context = dask.delayed(load_txt_array)(os.path.join(context_path, '1k_context_SNV_' + sample + '.txt.gz'))
                SNV_motif = dask.compute(SNV_motif)[0]
                #if avail_count[0] > SNV_motif.shape[0]:
                #    avail_count[0] = SNV_motif.shape[0]

                #print(SNV_motif)
                #print(avail_count)

                # Create list of indices to use
                data_sample_snv = np.random.choice(len(SNV_motif), size = avail_count[0], replace=False)
                if self.muat_orig:
                    SNV_motif = SNV_motif[data_sample_snv].reshape(-1, 1)
                elif self.one_hot:
                    SNV_motif = SNV_motif[data_sample_snv, :, :]
                else:
                    SNV_motif = SNV_motif[data_sample_snv, :]
                data_motif = np.vstack((data_motif, SNV_motif))

                if self.pos:
                    SNV_pos = dask.compute(SNV_pos)[0]
                    SNV_pos = SNV_pos[data_sample_snv].reshape(-1, 1)
                    data_pos = np.vstack((data_pos, SNV_pos))
                if self.ges: 
                    SNV_ges = dask.compute(SNV_ges)[0]
                    SNV_ges = SNV_ges[data_sample_snv].reshape(-1, 1)
                    data_ges = np.vstack((data_ges, SNV_ges))
                
                if self.epipos:
                    epipos_SNV = dask.compute(SNV_epipos)[0]
                    #print(f'dl: {epipos_SNV.shape}')
                    if len(epipos_SNV.shape) != 2:
                        epipos_SNV = epipos_SNV.reshape(1,50)
                    epipos_SNV = epipos_SNV[data_sample_snv,:]
                    data_epipos = np.vstack((data_epipos, epipos_SNV))
                
                if self.context:
                    context_SNV = dask.compute(SNV_context)[0]
                    #print(f'dl: {epipos_SNV.shape}')
                    if len(context_SNV.shape) != 2:
                        context_SNV = context_SNV.reshape(1,self.context_len)
                    context_SNV = context_SNV[data_sample_snv,:]
                    data_context = np.vstack((data_context, context_SNV))



            if self.MNV and avail_count[1] > 0:
                #print(os.path.join(file_path, 'MNV_' + sample + '.npz'))
                MNV_motif = dask.delayed(load_array)(os.path.join(file_path, 'MNV_' + sample + '.npz'), 'motif')
                if self.pos:
                    MNV_pos = dask.delayed(load_array)(os.path.join(file_path, 'MNV_' + sample + '.npz'), 'position')
                if self.ges:
                    MNV_ges = dask.delayed(load_array)(os.path.join(file_path, 'MNV_' + sample + '.npz'), 'GES')
                if self.epipos:
                    MNV_epipos = dask.delayed(load_txt_array)(os.path.join(epipos_path, 'MNV_' + sample + '.txt.gz'))
                if self.context:
                    MNV_context = dask.delayed(load_txt_array)(os.path.join(context_path, '1k_context_MNV_' + sample + '.txt.gz'))

                MNV_motif = dask.compute(MNV_motif)[0]
                #if avail_count[1] > MNV_motif.shape[0]:
                #    avail_count[1] = MNV_motif.shape[0]
                data_sample_mnv = np.random.choice(len(MNV_motif), size = avail_count[1], replace=False)
                if self.muat_orig:
                    MNV_motif = MNV_motif[data_sample_mnv].reshape(-1, 1)
                elif self.one_hot:
                    MNV_motif = MNV_motif[data_sample_mnv, :, :]
                else:
                    MNV_motif = MNV_motif[data_sample_mnv, :]
                data_motif = np.vstack((data_motif, MNV_motif))

                if self.pos:
                    MNV_pos = dask.compute(MNV_pos)[0]
                    MNV_pos = MNV_pos[data_sample_mnv].reshape(-1, 1)
                    data_pos = np.vstack((data_pos, MNV_pos))
                if self.ges: 
                    MNV_ges = dask.compute(MNV_ges)[0]
                    MNV_ges = MNV_ges[data_sample_mnv].reshape(-1, 1)
                    data_ges = np.vstack((data_ges, MNV_ges))

                if self.epipos:
                    epipos_MNV = dask.compute(MNV_epipos)[0]
                    if len(epipos_MNV.shape) != 2:
                        epipos_MNV = epipos_MNV.reshape(1,50)
                    epipos_MNV = epipos_MNV[data_sample_mnv,:]
                    data_epipos = np.vstack((data_epipos, epipos_MNV))
                
                if self.context:
                    context_MNV = dask.compute(MNV_context)[0]
                    #print(f'dl: {epipos_SNV.shape}')
                    if len(context_MNV.shape) != 2:
                        context_MNV = context_MNV.reshape(1,self.context_len)
                    context_MNV = context_MNV[data_sample_mnv,:]
                    data_context = np.vstack((data_context, context_MNV))

            if self.indel and avail_count[2] > 0:
                #print(os.path.join(file_path, 'indel_' + sample + '.npz'))
                indel_motif = dask.delayed(load_array)(os.path.join(file_path, 'indel_' + sample + '.npz'), 'motif')
                if self.pos:
                    indel_pos = dask.delayed(load_array)(os.path.join(file_path, 'indel_' + sample + '.npz'), 'position')
                if self.ges:    
                    indel_ges = dask.delayed(load_array)(os.path.join(file_path, 'indel_' + sample + '.npz'), 'GES')
                if self.epipos:
                    indel_epipos = dask.delayed(load_txt_array)(os.path.join(epipos_path, 'indel_' + sample + '.txt.gz'))
                if self.context:
                    indel_context = dask.delayed(load_txt_array)(os.path.join(context_path, '1k_context_indel_' + sample + '.txt.gz'))

                indel_motif = dask.compute(indel_motif)[0]
                #if avail_count[2] > indel_motif.shape[0]:
                #    avail_count[2] = indel_motif.shape[0]
                data_sample_indel = np.random.choice(len(indel_motif), size = avail_count[2], replace=False)
                if self.muat_orig:
                    indel_motif = indel_motif[data_sample_indel].reshape(-1, 1)
                elif self.one_hot:
                    indel_motif = indel_motif[data_sample_indel, :]
                else:
                    indel_motif = indel_motif[data_sample_indel, :]
                data_motif = np.vstack((data_motif, indel_motif))

                if self.pos:
                    indel_pos = dask.compute(indel_pos)[0]
                    indel_pos = indel_pos[data_sample_indel].reshape(-1, 1)
                    data_pos = np.vstack((data_pos, indel_pos))
                if self.ges: 
                    indel_ges = dask.compute(indel_ges)[0]
                    indel_ges = indel_ges[data_sample_indel].reshape(-1, 1)
                    data_ges = np.vstack((data_ges, indel_ges))
                
                if self.epipos:
                    epipos_indel = dask.compute(indel_epipos)[0]
                    if len(epipos_indel.shape) != 2:
                        epipos_indel = epipos_indel.reshape(1,50)
                    epipos_indel = epipos_indel[data_sample_indel,:]
                    data_epipos = np.vstack((data_epipos, epipos_indel))
                
                if self.context:
                    context_indel = dask.compute(indel_context)[0]
                    #print(f'dl: {epipos_SNV.shape}')
                    if len(context_indel.shape) != 2:
                        context_indel = context_indel.reshape(1,self.context_len)
                    context_indel = context_indel[data_sample_indel,:]
                    data_context = np.vstack((data_context, context_indel))
            
            if self.SVMEI and avail_count[3] > 0:
                raise NotImplementedError


            if self.Normal and avail_count[4] > 0:
                raise NotImplementedError
            
        except Exception as E:
            print(f"Something happened with {file_path}")
            raise E
        if len(data_motif) < self.block_size:
            mins = self.block_size - len(data_motif)
            if self.one_hot:
                motif_padd = np.zeros((mins, self.one_hot_length,  data_motif.shape[-1]))
            else:
                motif_padd = np.zeros((mins, data_motif.shape[-1]))
            data_motif = np.vstack((data_motif, motif_padd))
            if self.pos:
                pos_padd = np.zeros((mins, data_pos.shape[-1]))
                data_pos = np.vstack((data_pos, pos_padd))
            if self.ges:
                ges_padd = np.zeros((mins, data_ges.shape[-1]))
                data_ges = np.vstack((data_ges, ges_padd))
            if self.epipos:
                epipos_padd = np.zeros((mins, data_epipos.shape[-1]))
                data_epipos = np.vstack((data_epipos, epipos_padd))
            if self.context:
                context_padd = np.zeros((mins, data_context.shape[-1]))
                data_epipos = np.vstack((data_context, context_padd))


        #Pick the 
        # data_preprocessed = self.preprocess(df_data)
        target = np.array([self.tumour_dict[histology]])
        #if sample =='5187e77d-f412-4303-8049-11d1aa1a0235':
        #    print(f'Dataloader: {target}, {sample}')
        data = {'data': [], 
                'targets': torch.from_numpy(target),
                'sample': sample}
        if not self.muat_orig:
            data['data'].append(torch.from_numpy(data_motif).to(torch.float32))
        else:
            data['data'].append(torch.from_numpy(data_motif).to(torch.int).flatten())

        if self.pos:
            data['data'].append(torch.from_numpy(data_pos).to(torch.int).flatten())

        if self.ges:
            data['data'].append(torch.from_numpy(data_ges).to(torch.int).flatten())
        
        if self.epipos:
            data['data'].append(torch.from_numpy(data_epipos).to(torch.float32))
        
        if self.context:
            data['data'].append(torch.from_numpy(data_context).to(torch.float32))

        return data