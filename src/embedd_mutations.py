"""Module for embedding the mutation data using DNABERT-2 

DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2
See README to setup the working environment.

Author: Antti Huttunen, 15.5.

"""

import argparse
import torch
import os
import numpy as np
import pandas as pd
from utils import status, one_hot, list_files_in_dir

# If you use this class in dataloader during the training, remember to freeze the DNABERT
# That's because backpropagation will update also DNABERT's  parameters otherwise
class Embedder:
    def __init__(self, verbose =True, muat_orig = False, onehot = False):


        if muat_orig:
            self.dictMotif = pd.read_csv(os.path.join(os.pardir, 'extfiles', 'dictMotif_orig.csv'),low_memory=False)

        elif onehot:
            self.mutation_tokens = pd.read_csv(os.path.join(os.pardir, 'extfiles', 'one_hot_mutationdict.tsv'), sep='\t', header=0)
        
        else:
            from transformers import AutoTokenizer, AutoModel
            import einops
            import peft
            import omegaconf
            import evaluate
            import accelerate
            #import triton

            if torch.cuda.is_available():
                status('CUDA AVAILABLE', verbose)
                self.device = torch.device('cuda')
            else:
                status('CUDA NOT AVAILABLE', verbose)
                raise Exception("DNABERT 2 can only be run in GPU. Currently GPU is not available.")
        
            status('Downloading DNABERT tokenizer...', verbose=verbose)
            self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remsote_code=True)

            status('Downloading DNABERT-2-117M model...', verbose=verbose)
            model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

            status('Freezing DNABERT-2-117M...', verbose=verbose)
            for param in model.parameters():
                param.requires_grad = False

            self.model = model.to(self.device)

        self.dictChpos = pd.read_csv(os.path.join(os.pardir, 'extfiles', 'dictChpos.csv'),index_col=0,low_memory=False)
        self.dictGES = pd.read_csv(os.path.join(os.pardir, 'extfiles', 'dictGES.csv'),index_col=0,low_memory=False)
    
    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def embed_motif(self, seq, ref, batch_size=20):
        """Embed the sequence and reference sequence

        Args:   seq: List(str) mutation sequence
                ref: List(str) reference sequence  
        """
        '''
        seq_tokenization = self.tokenizer(seq, return_tensors = 'pt')["input_ids"]
        ref_tokenization = self.tokenizer(ref, return_tensors = 'pt')["input_ids"]
        '''
        embed = []

        if batch_size > len(seq) or batch_size<=0:
            batch_size =len(seq)
        # split the seq and ref batches 
        seq_batches = [seq[i:i+batch_size] if i+batch_size < len(seq) else seq[i:len(seq)] for i in range(0,len(seq), batch_size)]
        ref_batches = [ref[i:i+batch_size] if i+batch_size < len(ref) else seq[i:len(ref)] for i in range(0,len(ref), batch_size)]

        for idx, (s, r) in enumerate(zip(seq_batches, ref_batches)):
            
            #print(f'S: {s}\nR: {r}')
            s = [str(seq) for seq in s]
            r = [str(ref) for ref in r]
            seq_tokenization = self.tokenizer.batch_encode_plus(s, padding=True, return_attention_mask=True, return_tensors = 'pt')
            ref_tokenization = self.tokenizer.batch_encode_plus(r, padding=True, return_attention_mask=True, return_tensors = 'pt')

            seq_tokenization = {k: v.to(self.device) for k, v in seq_tokenization.items()}
            ref_tokenization = {k: v.to(self.device) for k, v in ref_tokenization.items()}

            #print(len(seq_tokenization['input_ids']))
            #print(seq_tokenization['input_ids'])

            seq_embed = self.model(**seq_tokenization)[0]
            ref_embed = self.model(**ref_tokenization)[0]

            assert not torch.isnan(seq_embed).any()
            assert not torch.isinf(seq_embed).any()
            assert not torch.isnan(ref_embed).any()
            assert not torch.isinf(ref_embed).any()
            '''
            seq_embed =  self.model(seq_tokenization.to(self.device))[0]
            ref_embed = self.model(ref_tokenization.to(self.device))[0]
            '''
            seq_tokenization = {k: v.detach() for k, v in seq_tokenization.items()}
            ref_tokenization = {k: v.detach() for k, v in ref_tokenization.items()}
            seq_embed = seq_embed.detach().cpu()
            ref_embed = ref_embed.detach().cpu()
            torch.cuda.empty_cache()

            #max_pooling
            seq_embed = torch.max(seq_embed, dim=1)[0]
            ref_embed = torch.max(ref_embed, dim=1)[0]

            assert seq_embed.shape[-1] == 768
            assert ref_embed.shape[-1] == 768
            
            embed.append(torch.concat([seq_embed, ref_embed], -1).detach().to('cpu')) 
            torch.cuda.empty_cache()
            #print(embed[-1].shape)
            #print(f'{idx+1} batch handled')
        return torch.cat(embed, dim=0)

    def encode_motif_old(self, triplettoken):
        triplettoken = [str(t) for t in triplettoken]
        triplettoken = pd.DataFrame(data={'triplet': triplettoken})
        mergetriplet = triplettoken.merge(self.dictMotif, left_on='triplet', right_on='triplet', how='left')
        assert not mergetriplet.loc[:,'triplettoken'].isnull().values.any(), f'{triplettoken.to_string()}\n\n{mergetriplet.to_string()}'
        return torch.tensor(mergetriplet.loc[:,'triplettoken'])

    def encode_pos_old(self, chr, pos):
        """
        Args:   chr: list
                pos: list
        """
        #pos = pd.DataFrame(data= map(int, pos))
        #chr = pd.DataFrame(data= chr)

        pos = [np.floor(int(p)/1000000) for p in pos]
        ps = map(str, map(int,pos))
        chrom = map(str, chr)

        chrom_pos = [ch + '_' + p for ch, p in zip(chrom, ps)]
        chrom_pos = pd.DataFrame(data={'chrompos': chrom_pos})
        #chrom_pos = pd.DataFrame(chrom_pos)
        #chrom_pos = chrom_pos.set_axis(['chrompos'], axis=1)
        mergechrompos = chrom_pos.merge(self.dictChpos, left_on='chrompos', right_on='chrompos', how='left')
        assert not (mergechrompos.loc[:,'token'] == 0).any()
        assert not (mergechrompos.loc[:,'token'] < 0).any()
        assert not (mergechrompos.loc[:,'token'] > 2915).any()
        return torch.tensor(mergechrompos.loc[:,'token'])
    
    def encode_ges_old(self, G, E, S):

        Ges = [str(g) + '_' + str(e) + '_' +str(s) for g,e,s in zip(G,E,S)]
        Ges = pd.DataFrame(data={'ges':Ges})
        mergeges = Ges.merge(self.dictGES, left_on='ges', right_on='ges', how='left')
        
        return torch.tensor(mergeges.loc[:,'token'])

    def encode_motif_one_hot(self, seqs, motif_len=3, max_token=25):
        encoded = np.zeros((len(seqs), motif_len, max_token))
        for i, s in enumerate(seqs.values.tolist()):
            motif_seq = pd.DataFrame({'sequence': list(str(s))})
            motif_seq = motif_seq.merge(self.mutation_tokens, left_on='sequence', right_on='mutation', how='left')
            #print("token:")
            #print(motif_seq.loc[:,'token'].values)
            #print("SEQ:")
            #print(motif_seq.loc[:,'sequence'].values)
            encoded[i,:, :] = one_hot(motif_seq.loc[:,'token'].values, max_token)
        return torch.from_numpy(encoded)
    

def create_embedding(temp_dir, report_interval =1, continue_from = -1, file = None, muat_orig = False, motif_one_hot = False, motif_len=3):

    if file != None:
        files = file
        status(f'{len(files)} file(s) found', True)
    else:
        status('Listing all the files...', True)
        paths = []
        list_files_in_dir(temp_dir, paths)
        files = []
        for f in paths:
            if f.split(os.sep)[-1][:5] != 'count' and f.split(os.sep)[-1][-3:] != 'npz':
                files.append(f) 
                
        status(f'{len(files)} file(s) found', True)

    embedder = Embedder(muat_orig=muat_orig, onehot=motif_one_hot)

    status(f'Starting motif embedding and pos and ges tokenization...', True)
    for i, f in enumerate(files):
        print(f)
        output_file_path = os.path.join(os.sep.join(f.split(os.sep)[:-1]), f.split(os.sep)[-1].split('.')[0] + '.npz')
        if os.path.exists(output_file_path):
            continue
        if i < continue_from -1:
            continue
        data_df = pd.read_csv(f, sep='\t', compression='gzip', header=0, index_col=0, low_memory=False)
        if len(data_df.loc[:,'seq']) == 0:
            continue

        if muat_orig:
            motif_embed = embedder.encode_motif_old(data_df.loc[:,'seq']).numpy()
        elif motif_one_hot:
            motif_embed =embedder.encode_motif_one_hot(data_df.loc[:,'seq'], motif_len=motif_len, max_token=25)
        else:
            motif_embed = embedder.embed_motif(data_df.loc[:, 'seq'].values.tolist(), data_df.loc[:, 'ref_seq'].values.tolist())
            motif_embed = motif_embed.numpy().astype(np.float32)
            
        print(f)
        tokenized_pos = embedder.encode_pos_old(data_df.loc[:,'chrom'], data_df.loc[:,'pos']).numpy()
        tokenized_ges = embedder.encode_ges_old(data_df.loc[:,'genic'], data_df.loc[:,'exonic'], data_df.loc[:,'strand']).numpy()
        assert motif_embed.shape[0] == tokenized_pos.shape[0] == tokenized_ges.shape[0], f'shape of motif\n{motif_embed.shape}\nshape of position\n{tokenized_pos.shape}\nshape of GES\n{tokenized_ges.shape}'
        np.savez_compressed(output_file_path, motif=motif_embed, position=tokenized_pos, GES = tokenized_ges)

        if i % report_interval == 0:
            status(f'{i+1}/{len(files)} files processed.The last saved file was {output_file_path}.', True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to temp directory with annoteated mutations csv files')
    parser.add_argument('-r', '--report_interval', type=int, default=1, help='interval between items to print the report')
    parser.add_argument('-c', '--continue_from', type=int, default=-1, help='index of files whre to continue embedding, optional')
    parser.add_argument('-f', '--files', default=None, help='list of paths to files to embed, optinal')
    parser.add_argument('-o', '--muat_orig', action='store_true', help='embed mutations with motif length 3 using muattion tokens and 3-mer tokenisation')
    parser.add_argument('-y', '--one_hot', action='store_true', help='use one-hot embedding for the motif')
    parser.add_argument('-l', '--length', type=int, default=3, help='motif length, default 3' )
    #parser.add_argument('-n', '--num_of_classes', type=int, default=24, help='number f tumour classes')
    args = parser.parse_args()

    files = args.files
    if args.files != None:
        files = files.split(',') 
    create_embedding(temp_dir=args.path,
                     report_interval=args.report_interval,
                     continue_from=args.continue_from,
                     file=files,
                     muat_orig=args.muat_orig,
                     motif_one_hot=args.one_hot,
                     motif_len=args.length)
                     #num_of_class=args.num_of_classes)
    
   