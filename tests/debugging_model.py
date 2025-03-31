import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

torch.set_printoptions(profile="full")
logger = logging.getLogger(__name__)

class ModelConfig:
    """ base GPT config, params common to all GPT versions 
    
    wirtten by Prima
    """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self,vocab_size=0, block_size=5000, num_class=24,position_size = 1, ges_size=1,dnn_input=1,epi_size=1, n_embd = 1536, args=None, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_class = int(num_class)
        self.position_size = position_size
        self.ges_size = ges_size
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = n_embd
        #self.dnn_input=dnn_input
        #self.epi_size=epi_size

        for k,v in kwargs.items():
            setattr(self, k, v)

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:

        Written by Prima
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)
    
    def contains_nan(self,tensor):
        return bool((tensor != tensor).sum() > 0)

    def forward(self, x,vis=False):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        if keys.isnan().any():
            print('line 67')
            print(f'\nkeys{keys}')

        if queries.isnan().any():
            print('line 68')
            print(f'\nqueries:{queries}')

        if values.isnan().any():
            print('line 80')
            print(f'values:\n{values}')

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if dot.isnan().any():
            print('line 79')
            print(f'dot:\n{dot}\nqueries:{queries}\nkeys{keys.transpose(1, 2)}')
        #pdb.set_trace()
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        if vis:
            return dot

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities        
        if dot.isnan().any():
            print('line 95')
            print(f'dot:\n{dot}')

        assert not self.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)


        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    """Written by Prima"""
    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
    
###########################################################
# Muat model using only Motif
# Modified part

class MuAtMotif(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.vocab_size, False

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(config.n_embd, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        x = inputs[0]

        x = self.do(x)
        x = self.tblocks(x)

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('prepare model and config..')
    config = ModelConfig()
    model = MuAtMotif(config).to(device)
    print('loading the data..')
    data = np.load('/mnt/ahuttun/multimodal/data/temp/motif_len_3_fixed/CNS-GBM/d1132127-1250-43af-9c16-425798a3d1a7/indel_d1132127-1250-43af-9c16-425798a3d1a7.npz')
    print('array loaded')
    motif = data['motif']
    print('start training...')
    model([torch.from_numpy(motif).reshape([1, motif.shape[0],1536]).to(device)])
    print('finnish without errors')