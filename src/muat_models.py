'''Different MuAt models 

Inspired from https://github.com/primasanjaya/muat-github/blob/1f91c38d00d2f2156df9c2cb0f4e21dba673cf85/models/model.py
Some models are modified compared to original MuAt model.
'''

import math
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F


class ModelConfig:
    """ base GPT config, params common to all GPT versions 
    
    wirtten by Prima
    """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self,
                 muat_orig = False,
                 vocab_size=3692,
                 block_size=5000,
                 num_class=24,
                 position_size = 2915,
                 ges_size=16,
                 embed_dim = 512,
                 motif_len = 3,
                 n_heads = 2,
                 n_layer = 1,
                **kwargs):
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_class = int(num_class)
        self.position_size = position_size
        self.ges_size = ges_size
        self.n_layer = n_layer
        self.n_head = n_heads
        self.n_embd = embed_dim
        self.muat_orig = muat_orig
        self.motif_len = motif_len

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
        #assert not torch.isnan(x).any(), f'Input of SelfAttention contains NaNs'
        #assert not torch.isinf(x).any(), f'Input of SelfAttention contains infinite values'

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        
        #assert not torch.isnan(keys).any(), f'Keys tensor contain Nans'
        #assert not torch.isnan(queries).any(), f'Queriess tensor contain Nans'
        #assert not torch.isnan(values).any(), f'Values tensor contain Nans'

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        #pdb.set_trace()
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        if vis:
            return dot

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        '''
        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        '''
        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities        
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
        #self.attention = nn.MultiheadAttention(emb, heads)
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
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        if config.muat_orig:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        x = inputs[0]

        if self.config.muat_orig:
            x = self.token_embedding(x)
            #print('Embed ok')

        assert not torch.isnan(torch.any(x))
        assert not torch.isinf(torch.any(x))
        assert x.numel() != 0 

        x = self.do(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class MuAtOneHotMotif(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        self.ln = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                nn.Linear(config.motif_len * config.vocab_size, int(config.n_embd)))
        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)


    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        x = inputs[0][0]
        
        x = self.ln(x)
        x = torch.unsqueeze(x, 0)
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        x = self.do(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
class MuAtOneHotMotifWithReLU(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        self.ln = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                nn.Linear(config.motif_len * config.vocab_size, int(config.n_embd), bias=False),
                                nn.ReLU())
        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)


    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        x = torch.squeeze(inputs[0], 0)
        #print(f'Input shape {x.shape}')#\nShould be {self.config.motif_len * self.config.vocab_size}')
        #print(f'Shape after flatten {torch.flatten(x, start_dim=1, end_dim=-1).shape}')
        
        x = self.ln(x)
        #print(f'Shape after linear {x.shape}')
        x = torch.unsqueeze(x, 0)
        #print(f'Shape after embedding {x.shape}')        
        x = self.do(x)

        x = self.tblocks(x)
        
        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
class MuAtMotifPosition(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.num_tokens, self.max_pool = config.vocab_size, False
        
        if config.muat_orig:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None, vis=None):

        tokens = x[0]
        
        #pdb.set_trace()
        postoken = x[1]
        if self.config.muat_orig:
            tokens = self.token_embedding(tokens)
            # assert not torch.isnan(torch.any(tokens))
            # assert not torch.isinf(torch.any(tokens))
            # assert tokens.numel() != 0 


        positions = self.position_embedding(postoken)

        # assert not torch.isnan(torch.any(positions)), f'{targets}, {torch.max(postoken), {torch.min(postoken)}}'
        # assert not torch.isinf(torch.any(positions))
        # assert positions.numel() != 0 


        x = torch.cat((tokens,positions),axis=2)

        x = self.do(x)
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class MuAtOneHotMotifPosition(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.num_tokens, self.max_pool = config.vocab_size, False
        
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)

        self.ln = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                nn.Linear(config.motif_len * config.vocab_size, int(config.n_embd)))

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None, vis=None):

        tokens = x[0][0]
        
        #pdb.set_trace()
        postoken = x[1]
        
        tokens = self.ln(tokens)
        tokens = torch.unsqueeze(tokens, 0)

        positions = self.position_embedding(postoken)

        # assert not torch.isnan(torch.any(positions)), f'{targets}, {torch.max(postoken), {torch.min(postoken)}}'
        # assert not torch.isinf(torch.any(positions))
        # assert positions.numel() != 0 


        x = torch.cat((tokens,positions),axis=2)
    
        x = self.do(x)
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class MuAtOneHotMotifPositionWithReLU(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.num_tokens, self.max_pool = config.vocab_size, False
        
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)

        self.ln = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                nn.Linear(config.motif_len * config.vocab_size, int(config.n_embd), bias = False),
                                nn.ReLU())

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None, vis=None):

        tokens = torch.squeeze(x[0], 0)
        
        #pdb.set_trace()
        postoken = x[1]
        
        tokens = self.ln(tokens)
        tokens = torch.unsqueeze(tokens, 0)

        positions = self.position_embedding(postoken)

        # assert not torch.isnan(torch.any(positions)), f'{targets}, {torch.max(postoken), {torch.min(postoken)}}'
        # assert not torch.isinf(torch.any(positions))
        # assert positions.numel() != 0 


        x = torch.cat((tokens,positions),axis=2)
    
        x = self.do(x)
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    

class MuAtMotifPositionGES(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        if config.muat_orig:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size, 4,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd + 4), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd + 4), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd + 4), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None,vis=None):

        tokens = x[0]
        postoken = x[1]
        gestoken = x[2]

        if self.config.muat_orig:
            tokens = self.token_embedding(tokens)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)

        x = torch.cat((tokens,positions,ges),axis=2)

        x = self.do(x)
        #pdb.set_trace()
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
class MuAtMotifOneHotPositionGES(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size, 4,padding_idx=0)

        self.ln = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                nn.Linear(config.motif_len * config.vocab_size, int(config.n_embd)))

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd + 4), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd + 4), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd + 4), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None,vis=None):

        tokens = x[0][0]
        postoken = x[1]
        gestoken = x[2]

        
        tokens = self.ln(tokens)
        tokens = torch.unsqueeze(tokens, 0)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)

        x = torch.cat((tokens,positions,ges),axis=2)

        x = self.do(x)
        #pdb.set_trace()
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
class MuAtMotifOneHotPositionGESWithReLU(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size, 4,padding_idx=0)

        self.ln = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                nn.Linear(config.motif_len * config.vocab_size, int(config.n_embd), bias=False),
                                nn.ReLU())

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd + 4), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd + config.n_embd + 4), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd + 4), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None,vis=None):

        tokens = torch.squeeze(x[0], 0)
        postoken = x[1]
        gestoken = x[2]

        
        tokens = self.ln(tokens)
        tokens = torch.unsqueeze(tokens, 0)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)

        x = torch.cat((tokens,positions,ges),axis=2)

        x = self.do(x)
        #pdb.set_trace()
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        #x = self.tofeature(x)

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class MuAtMotifEpiPos(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd + 50, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd + 50, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        tokens = inputs[0]
        epipos = inputs[1]

        
        x = self.token_embedding(tokens)
        x = torch.cat((x, epipos),axis=2)

        x = self.do(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
class MuAtMotifPositionEpiPos(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd + config.n_embd + 50, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd + config.n_embd + 50, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        tokens = inputs[0]
        postoken = inputs[1]
        epipos = inputs[2]

        
        x = self.token_embedding(tokens)
        positions = self.position_embedding(postoken)
        x = torch.cat((x, positions, epipos),axis=2)

        x = self.do(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class MuAtMotifEpiPosGES(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size, 4,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd + 50 +4, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd + 50+ 4, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        tokens = inputs[0]
        gestoken = inputs[1]
        epipos = inputs[2]

        
        motif = self.token_embedding(tokens)
        ges = self.ges_embedding(gestoken)
        x = torch.cat((motif, epipos, ges),axis=2)

        x = self.do(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
class MuAtMotifPositionGESEpiPos(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)
        self.ges_embedding = nn.Embedding(config.ges_size, 4,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd + config.n_embd + 50 + 4, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd + config.n_embd + 50 + 4, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)
    

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        tokens = inputs[0]
        postoken = inputs[1]
        gestoken = inputs[2]
        epipos = inputs[3]

        
        x = self.token_embedding(tokens)
        positions = self.position_embedding(postoken)
        ges = self.ges_embedding(gestoken)
        x = torch.cat((x, positions, ges, epipos),axis=2)

        x = self.do(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class MuAtMotifContext(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tokens, self.max_pool = config.vocab_size, False

        if config.muat_orig:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=config.n_embd + 768, heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)
        
        # features are utilised only UMAP visualisation
        #self.tofeature = nn.Sequential(nn.Linear(int(config.n_embd), 24),
        #                                nn.ReLU())

        self.toprobs = nn.Linear(config.n_embd + 768, config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, inputs,targets=None,vis=None):

        #b, t, e = inputs.size()
        motif = inputs[0]
        context = inputs[-1]

        if self.config.muat_orig:
            tokens = self.token_embedding(motif)
            #print('Embed ok')

        assert not torch.isnan(torch.any(x))
        assert not torch.isinf(torch.any(x))
        assert x.numel() != 0 

        x = torch.cat((tokens, context),axis=2)
        x = self.do(x)


        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        x = self.tblocks(x)
        
        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0

        if vis:
            return x

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.tofeature(x)

        #assert not torch.isnan(torch.any(x))
        #assert not torch.isinf(torch.any(x))
        #assert x.numel() != 0 

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

###########################
def get_model(arch, mconf):
    if arch == 'MuAtMotif':
        pos = False
        ges = False
        one_hot = False
        return MuAtMotif(mconf), pos, ges, one_hot
    elif arch == 'MuAtMotifPosition':
        pos = True
        ges = False
        one_hot = False
        return MuAtMotifPosition(mconf), pos, ges, one_hot    
    elif arch == 'MuAtMotifPositionGES':
        pos = True
        ges = True
        one_hot = False
        return MuAtMotifPositionGES(mconf), pos, ges, one_hot
    elif arch == 'MuAtOneHotMotif':
        pos = False
        ges = False
        one_hot = True
        return MuAtOneHotMotif(mconf), pos, ges, one_hot   
    elif arch == 'MuAtOneHotMotifPosition':
        pos = True
        ges = False
        one_hot = True
        return MuAtOneHotMotifPosition(mconf), pos, ges, one_hot 
    elif arch == 'MuAtOneHotMotifPositionGES':
        pos = True
        ges = True
        one_hot = True
        return MuAtMotifOneHotPositionGES(mconf), pos, ges, one_hot 
    elif arch == 'MuAtOneHotMotifWithoutReLU':
        pos = False
        ges = False
        one_hot = True
        return MuAtOneHotMotifWithReLU(mconf), pos, ges, one_hot   
    elif arch == 'MuAtOneHotMotifPositionWithoutReLU':
        pos = True
        ges = False
        one_hot = True
        return MuAtOneHotMotifPositionWithReLU(mconf), pos, ges, one_hot 
    elif arch == 'MuAtOneHotMotifPositionGESWithoutReLU':
        pos = True
        ges = True
        one_hot = True
        return MuAtMotifOneHotPositionGESWithReLU(mconf), pos, ges, one_hot 
    elif arch == 'MuAtMotifEpiPos':
        pos = False
        ges = False
        one_hot = False
        return MuAtMotifEpiPos(mconf), pos, ges, one_hot
    elif arch == 'MuAtMotifEpiPosGES':
        pos = False
        ges = True
        one_hot = False
        return MuAtMotifEpiPosGES(mconf), pos, ges, one_hot
    elif arch =='MuAtMotifPositionEpiPos':
        pos = True
        ges = False
        one_hot = False
        return MuAtMotifPositionEpiPos(mconf), pos, ges, one_hot
    elif arch =='MuAtMotifPositionGESEpiPos':
        pos = True
        ges = True
        one_hot = False
        return MuAtMotifPositionGESEpiPos(mconf), pos, ges, one_hot
    
    elif arch =='MuAtMotifContext':
        pos = False
        ges = False
        one_hot = False
        return MuAtMotifContext(mconf), pos, ges, one_hot



    