import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class ModelConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self,vocab_size, block_size, num_class,position_size = 1, ges_size=1,dnn_input=1,epi_size=1,args=None, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_class = num_class
        self.position_size = position_size
        self.ges_size = ges_size
        self.args = args
        self.dnn_input=dnn_input
        self.epi_size=epi_size

        for k,v in kwargs.items():
            setattr(self, k, v)

class MuAtMotifPosition(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, config):
        super().__init__()

        self.num_tokens, self.max_pool = config.vocab_size, False

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd,padding_idx=0)
        self.position_embedding = nn.Embedding(config.position_size, config.n_embd,padding_idx=0)

        tblocks = []
        for i in range(config.n_layer):
            tblocks.append(
                TransformerBlock(emb=int(config.n_embd + config.n_embd), heads=config.n_head, seq_length=config.block_size, mask=False, dropout=config.attn_pdrop))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(int(config.n_embd + config.n_embd), config.num_class)

        self.do = nn.Dropout(config.embd_pdrop)

    def forward(self, x,targets=None,vis=None):

        triplettoken = x[0]
        #pdb.set_trace()
        postoken = x[1]
        
    
        tokens = self.token_embedding(triplettoken)
        positions = self.position_embedding(postoken)

        x = torch.cat((tokens,positions),axis=2)

        x = self.do(x)
        
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        logits = self.toprobs(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class TransformerBlock(nn.Module):
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

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
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

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

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