# import necessary packages
import torch
import torch.nn as nn 
import torch.nn.functional as F

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads=8):
    super(SelfAttention, self).__init__()

    assert (embed_size % heads == 0), 'Embed size not divisible by heads'

    self.embed_size = embed_size
    self.heads = heads
    self.head_dimension = embed_size // heads

    # define linear layers to send queries, keys, and values through
    self.to_values = nn.Linear(self.embed_size, self.embed_size, bias=False)
    self.to_keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
    self.to_queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
    
    # concatenate heads after multi-head attention
    self.fully_connected_out = nn.Linear(embed_size, embed_size)

  def forward(self, x, mask=False):
    # num training examples to send in at one time
    # N = query.shape[0]

    b, t, e = x.size()
    assert e == self.embed_size, f'Input embedding dim ({e}) should match layer embedding dim ({self.embed_size})'

    h = self.heads
    
    s = e//h

    keys = self.to_keys(x)
    queries = self.to_queries(x)
    values = self.to_keys(x)

    keys    = keys.view(b, t, h, s)
    queries = queries.view(b, t, h, s)
    values  = values.view(b, t, h, s)

    # # these correspond to source sentence length and target sentence length
    # value_len = values.shape[1]
    # key_len = keys.shape[1]
    # query_len = query.shape[1]

    # # split embeddings into multiple heads
    # values = values.reshape(N, value_len, self.heads, self.head_dimension)
    # keys = keys.reshape(N, key_len, self.heads, self.head_dimension)
    # queries = query.reshape(N, query_len, self.heads, self.head_dimension)

    # - fold heads into the batch dimension
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
    values = values.transpose(1, 2).contiguous().view(b * h, t, s)

    queries = queries / (e ** (1/4))
    keys    = keys / (e ** (1/4))

    # # send through linear layers
    # values = self.values(values)
    # keys = self.keys(keys)
    # queries = self.queries(queries)

    #------- MatMul Q and K(Transposed) ---------#
      # queries shape: (N, query_len, heads, head_dimension)
      # keys shape: (N, key_len, heads, head_dimension)
    # we want
      # QdotK shape: (N, heads, query_len, key_len)
    # QdotKt = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

    dot = torch.bmm(queries, keys.transpose(1, 2))

    assert dot.size() == (b*h, t, t)

    #------------ Scale ------------#
    # QdotKt = QdotKt / (self.embed_size ** (1/2))

    #----- Mask (for decoder) ------#
    # decoder requires masked multi-head attention
    if mask == True:
      # closes elements above the diagonal so that the model can't see future values
      QdotKt = QdotKt.masked_fill(mask == 0, float('-1e20'))

      h, w = dot.size(-2), dot.size(-1)

      indices = torch.triu_indices(h, w, offset=0)
      dot[..., indices[0], indices[1]] = float('-1e20')

    #---------- Softmax ------------#
    # soft = torch.softmax(QdotKt / (self.embed_size ** (1/2)), dim=3)
    dot = F.softmax(dot, dim=2)

    #------ MatMul soft and V ------#
      # soft shape: (N, heads, query_len, key_len)
      # values shape: (N, value_len, heads, head_dimension)
    # we want
      # (N, query_len, heads, head_dimension)
      # after multiplying, flatten last two dimensions
    # out = torch.einsum('nhql,nlhd->nqhd', [soft,values]).reshape(
    #   N, query_len, self.heads*self.head_dimension)

    # apply the matrix multiplication
    out = torch.bmm(dot, values).view(b, h, t, s)

    # swap h, t back, unify heads
    out = out.transpose(1, 2).contiguous().view(b, t, s * h)

    #------ Concatenate heads ------#
    out = self.fully_connected_out(out)
    
    return out
