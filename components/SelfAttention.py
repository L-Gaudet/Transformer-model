# import necessary packages
import torch
import torch.nn as nn 

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()

    assert (embed_size % heads == 0), 'Embed size not divisible by heads'

    self.embed_size = embed_size
    self.heads = heads
    self.head_dimension = embed_size // heads

    # define linear layers to send queries, keys, and values through
    self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
    self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
    self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
    
    # concatenate heads after multi-head attention
    self.fully_connected_out = nn.Linear(embed_size, embed_size)

  def forward(self, values, keys, query, mask):
    # num training examples to send in at one time
    N = query.shape[0]

    # these correspond to source sentence length and target sentence length
    value_len = values.shape[1]
    key_len = keys.shape[1]
    query_len = query.shape[1]

    # split embeddings into multiple heads
    values = values.reshape(N, value_len, self.heads, self.head_dimension)
    keys = keys.reshape(N, key_len, self.heads, self.head_dimension)
    queries = query.reshape(N, query_len, self.heads, self.head_dimension)

    # send through linear layers
    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)

    #------- MatMul Q and K(Transposed) ---------#
      # queries shape: (N, query_len, heads, head_dimension)
      # keys shape: (N, key_len, heads, head_dimension)
    # we want
      # QdotK shape: (N, heads, query_len, key_len)
    QdotKt = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

    #------------ Scale ------------#
    # QdotKt = QdotKt / (self.embed_size ** (1/2))

    #----- Mask (for decoder) ------#
    # decoder requires masked multi-head attention
    if mask is not None:
      # closes elements above the diagonal so that the model can't see future values
      QdotKt = QdotKt.masked_fill(mask == 0, float('-1e20'))

    #---------- Softmax ------------#
    soft = torch.softmax(QdotKt / (self.embed_size ** (1/2)), dim=3)
    # attention = torch.softmax(QdotKt)

    #------ MatMul soft and V ------#
      # soft shape: (N, heads, query_len, key_len)
      # values shape: (N, value_len, heads, head_dimension)
    # we want
      # (N, query_len, heads, head_dimension)
      # after multiplying, flatten last two dimensions
    out = torch.einsum('nhql,nlhd->nqhd', [soft,values]).reshape(
      N, query_len, self.heads*self.head_dimension)

    #------ Concatenate heads ------#
    out = self.fully_connected_out(out)
    
    return out
