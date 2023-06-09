import torch.nn as nn
from components.SelfAttention import SelfAttention
from components.TransformerBlock import TransformerBlock

class DecoderBlock(nn.Module):
  def __init__(
    self,
    embed_size,
    heads,
    forward_expansion,
    dropout,
    device
  ):
    super(DecoderBlock, self).__init__()
    
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(
      embed_size, heads, dropout, forward_expansion
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, value, key, src_mask, trg_mask):
    # masked attention
    attention = self.attention(x,x,x, trg_mask)

    # add and norm
    query = self.dropout(self.norm(attention + x))

    # transformer block
    out = self.transformer_block(value, key, query, src_mask)

    return out
