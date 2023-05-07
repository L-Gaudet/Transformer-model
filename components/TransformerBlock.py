# import necessary packages
import torch.nn as nn 
from components.SelfAttention import SelfAttention

class TransformerBlock(nn.Module):
  # transformer block architecture
    # attention block -> add & normalize -> feed forward -> add & normalize

  def __init__(self, embed_size, heads, mask, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.mask = mask
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion * embed_size, embed_size)
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # attention
    attention = self.attention(x, self.mask)

    # add & norm
    x = self.dropout(self.norm1(attention + x))

    # feed forward
    forward = self.feed_forward(x)

    # add & norm
    out = self.dropout(self.norm2(forward + x))

    return out
