# import necessary packages
import torch.nn as nn 
from components.SelfAttention import SelfAttention

class TransformerBlock(nn.Module):
  # transformer block architecture
    # attention block -> add & normalize -> feed forward -> add & normalize

  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion * embed_size, embed_size)
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, value, key, query, mask):
    # attention
    attention = self.attention(value, key, query, mask)

    # add & norm
    x = self.dropout(self.norm1(attention + query))

    # feed forward
    forward = self.feed_forward(x)

    # add & norm
    out = self.dropout(self.norm2(forward + x))

    return out
