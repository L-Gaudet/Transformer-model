import torch
import torch.nn as nn
from components.TransformerBlock import TransformerBlock

class Encoder(nn.Module):
  def __init__(
    self,
    src_vocab_size,
    embed_size,
    num_layers,
    heads,
    device,
    forward_expansion,
    dropout,
    max_length
  ):

    super(Encoder, self).__init__()
    # define embeddings for input
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.positional_embedding = nn.Embedding(max_length, embed_size)

    # define layers, which consists of a single transformer blocks
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion
        )
        for _ in range(num_layers)
      ]
    )
    
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    # we get the number of examples and the sequence legnth
    N, seq_len = x.shape

    # we create a range of 0 to the sequence length for every example
    positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

    # we give the word embedding and the postions and the model will now know the positions of words
    out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

    for layer in self.layers:
      # in the encoder, all our v, k, q input vectors are the same
      out = layer(out, out, out, mask)

    return out