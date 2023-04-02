import torch
import torch.nn as nn
from components.DecoderBlock import DecoderBlock

class Decoder(nn.Module):
  def __init__(
    self,
    trg_vocab_size,
    embed_size,
    num_layers,
    heads,
    forward_expansion,
    dropout,
    device,
    max_length
  ):
    super(Decoder, self).__init__()

    # define embeddings 
    self.device = device
    self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
    self.positional_embedding = nn.Embedding(max_length, embed_size)

    # create num_layers of the decoder block 
    self.layers = nn.ModuleList(
      [
        DecoderBlock(
          embed_size, heads, forward_expansion, dropout, device
        )
        for _ in range(num_layers)
      ]
    )

    # final linear layer
    self.fully_connected_out = nn.Linear(embed_size, trg_vocab_size)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_out, src_mask, trg_mask):
    # we get the number of examples and the sequence legnth
    N, seq_len = x.shape
    
    # we create a range of 0 to the sequence length for every example
    positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

    # we give the word embedding and the postions and the model will now know the positions of words
    x = self.dropout((self.word_embedding(x) + self.positional_embedding(positions)))

    # run the decoder block for all of the layers
    for layer in self.layers:
      x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)

    # run final linear layer
    out = self.fully_connected_out(x)

    # final softmax 
    # out = torch.softmax(out, dim=2)

    return out
