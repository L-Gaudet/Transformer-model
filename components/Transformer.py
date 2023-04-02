import torch
import torch.nn as nn
from components.Encoder import Encoder
from components.Decoder import Decoder

class Transformer(nn.Module):
  def __init__(
    self,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embed_size=256,
    num_layers=6,
    forward_expansion=4,
    heads=8,
    dropout=0,
    # device='cuda',
    device = 'cpu',
    max_length=100
  ):
    super(Transformer, self).__init__()

    # define encoder
    self.encoder = Encoder(
      src_vocab_size,
      embed_size,
      num_layers,
      heads,
      device,
      forward_expansion,
      dropout,
      max_length
    )

    # define decoder
    self.decoder = Decoder(
      trg_vocab_size,
      embed_size,
      num_layers,
      heads,
      forward_expansion,
      dropout,
      device,
      max_length
    )

    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.device = device

  # define make src mask
  def make_src_mask(self, src):
    # we want the src_mask in the shape of (N, 1, 1, src_len)
    # if src is the src pad index then it will be 1, if not it will be 0
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(self.device)

  # define make trg mask
  def make_trg_mask(self, trg):
    N, trg_len = trg.shape

    # we want a lower triangular matrix 
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
      N, 1, trg_len, trg_len
    )

    return trg_mask.to(self.device)

  def forward(self, src, trg):
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)

    enc_src = self.encoder(src, src_mask)

    out = self.decoder(trg, enc_src, src_mask, trg_mask)

    return out