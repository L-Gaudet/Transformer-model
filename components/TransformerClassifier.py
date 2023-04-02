import torch
import torch.nn as nn
import torch.nn.functional as F
from components.TransformerBlock import TransformerBlock

class TransformerClassifier(nn.Module):
    # transformer sequence classifier to classify sentiment
    def __init__(
            self,
            emb_dimension,
            heads,
            layers,
            seq_len,
            num_tokens,
            num_classes,
            max_pool=True,
            dropout=0.0,
            wide=False
    ):
        super.__init__()

        # define token sizes
        self.num_token = num_tokens
        self.max_pool = max_pool

        # define embeddings
        self.token_embedding = nn.Embedding(embedding_dim=emb_dimension,
                                            num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb_dimension,
                                          num_embeddings=seq_len)
        
        # create N transformer layers
        transformer_blocks = []
        for _ in range(layers):
            transformer_blocks.append(
                TransformerBlock(embed_size=emb_dimension, 
                                heads=heads,
                                mask=False,
                                dropout=dropout
                                forward_expansion=seq_len)
            )

        # make transformer blocks sequential
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        # use linear layer to get probabilities
        self.to_probabilities = nn.Linear(emb_dimension, num_classes)

        # dropout normalization
        self.dropout = nn.Dropout(dropout)
        


    def forward(self, x):
        # x is a tensor of token indeces of the batch by sequence length integer
        # returns predicted log(probability) vectors for the tokens based on preceding tokens

        # create positional embeddings
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device='cpu'))[None,:,:].expand(b,t,e)

        x = tokens + positions

        # send through dropout
        x = self.dropout(x)

        # send through transformer layers
        x = self.tblocks(x)

        # pool to get narrow classification
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) 

        # get probabilities
        x = self.to_probabilities(x)

        # log softmax b/c log(probs)
        out = F.log_softmax(x, dim=1)

        return out