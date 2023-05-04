import torch
from torchtext.legacy import data, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam

class SentimentClassifier(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, num_classes):
        super(SentimentClassifier, self).__init__()

        self.token_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)

        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(
                Encoder(embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
            )
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.linear = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask):
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        positions = self.pos_embedding(torch.arange(t, device=device))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.dropout(x)

        out = self.transformer_blocks(x, mask)
        out = out.mean(dim=1)
        out = self.linear(out)
        return out


TEXT = torchtext.data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", batch_first=True, lower=True, fix_length=max_length)
LABEL = torchtext.data.LabelField(dtype=torch.float)

train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=src_vocab_size - 2)
LABEL.build_vocab(train_data)

train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
)

# Hyperparameters
src_vocab_size = len(TEXT.vocab)
embed_size = 256
num_layers = 3
heads = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forward_expansion = 4
dropout = 0.1
max_length = 100
num_classes = 2
num_epochs = 5
lr = 0.0005
batch_size = 32

# Initialize the model
model = SentimentClassifier(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, num_classes)
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

# Train and validate the model
best_valid_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train(model, train_iterator, criterion, optimizer, device)
    valid_loss = evaluate(model, valid_iterator, criterion, device)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # Save the model if the validation loss is lower than the best validation loss found so far
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')
