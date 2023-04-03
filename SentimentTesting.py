import torch
import torch.nn as nn
import torch.nn.functional as F
from components.TransformerClassifier import TransformerClassifier
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
# from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter
import random, tqdm, math

print(torch.__version__)

LOG2E = math.log2(math.e)
# TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
  # lower case the text, 
  # return a tuple of a padded minibatch and a list containing the lengths of each examples,
  # produce tensors with the batch dimension first.
# LABEL = data.Field(sequential=False)
NUM_CLASSES = 2

def start(arg):
  # prep data in vocab -> create and train transformer

  # Tensorboard logging
  tbw = SummaryWriter(log_dir=arg.tb_dir) 

  

  # load data and create vocab for our model
  if not arg.final: # run on test set
    train, test = IMDB(split=('train', 'test'))
    tokenizer = get_tokenizer('basic_english')

    # build the train vocab
    counter = Counter()
    for text, label in train:
        counter.update(tokenizer(text))
    vocab = Vocab(counter, min_freq=1)

    text_transform = lambda x: [vocab['']] + [vocab[token] for token in tokenizer(x)] + [vocab['']]
    label_transform = lambda x: 1 if x == 'pos' else 0

    def collate_batch(batch):
      label_list, text_list = [], []
      for (_label, _text) in batch:
            label_list.append(label_transform(_label))
            processed_text = torch.tensor(text_transform(_text))
            text_list.append(processed_text)
      return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

    train_iter = DataLoader(list(train), batch_size=8, shuffle=True, 
                              collate_fn=collate_batch)
    
    # build the test vocab
    counter = Counter()
    for text, label in test:
        counter.update(tokenizer(text))
    vocab = Vocab(counter, min_freq=1)

    test_iter = DataLoader(list(test), batch_size=8, shuffle=False, 
                              collate_fn=collate_batch)

  else: # run on val set
    # NEW
    dtrain, _ = IMDB(split=('train', 'test'))

    # train, test = dtrain.split()
    # print(dtrain)
    # Define the sizes of the train and validation sets
    # TRAIN_SIZE = int(len(dtrain) * 0.8)
    # VAL_SIZE = len(dtrain) - TRAIN_SIZE

    # # Split the train dataset into train and validation sets
    # train, test = random_split(dtrain, [TRAIN_SIZE, VAL_SIZE])

    # train, test = tdata.split(split_ratio=0.8)
    tokenizer = get_tokenizer('basic_english')

    # build the train vocab
    counter = Counter()
    for text, label in train:
        counter.update(tokenizer(text))
    vocab = Vocab(counter, min_freq=1)

    text_transform = lambda x: [vocab['']] + [vocab[token] for token in tokenizer(x)] + [vocab['']]
    label_transform = lambda x: 1 if x == 'pos' else 0

    def collate_batch(batch):
      label_list, text_list = [], []
      for (_label, _text) in batch:
            label_list.append(label_transform(_label))
            processed_text = torch.tensor(text_transform(_text))
            text_list.append(processed_text)
      return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

    train_iter = DataLoader(list(train), batch_size=8, shuffle=True, 
                              collate_fn=collate_batch)
    
    # build the test vocab
    counter = Counter()
    for text, label in test:
        counter.update(tokenizer(text))
    vocab = Vocab(counter, min_freq=1)

    test_iter = DataLoader(list(test), batch_size=8, shuffle=False, 
                              collate_fn=collate_batch)


  print(f'- num of training examples {len(train_iter)}')
  print(f'- num of {"test" if arg.final else "validation"} examples {len(test_iter)}')

  if arg.max_length < 0:
    mx = max([input.text[0].size(1) for input in train_iter])
    mx = mx*2
    print(f'- maximum sequence length: {mx}')
  else:
    mx = arg.max_length

  # create model
  model = TransformerClassifier(emb_dimension=arg.embedding_size, 
                                heads=arg.num_heads, 
                                depth=arg.depth,
                                seq_length=mx,
                                num_tokens=arg.vocab_size,
                                num_classes=NUM_CLASSES,
                                max_pool=arg.max_pool)
  
  optimizer = torch.optim.Adam(lr=arg.lr, params=model.parameters())
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

  # training loop
  seen = 0
  for e in range(arg.num_epochs):
    print(f'\n epoch {e}')
    model.train(True)

    for batch in tqdm.tqdm(train_iter):
      optimizer.zero_grad()

      input = batch.text[0]
      label = batch.label - 1

      if input.size(1) > mx:
        input = input[:, :mx]
      
      # forward pass
      out = model(input)

      # compute gradient
      loss = F.nll_loss(out, label)

      # backprop
      loss.backward()

      # clip gradients
      # - If the total gradient vector has a length > 1, we clip it back down to 1.
      if arg.gadient_clipping > 0.0:
        nn.utils.clip_grad_norm(model.parameters(), arg.gradient_clipping)

      optimizer.step()
      scheduler.step()

      seen += input.size(0)
      tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

    with torch.no_grad():
      model.tain(False)
      total = 0.0
      correct = 0.0

      for batch in test_iter:
        input = batch.text[0]
        label = batch.label - 1

        if input.size(1) > mx:
          input = input[:, :mx]
        
        # take one ouput of model
        out = model(input).argmax(dim=1)

        total += float(input.size(0))
        correct += float((label == out).sum().item())

      # calculate and ouput accuracy
      accuracy = correct/total
      print(f'-- {"test" if arg.final else "validation"} accuracy {accuracy:.3}')
      tbw.add_scalar('classification/test-loss', float(loss.item()), e)

if __name__ == '__main__':
  parser = ArgumentParser()

  parser.add_argument("-e", "--num-epochs",
                      dest="num_epochs",
                      help="Number of epochs.",
                      default=80, type=int)

  parser.add_argument("-b", "--batch-size",
                      dest="batch_size",
                      help="The batch size.",
                      default=4, type=int)

  parser.add_argument("-l", "--learn-rate",
                      dest="lr",
                      help="Learning rate",
                      default=0.0001, type=float)

  parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                      help="Tensorboard logging directory",
                      default='./runs')

  parser.add_argument("-f", "--final", dest="final",
                      help="Whether to run on the real test set (if not included, the validation set is used).",
                      action="store_true")

  parser.add_argument("--max-pool", dest="max_pool",
                      help="Use max pooling in the final classification layer.",
                      action="store_true")

  parser.add_argument("-E", "--embedding", dest="embedding_size",
                      help="Size of the character embeddings.",
                      default=128, type=int)

  parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                      help="Number of words in the vocabulary.",
                      default=50_000, type=int)

  parser.add_argument("-M", "--max", dest="max_length",
                      help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                      default=512, type=int)

  parser.add_argument("-H", "--heads", dest="num_heads",
                      help="Number of attention heads.",
                      default=8, type=int)

  parser.add_argument("-d", "--depth", dest="depth",
                      help="Depth of the network (nr. of self-attention layers)",
                      default=6, type=int)

  parser.add_argument("-r", "--random-seed",
                      dest="seed",
                      help="RNG seed. Negative for random",
                      default=1, type=int)

  parser.add_argument("--lr-warmup",
                      dest="lr_warmup",
                      help="Learning rate warmup.",
                      default=10_000, type=int)

  parser.add_argument("--gradient-clipping",
                      dest="gradient_clipping",
                      help="Gradient clipping.",
                      default=1.0, type=float)

  options = parser.parse_args()

  print('OPTIONS ', options)

  start(options)
