from components.TransformerClassifier import TransformerClassifier

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from torchtext import data, datasets, vocab
from torchtext import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torchtext.vocab import Vocab
# from torchtext.data.utils import get_tokenizer
# from torchtext.datasets import IMDB
# from collections import Counter
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader, random_split
# from argparse import ArgumentParser
# # from torchtext.data.utils import get_tokenizer
# from torch.utils.tensorboard import SummaryWriter
# import random, tqdm, math

# from torchtext.legacy import data, datasets, Vocab
# # from torchtext import data, datasets
# # from torchtext.vocab import Vocab

print(torch.__version__)

LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
#   lower case the text, 
#   return a tuple of a padded minibatch and a list containing the lengths of each examples,
#   produce tensors with the batch dimension first.
LABEL = data.Field(sequential=False)
NUM_CLASSES = 2

def getDevice(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def start(arg):
# prep data in vocab -> create and train transformer

	# Tensorboard logging
	tbw = SummaryWriter(log_dir=arg.tb_dir) 



# load data and create vocab for our model
	if arg.final:
		train, test = datasets.IMDB.splits(TEXT, LABEL)

		TEXT.build_vocab(train, max_size=arg.vocab_size - 2)
		LABEL.build_vocab(train)

		train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=getDevice())
	else:
		tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
		train, test = tdata.split(split_ratio=0.8)

		TEXT.build_vocab(train, max_size=arg.vocab_size - 2) # - 2 to make space for <unk> and <pad>
		LABEL.build_vocab(train)

		train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=getDevice())

	print(f'- nr. of training examples {len(train_iter)}')
	print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

	if arg.max_length < 0:
		mx = max([input.text[0].size(1) for input in train_iter])
		mx = mx*2
		print(f'- maximum sequence length: {mx}')
	else:
		mx = arg.max_length

	# create model
	model = TransformerClassifier(emb_dimension=arg.embedding_size, 
								heads=arg.num_heads, 
								layers=arg.depth,
								seq_len=mx,
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
			if arg.gradient_clipping > 0.0:
				nn.utils.clip_grad_norm(model.parameters(), arg.gradient_clipping)

			optimizer.step()
			scheduler.step()

			seen += input.size(0)
			tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

		with torch.no_grad():
			model.train(False)
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

	torch.save(model, './TrainedClassifier')

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
