# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/12/3 3:48 下午
# @File    : main.py


import argparse
import os.path
import random
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

import train
from model import Ernie
from utils import read_train_ds, collate_fn, MyData, convert_example
from vocab import Vocab


def config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument("--model_name_or_path", type=str, default="ernie",
	                    help="Pretraining model name or path")
	parser.add_argument("--max_seq_length", type=int, default=128,
	                    help="The maximum total input sequence length after SentencePiece tokenization.")
	parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train.")
	parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
	parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
	parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
	parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")

	parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
	parser.add_argument("--warmup_proportion", default=0.1, type=float,
	                    help="Linear warmup proption over the training process.")
	parser.add_argument("--max_steps", default=-1, type=int,
	                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
	parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--ignore_label", default=-100, type=int, help="Ignore label for CrossEntropyLoss")
	parser.add_argument("--extra_train_ds_dir", default=None, type=str, help="The directory of extra train dataset.")

	args = parser.parse_args()

	return args


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
	setup_seed(20)
	args = config()
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	pinyin_vocab = Vocab.load_vocabulary(
		args.pinyin_vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')

	train_list = list(read_train_ds('Data/AutomaticCorpusGeneration.txt'))
	train_list.extend(list(read_train_ds("Data/sighanCntrain.txt")))
	test_List = list(read_train_ds('Data/sighanCntest.txt'))

	modelPath = args.model_name_or_path
	tokenizer = AutoTokenizer.from_pretrained(modelPath)
	model = Ernie(modelPath, pinyin_vocab_size=len(pinyin_vocab), pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token],
	              tie_weight=False)
	vocab_size = model.vocab_size

	train_ids = [convert_example(example, tokenizer, pinyin_vocab, args.max_seq_length, ignore_label=args.ignore_label)
	             for example in train_list]
	test_ids = [convert_example(example, tokenizer, pinyin_vocab, args.max_seq_length, ignore_label=args.ignore_label)
	            for example in test_List]

	train_x = MyData(train_ids)
	test_x = MyData(test_ids)

	collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id,
	                  pad_pinyin_id=pinyin_vocab.token_to_idx[pinyin_vocab.pad_token], ignore_index=args.ignore_label)
	train_data_loader = DataLoader(train_x, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
	test_data_loader = DataLoader(test_x, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

	train.trainer(model, train_data_loader, test_data_loader,args, device,vocab_size)
