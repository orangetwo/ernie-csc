# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/12/1 5:17 下午
# @File    : model.py
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

"""
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0")
"""


class Ernie(nn.Module):
	def __init__(self, modelPath, pinyin_vocab_size, pad_pinyin_id, tie_weight=False):
		super(Ernie, self).__init__()
		self.model = AutoModel.from_pretrained(modelPath)
		# self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
		self.embed_size = self.model.config.hidden_size
		self.hidden_size = self.model.config.hidden_size
		self.vocab_size = self.model.config.vocab_size

		self.pad_token_id = self.model.config.pad_token_id
		# self.pinyin_vocab_size = pinyin_vocab_size
		# self.pad_pinyin_id = pad_pinyin_id
		self.pinyin_embeddings = nn.Embedding(pinyin_vocab_size, self.embed_size, padding_idx=pad_pinyin_id)

		self.detection_layer = nn.Linear(self.hidden_size, 2)
		if not tie_weight:
			self.correction_layer = nn.Linear(self.hidden_size, self.vocab_size)
		else:
			self.correction_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
			self.correction_layer.weight = self.model.embeddings.word_embeddings.weight
			self.bias = nn.Parameter(torch.zeros(self.vocab_size))
			self.correction_layer.bias = self.bias

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, input_ids, pinyin_ids, token_type_ids=None, position_ids=None, attention_mask=None):

		# input_ids : [batch size, sequence length]
		# attention_mask : [batch size, sequence length]
		device = input_ids.device if input_ids is not None else input_ids.device

		if attention_mask is None:
			attention_mask = (input_ids != self.pad_token_id).long()
			attention_mask.to(device)

		assert attention_mask.shape == input_ids.shape, f"input_ids shape != attention_mask shape"

		# 对应 BertModel 中的 get_extended_attention_mask(attention_mask, input_shape, device)
		attention_mask = attention_mask[:, None, None, :]
		attention_mask = attention_mask.to(torch.float32)
		attention_mask = (1.0 - attention_mask) * -10000.0

		# embedding_output : [batch size, sequence length, hidden size]
		embedding_output = self.model.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
		pinyin_embedding_output = self.pinyin_embeddings(pinyin_ids)

		# detection_outputs['last_hidden_state'] : [batch size, sequence length, hidden size]
		detection_outputs = self.model.encoder(embedding_output, attention_mask)
		# detection_error_probs : [batch size, sequence length, 2]
		detection_logits = self.detection_layer(detection_outputs['last_hidden_state'])
		detection_error_probs = self.softmax(detection_logits)

		word_pinyin_embedding_output = detection_error_probs[:,:, 0:1] * embedding_output + \
			detection_error_probs[:,:,1:2] * pinyin_embedding_output

		# correction_outputs['last_hidden_state'] : [batch size, sequence length, hidden size]
		correction_outputs = self.model.encoder(word_pinyin_embedding_output, attention_mask)
		correction_logits = self.correction_layer(correction_outputs['last_hidden_state'])

		return detection_error_probs, correction_logits, detection_logits


if __name__ == "__main__":

	model = Ernie('ernie', 1000, 1, tie_weight=True)

	inputs_id = torch.LongTensor([[1,2,3,4,5,6,0,0],
	                               [3,4,5,6,0,0,0,0]])

	output = model(inputs_id, inputs_id)

	print(model)

	# model = AutoModel.from_pretrained('ernie')
	# attention_mask = (inputs_id != 0).long()
	# tmp = model(inputs_id, attention_mask)
	#
	# print(tmp[0].shape)

