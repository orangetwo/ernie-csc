# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/12/3 4:51 下午
# @File    : utils.py
import torch
from pypinyin import lazy_pinyin, Style
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_train_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            source, target = line.strip('\n').split('\t')[0:2]
            yield {'source': source, 'target': target}


# input_ids, token_type_ids, pinyin_ids, detection_labels, correction_labels, length
def collate_fn(examples, pad_token_id, pad_pinyin_id, ignore_index=-100):
    input_ids = [torch.LongTensor(example[0]) for example in examples]
    pinyin_ids = [torch.LongTensor(example[2]) for example in examples]
    detection_labels = [torch.LongTensor(example[3]) for example in examples]
    correction_labels = [torch.LongTensor(example[4]) for example in examples]

    input_ids = pad_sequence(input_ids, padding_value=pad_token_id, batch_first=True)
    pinyin_ids = pad_sequence(pinyin_ids, padding_value=pad_pinyin_id, batch_first=True)
    detection_labels = pad_sequence(detection_labels, padding_value=ignore_index, batch_first=True)
    correction_labels = pad_sequence(correction_labels, padding_value=ignore_index, batch_first=True)

    lengths = torch.LongTensor([example[5] for example in examples])

    return input_ids, pinyin_ids, detection_labels, correction_labels, lengths


def convert_example(example,
                    tokenizer,
                    pinyin_vocab,
                    max_seq_length=128,
                    ignore_label=-100,
                    is_test=False):
    source = example["source"]
    words = list(source)
    if len(words) > max_seq_length - 2:
        words = words[:max_seq_length - 2]
    length = len(words)
    words = ['[CLS]'] + words + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(words)
    token_type_ids = [0] * len(input_ids)

    # Use pad token in pinyin emb to map word emb [CLS], [SEP]
    pinyins = lazy_pinyin(
        source, style=Style.TONE3, neutral_tone_with_five=True)
    pinyin_ids = [0]
    # Align pinyin and chinese char
    # 对于长度不为1的字符(不太理解这种情况)或不为中文的字符 将pinyin_vocab['UNK']或pinyin['PAD']添加至pinyin_ids
    pinyin_offset = 0
    for i, word in enumerate(words[1:-1]):
        pinyin = '[UNK]' if word != '[PAD]' else '[PAD]'
        if len(word) == 1 and is_chinese_char(ord(word)):
            while pinyin_offset < len(pinyins):
                current_pinyin = pinyins[pinyin_offset][:-1]
                pinyin_offset += 1
                if current_pinyin in pinyin_vocab:
                    pinyin = current_pinyin
                    break
        pinyin_ids.append(pinyin_vocab[pinyin])

    pinyin_ids.append(0)
    assert len(input_ids) == len(
        pinyin_ids), "length of input_ids must be equal to length of pinyin_ids"

    if not is_test:
        target = example["target"]
        correction_labels = list(target)
        if len(correction_labels) > max_seq_length - 2:
            correction_labels = correction_labels[:max_seq_length - 2]
        correction_labels = tokenizer.convert_tokens_to_ids(correction_labels)
        correction_labels = [ignore_label] + correction_labels + [ignore_label]

        detection_labels = []
        for input_id, label in zip(input_ids[1:-1], correction_labels[1:-1]):
            detection_label = 0 if input_id == label else 1
            detection_labels += [detection_label]
        detection_labels = [ignore_label] + detection_labels + [ignore_label]
        return input_ids, token_type_ids, pinyin_ids, detection_labels, correction_labels, length
    else:
        return input_ids, token_type_ids, pinyin_ids, length


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


class MyData(Dataset):
    def __init__(self, data):
        super(MyData, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
