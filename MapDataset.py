# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/12/2 10:50 下午
# @File    : MapDataset.py
# from datasets import Dataset
import math
from functools import partial
from multiprocessing import Pool, RLock
from pypinyin import lazy_pinyin, Style
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import collate_fn
from vocab import Vocab


class MapDataset(Dataset):
    """
    Wraps a map-style dataset-like object as an instance of `MapDataset`, and equips it
    with `map` and other utility methods. All non-magic methods of the raw object
    are also accessible.
    Args:
        data (list|Dataset): An object with `__getitem__` and `__len__` methods. It could
            be a list or a subclass of `paddle.io.Dataset`.
        kwargs (dict, optional): Other information to be passed to the dataset.
    For examples of this class, please see `dataset_self_defined
    <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html>`__.
    """

    def __init__(self, data, **kwargs):
        self.data = data
        self._transform_pipline = []
        self.new_data = self.data

        self.label_list = kwargs.pop('label_list', None)
        self.vocab_info = kwargs.pop('vocab_info', None)

    def _transform(self, data):
        for fn in self._transform_pipline:
            data = fn(data)
        return data

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given
        index.
        """
        return self._transform(self.new_data[
                                   idx]) if self._transform_pipline else self.new_data[idx]

    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.new_data)

    def map(self, fn, lazy=True, batched=False, num_workers=0):
        """
        Performs specific function on the dataset to transform and update every sample.
        Args:
            fn (callable): Transformations to be performed. It receives single
                sample as argument if batched is False. Else it receives all examples.
            lazy (bool, optional): If True, transformations would be delayed and
                performed on demand. Otherwise, transforms all samples at once. Note that
                if `fn` is stochastic, `lazy` should be True or you will get the same
                result on all epochs. Defaults to False.
            batched(bool, optional): If True, transformations would take all examples as
                input and return a collection of transformed examples. Note that if set
                True, `lazy` option would be ignored. Defaults to False.
            num_workers(int, optional): Number of processes for multiprocessing. If
                set to 0, it doesn't use multiprocessing. Note that if set to positive
                value, `lazy` option would be ignored. Defaults to 0.
        """

        assert num_workers >= 0, "num_workers should be a non-negative value"
        if num_workers > 0:
            with Pool(num_workers, initargs=(RLock(),)) as pool:

                def map_shard(num_workers, index, fn, batched):
                    self.shard(
                        num_shards=num_workers, index=index, contiguous=True)
                    self._map(fn=fn, lazy=False, batched=batched)
                    return self

                kwds_per_shard = [
                    dict(
                        num_workers=num_workers,
                        index=rank,
                        fn=fn,
                        batched=batched) for rank in range(num_workers)
                ]
                results = [
                    pool.apply_async(
                        map_shard, kwds=kwds) for kwds in kwds_per_shard
                ]
                transformed_shards = [r.get() for r in results]

                self.new_data = []
                for i in range(num_workers):
                    self.new_data += transformed_shards[i].new_data

            return self
        else:
            return self._map(fn, lazy=lazy, batched=batched)

    def _map(self, fn, lazy=True, batched=False):
        if batched:
            self.new_data = fn(self.new_data)
        elif lazy:
            self._transform_pipline.append(fn)
        else:
            self.new_data = [
                fn(self.new_data[idx]) for idx in range(len(self.new_data))
            ]
        return self


def convert_example(example,
                    tokenizer,
                    pinyin_vocab,
                    max_seq_length=128,
                    ignore_label=-1,
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


if __name__ == "__main__":

    data = [{'source': '我看过许多勇敢的人，不怕措折地奋斗，这种精神值得我们学习。', 'target': '我看过许多勇敢的人，不怕挫折地奋斗，这种精神值得我们学习。'}, {'source': '有一天的晚上，大家都睡得很安祥，突然一阵巨晃，我们家书贵倒了，这次地震惊动了全国。', 'target': '有一天的晚上，大家都睡得很安祥，突然一阵巨晃，我们家书柜倒了，这次地震惊动了全国。'}, {'source': '有几千个人甚至万人，无家可归，受这样大的打击，想必他们很烦脑吧！虽然发生这样的事件，但大家并没有放弃。', 'target': '有几千个人甚至万人，无家可归，受这样大的打击，想必他们很烦恼吧！虽然发生这样的事件，但大家并没有放弃。'}, {'source': '我觉得他们拥有不怕固难勇于面对的心，是不在灾区的我们所没有的。', 'target': '我觉得他们拥有不怕困难勇于面对的心，是不在灾区的我们所没有的。'}, {'source': '还记得那天零晨正在美好的睡梦中，忽然来一个地牛大翻身，吓坏了全台湾的人。', 'target': '还记得那天凌晨正在美好的睡梦中，忽然来一个地牛大翻身，吓坏了全台湾的人。'}, {'source': '一袭强烈的台风扫过大地，为大地带来了新的面貌，强壮的树木经不起大雨的打击，倒了。', 'target': '一袭强烈的台风扫过大地，为大地带来了新的面貌，强壮的树木禁不起大雨的打击，倒了。'}, {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。'}, {'source': '重新面对阳光时，又是令一种气象，我们不就该像大自然一样坚强地面对未来吗？', 'target': '重新面对阳光时，又是另一种气象，我们不就该像大自然一样坚强地面对未来吗？'}]

    train_ds = MapDataset(data)
    modelPath = 'ernie'
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    pinyin_vocab = Vocab.load_vocabulary(
	    'pinyin_vocab.txt', unk_token='[UNK]', pad_token='[PAD]')
    # trans_func = partial(
	#     convert_example,
	#     tokenizer=tokenizer,
	#     pinyin_vocab=pinyin_vocab,
	#     max_seq_length=128)
    #
    # train_ds.map(trans_func)
    #
    # print(trans_func)

    examples = [{'source': '我看过许多勇敢的人，不怕措折地奋斗，这种精神值得我们学习。',
                'target': '我看过许多勇敢的人，不怕挫折地奋斗，这种精神值得我们学习。'},
               {'source':"我的心情顿时感到雀跃不己，因为，再过三秒，我就要听到钟声啦！无情的暴雨在不停的下",
                "target":"我的心情顿时感到雀跃不已，因为，再过三秒，我就要听到钟声啦！无情的暴雨在不停的下"}]

    res = [convert_example(example, tokenizer, pinyin_vocab, 128, ignore_label=-100) for example in examples]

    t = collate_fn(res, tokenizer.pad_token_id, pinyin_vocab.token_to_idx[pinyin_vocab.pad_token], ignore_index=-100)


    pass


