# ERNIE for Chinese Spelling Correction

## 简介
本文主要用PyTorch实现了百度在ACL 2021上提出结合拼音特征的Softmask策略的中文错别字纠错的下游任务网络，并提供预训练模型，模型结构如下：

![image](https://user-images.githubusercontent.com/10826371/131974040-fc84ec04-566f-4310-9839-862bfb27172e.png)

"Correcting Chinese Spelling Errors with Phonetic Pre-training"非官方实现。

* 注：论文中暂未开源融合字音特征的预训练模型参数(即MLM-phonetics)，所以本文提供的纠错模型是在ERNIE-1.0的参数上进行Finetune，纠错模型结构与论文保持一致。

### 训练数据

该模型在SIGHAN简体版数据集以及[Automatic Corpus Generation生成的中文纠错数据集](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)上进行Finetune训练。本仓库已经把原始的语料进行处理，即可以直接用本仓库提供的语料进行训练。

### 单卡训练

```python
python train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5  --max_seq_length 192
```

