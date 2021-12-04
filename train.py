# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/12/4 8:07 下午
# @File    : train.py
import os
import sys
import time

import torch
import logging
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from metrics import DetectionF1, CorrectionF1

logger = logging.getLogger(__name__)


def trainer(model, train_data_loader, eval_data_loader, args, device, vocab_size):
    num_training_steps = args.max_steps if args.max_steps > 0 else len(train_data_loader
                                                                       ) * args.epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_training_steps * args.warmup_proportion),
                                                num_training_steps=num_training_steps)

    global_steps = 1
    best_f1 = -1
    tic_train = time.time()

    model = model.to(device)
    ce_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            model.train()
            input_ids, pinyin_ids, det_labels, corr_labels, length = tuple(t.to(device) for t in batch)
            det_error_probs, corr_logits, det_logits = model(input_ids, pinyin_ids)

            det_loss = ce_loss(det_logits.view(-1, 2), det_labels.view(-1))
            corr_loss = ce_loss(corr_logits.view(-1, vocab_size), corr_labels.view(-1))
            loss = det_loss + corr_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_steps, epoch, step, loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_steps % args.save_steps == 0:

                logger.info("Eval:")
                det_f1, corr_f1 = evaluate(model, eval_data_loader, device)
                f1 = (det_f1 + corr_f1) / 2
                model_file = "model_%d" % global_steps
                if f1 > best_f1:
                    # save best model
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir,
                                            "best_model.pth"))
                    logger.info("Save best model at {} step.".format(
                        global_steps))
                    best_f1 = f1
                    model_file = model_file + "_best"
                """
                model_file = model_file + ".pth"
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, model_file))
                logger.info("Save model at {} step.".format(global_steps))
                """
            if 0 < args.max_steps <= global_steps:
                sys.exit(0)
            global_steps += 1


@torch.no_grad()
def evaluate(model, eval_data_loader, device):
    model.eval()
    det_metric = DetectionF1()
    corr_metric = CorrectionF1()
    for step, batch in enumerate(eval_data_loader, start=1):
        input_ids, pinyin_ids, det_labels, corr_labels, length = tuple(t.to(device) for t in batch)
        det_error_probs, corr_logits, det_logits = model(input_ids, pinyin_ids)

        det_metric.update(det_error_probs, det_labels, length)
        corr_metric.update(det_error_probs, det_labels, corr_logits,
                           corr_labels, length)

    det_f1, det_precision, det_recall = det_metric.accumulate()
    corr_f1, corr_precision, corr_recall = corr_metric.accumulate()
    logger.info("Sentence-Level Performance:")
    logger.info("Detection  metric: F1={:.4f}, Recall={:.4f}, Precision={:.4f}".
                format(det_f1, det_recall, det_precision))
    logger.info("Correction metric: F1={:.4f}, Recall={:.4f}, Precision={:.4f}".
                format(corr_f1, corr_recall, corr_precision))
    return det_f1, corr_f1
