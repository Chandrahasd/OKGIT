import sys
import os

import argparse
import numpy as np
from lama.modules import build_model_by_name
from lama.modules.base_connector import *
import torch

class BertUtils(object):
    def __init__(self, args):
        self.model = build_model_by_name(args.models, args)
        if args.use_cuda:
            self.model.try_cuda()

    def getBertTails(self, heads, relations, batch_size=-1):
        num_sentences = len(heads)
        sentence_list = []
        for idx in range(num_sentences):
            sentence = [" ".join([heads[idx], relations[idx], MASK])+"."]
            # sentence = [" ".join([heads[idx], relations[idx], MASK])]
            sentence_list.append(sentence)
        if batch_size < 0:
            context_embeddings, sentence_lengths, tokenized_text_list, masked_indices = self.model.get_contextual_embeddings_with_mask_indices(sentence_list)
            context_embeddings = context_embeddings[-1]
            bert_tail_embs = []
            for idx in range(num_sentences):
                 bert_tail_embs.append(context_embeddings[idx, masked_indices[idx][0], :])
        else:
            bert_tail_embs = []
            for st_idx in range(0, num_sentences, batch_size):
                cur_sentences = sentence_list[st_idx:st_idx+batch_size]
                context_embeddings, sentence_lengths, tokenized_text_list, masked_indices = self.model.get_contextual_embeddings_with_mask_indices(cur_sentences)
                context_embeddings = context_embeddings[-1]
                for idx in range(len(cur_sentences)):
                    # bert_tail_embs.append(context_embeddings[idx, 0, :])
                    bert_tail_embs.append(context_embeddings[idx, masked_indices[idx][0], :])
        return torch.stack(bert_tail_embs)

    def getBertHeads(self, tails, relations, batch_size=-1):
        num_sentences = len(tails)
        sentence_list = []
        for idx in range(num_sentences):
            sentence = [" ".join([MASK, relations[idx], tails[idx]])+"."]
            # sentence = [" ".join([MASK, relations[idx], tails[idx]])]
            sentence_list.append(sentence)
        if batch_size < 0:
            context_embeddings, sentence_lengths, tokenized_text_list, masked_indices = self.model.get_contextual_embeddings_with_mask_indices(sentence_list)
            context_embeddings = context_embeddings[-1]
            bert_head_embs = []
            for idx in range(num_sentences):
                 bert_head_embs.append(context_embeddings[idx, masked_indices[idx][0], :])
        else:
            bert_head_embs = []
            for st_idx in range(0, num_sentences, batch_size):
                cur_sentences = sentence_list[st_idx:st_idx+batch_size]
                context_embeddings, sentence_lengths, tokenized_text_list, masked_indices = self.model.get_contextual_embeddings_with_mask_indices(cur_sentences)
                context_embeddings = context_embeddings[-1]
                for idx in range(len(cur_sentences)):
                    # bert_head_embs.append(context_embeddings[idx, 0, :])
                    bert_head_embs.append(context_embeddings[idx, masked_indices[idx][0], :])
        return torch.stack(bert_head_embs)

    def getBertEmbs(self, heads, relations, tails, batch_size=-1):
        tail_emb = self.getBertTails(heads, relations, batch_size)
        head_emb = self.getBertHeads(tails, relations, batch_size)
        all_emb = torch.cat([tail_emb, head_emb], 1).reshape(len(heads)+len(tails), head_emb.shape[1])
        return all_emb