import sys
import os

import argparse
import numpy as np
import ufet
import torch

class ETUtils(object):
    def __init__(self, args, glove=None):
        with open(args.et_config, 'r') as fin:
            self.et_params = ufet.parser.parse_args(fin.read().split())
        self.model = ufet.Model(self.et_params, ufet.constant.ANSWER_NUM_DICT[self.et_params.goal])
        self.char_vocab, self.glove_dict = ufet.data_utils.get_vocab(glove)
        self.word2id = ufet.constant.ANS2ID_DICT[self.et_params.goal]
        self.id2word = ufet.constant.ID2ANS_DICT[self.et_params.goal]
        self.answer_num = ufet.constant.ANSWER_NUM_DICT[self.et_params.goal]
        if args.use_cuda:
            self.model.cuda()
        self.ENT_LABEL = "?" #entity"
        self.dataset = args.dataset

    def getBertTails(self, heads, relations, batch_size=-1, tails=None, num_types=-1):
        if tails is None:
            tails = [self.ENT_LABEL for head in heads]
        queries = self.prepareOneForET(heads, relations, tails, target='tail')
        batch = ufet.data_utils.get_example(
                queries,
                self.glove_dict,
                self.et_params.batch_size,
                self.answer_num,
                eval_data=True,
                lstm_type=self.et_params.lstm_type,
                simple_mention=not self.et_params.enhanced_mention,
                gen=False,
            )
        batch = next(batch)
        embs = self.getBatchEmb(batch, num_types)
        return embs
 
    def getBertHeads(self, tails, relations, batch_size=-1, heads=None, num_types=-1):
        if heads is None:
            heads = [self.ENT_LABEL for tail in tails]
        queries = self.prepareOneForET(heads, relations, tails, target='head')
        batch = ufet.data_utils.get_example(
                queries,
                self.glove_dict,
                self.et_params.batch_size,
                self.answer_num,
                eval_data=True,
                lstm_type=self.et_params.lstm_type,
                simple_mention=not self.et_params.enhanced_mention,
                gen=False,
            )
        batch = next(batch)
        embs = self.getBatchEmb(batch, num_types)
        return embs

    def getBertEmbs(self, heads, relations, tails, batch_size=-1):
        tail_emb = self.getBertTails(heads, relations, batch_size, tails)
        head_emb = self.getBertHeads(tails, relations, batch_size, heads)
        # tail_emb = self.getBertTails(heads, relations, batch_size)
        # head_emb = self.getBertHeads(tails, relations, batch_size)
        all_emb = torch.cat([tail_emb, head_emb], 1).reshape(len(heads)+len(tails), head_emb.shape[1])
        return all_emb

    def prepareOneForET(self, heads, rels, tails, target='tail'):
        num_points = len(heads)
        queries = [[],[],[],[],[],[]]
        count = 0
        delim1 = ' '
        for idx in range(num_points):
            count += 1
            annot_id = self.dataset + "_%s_%d" % (target, count)
            if target  == 'tail':
                left_context_token = heads[idx].split(delim1)+rels[idx].split(delim1)
                right_context_token = []
                mention_span = [self.char_vocab[x] for x in list(tails[idx])]
                mention_seq = tails[idx].split()
            else:
                left_context_token = []
                right_context_token = rels[idx].split(delim1) + tails[idx].split(delim1)
                mention_span = [self.char_vocab[x] for x in list(heads[idx])]
                mention_seq = heads[idx].split()
            y_str = ['politician']
            y_ids = [self.word2id[x] for x in y_str if x in self.word2id]
            queries[0].append(annot_id)
            queries[1].append(left_context_token)
            queries[2].append(right_context_token)
            queries[3].append(mention_seq)
            queries[4].append(y_ids)
            queries[5].append(mention_span)
        return zip(*queries)

    def getTypsFromLogits(self, logits, k=10):
        _, indices = torch.topk(logits, k, dim=-1, largest=True, sorted=True)
        indices = indices.cpu().numpy()
        typs = []
        for i in range(indices.shape[0]):
            typs.append([self.id2word[idx] for idx in indices[i,:]])
        return typs

    def getBatchEmb(self, batch, num_types=-1):
        eval_batch, annot_ids = ufet.data_utils.to_torch(batch)
        self.model.eval()
        with torch.no_grad():
            loss, output_logits = self.model(eval_batch, self.et_params.goal)
            if num_types > 0:
                # embs = [typs[0] for typs in ufet.model_utils.get_output_index(output_logits)]
                embs = self.getTypsFromLogits(output_logits, num_types)
            else:
                embs = self.model.sigmoid_fn(output_logits) #.data.cpu().clone().numpy()
                # embs = output_logits
        return embs

    def prepareForET(self, heads, relations, tails):
        queries = {}
        queries['head'] = self.prepareOneForET(heads, relations, tails, target='head')
        queries['tail'] = self.prepareOneForET(heads, relations, tails, target='tail')
        batches = {}
        embs = {}
        for key in ['head', 'tail']:
            batches[key] = next(ufet.data_utils.get_example(
                queries[key],
                self.glove_dict,
                self.et_params.batch_size,
                self.answer_num,
                eval_data=True,
                lstm_type=self.et_params.lstm_type,
                simple_mention=not self.et_params.enhanced_mention,
                gen=False,
            ))
            embs[key] = self.getBatchEmb(batches[key])
        return embs

    def getPhraseEncodings(self, phrases, batch_size=-1):
        raise NotImplementedError 