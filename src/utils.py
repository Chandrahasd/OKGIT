import sys
import os
import random
import copy
import time
import json
from collections import MutableMapping

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
import torch.nn.functional as F

import warnings
import logging
import logging.config
warnings.filterwarnings("ignore")

def seq_batch(phrase_id, args, phrase2word):
    phrase_batch = np.ones((len(phrase_id),11),dtype = int)*args.pad_id
    phrase_len = torch.LongTensor(len(phrase_id))

    for i,ID in enumerate(phrase_id):
        phrase_batch[i,0:len(phrase2word[ID])] = np.array(phrase2word[ID])
        phrase_len[i] = len(phrase2word[ID])

    phrase_batch = torch.from_numpy(phrase_batch)
    phrase_batch = Variable(torch.LongTensor(phrase_batch))
    phrase_len = Variable(phrase_len)

    if args.use_cuda:
        phrase_batch = phrase_batch.cuda()
        phrase_len = phrase_len.cuda()

    return phrase_batch, phrase_len


def get_next_batch(id_list, data, args, train):
    entTotal = args.num_nodes
    samples = []
    labels = np.zeros((len(id_list),entTotal))
    for i in range(len(id_list)):
        trip = train[id_list[i]]
        samples.append([trip[0],trip[1]])
        pos_ids = list(data.label_graph[(trip[0],trip[1])])
        labels[i][pos_ids] = 1
    return np.array(samples),labels

def get_rank(scores, clust, Hits, entid2clustid, filter_clustID, candidates, K=10):
    hits = np.ones((len(Hits)))
    scores = np.argsort(scores)
    rank = 1
    high_rank_clust = set()
    for i in range(scores.shape[0]):
        if scores[i] not in candidates:
            continue
        if scores[i] in clust:
            break
        else:
            if entid2clustid[scores[i]] not in high_rank_clust and entid2clustid[scores[i]] not in filter_clustID:
                rank+=1
                high_rank_clust.add(entid2clustid[scores[i]])
    for i,r in enumerate(Hits):
        if rank>r:
            hits[i]=0
        else:
            break
    count = 0
    top_cands = np.zeros((K,), dtype=np.int32)
    for score in scores:
        if score not in candidates:
            continue
        else:
            top_cands[count] = score
            count += 1
        if count >= K:
            break
    # return rank, hits, scores[:K]
    return rank, hits, top_cands

def evaluate(model, entTotal, test_trips, args, data, bert_tail_embs, K=10, only_tails=False):
    ents = torch.arange(0, entTotal, dtype=torch.long)
    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    head = test_trips[:,0]
    rel = test_trips[:,1]
    tail = test_trips[:,2]
    id2ent = data.id2ent
    id2rel = data.id2rel
    true_clusts = data.true_clusts
    entid2clustid = data.entid2clustid
    ent_filter = data.label_filter
    bs = args.batch_size

    edges = torch.tensor(data.edges,dtype=torch.long)
    if args.use_cuda:
        ents = ents.cuda()
        edges = edges.cuda()

    r_embed,ent_embed = model.get_embed(edges,ents,rel)

    test_scores = np.zeros((test_trips.shape[0],entTotal))
    n_batches = int(test_trips.shape[0]/bs) + 1
    for i in range(n_batches):
        ent = head[i*bs:min((i+1)*bs,test_trips.shape[0])]
        ent = ent_embed[ent,:]
        r = r_embed[i*bs:min((i+1)*bs,test_trips.shape[0]),:]
        bert_embed = bert_tail_embs[i*bs:min((i+1)*bs,test_trips.shape[0]),:]
        scores = model.get_scores(ent,r,ent_embed,ent.shape[0], bert_embed).cpu().data.numpy()
        test_scores[i*bs:min((i+1)*bs,test_trips.shape[0]),:] = scores

    ranked_cands = np.zeros((test_trips.shape[0], K), dtype=np.int32)
    if only_tails:
        candidates = data.get_all_tails()
    else:
        candidates = data.get_all_ents()
    for j in range(test_trips.shape[0]):
        sample_scores = -test_scores[j,:]
        t_clust = set(true_clusts[tail[j]])
        _filter = []
        if (head[j],rel[j]) in ent_filter:
            _filter = ent_filter[(head[j],rel[j])]
        if j%2==1:
            H_r, H_h, cur_ranked_cands = get_rank(sample_scores, t_clust, args.Hits, entid2clustid, _filter, candidates, K)
            H_Rank.append(H_r)
            H_inv_Rank.append(1/H_r)
            H_Hits += H_h
        else:
            T_r, T_h, cur_ranked_cands = get_rank(sample_scores, t_clust, args.Hits, entid2clustid, _filter, candidates, K)
            T_Rank.append(T_r)
            T_inv_Rank.append(1/T_r)
            T_Hits += T_h
        ranked_cands[j,:] = cur_ranked_cands
    mean_rank_head = np.mean(np.array(H_Rank))
    mean_rank_tail = np.mean(np.array(T_Rank))
    mean_rank = 0.5*(mean_rank_head+mean_rank_tail) 
    mean_inv_rank_head = np.mean(np.array(H_inv_Rank))
    mean_inv_rank_tail = np.mean(np.array(T_inv_Rank))
    mean_inv_rank = 0.5*(mean_inv_rank_head+mean_inv_rank_tail)
    hits_at_head = {}
    hits_at_tail = {}
    hits_at = {}
    for i, hits in enumerate(args.Hits):
        hits_at_head[hits] = H_Hits[i]/len(H_Rank)
        hits_at_tail[hits] = T_Hits[i]/len(T_Rank)
        hits_at[hits] = 0.5*(hits_at_head[hits]+hits_at_tail[hits])
    # hits_1 = T_Hits[0]/len(T_Rank)
    # hits_10 = T_Hits[1]/len(T_Rank)
    print("%f %f %f %f %f" % (mean_inv_rank, mean_rank, hits_at[1], hits_at[3], hits_at[10]))
    perf = {'mr': mean_rank,
            'mrr': mean_inv_rank,
            'hits@': hits_at,
            'head_mr': mean_rank_head,
            'head_mrr': mean_inv_rank_head,
            'head_hits@': hits_at_head,
            'tail_mr': mean_rank_tail,
            'tail_mrr': mean_inv_rank_tail,
            'tail_hits@': hits_at_tail,
            }
    return perf, {'tail':T_Rank, 'head':H_Rank}, ranked_cands

class dummy_context_manager():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_logger(file_name, log_dir, config_dir):
    config_dict = json.load(open( os.path.join(config_dir, 'log_config.json')))
    config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir,  file_name.replace('/', '-'))
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(file_name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

