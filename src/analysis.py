import argparse
import multiprocessing as mp
import os
import sys
import json

from matplotlib import pyplot as plt

import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import pickle
import models
from tabulate import tabulate


def findkNN(queries, keys, k=5, normalize=True):
    if normalize:
        keynorms = np.linalg.norm(keys, ord=2, axis=1)
        keys = keys/np.expand_dims(keynorms, 1)
        querynorms = np.linalg.norm(queries, ord=2, axis=1)
        queries = queries/np.expand_dims(querynorms, 1)

    nbrs = NearestNeighbors(n_neighbors=k).fit(keys)
    _, indices = nbrs.kneighbors(queries)
    return indices

def plotTSNE(tail_emb, query_emb, values_embs, all_embs, names, fname=None):
    colors = {'query':'r', 'value':'b', 'all':'#999966', 'tail':'g'}
    fig, ax = plt.subplots()
    plt.figure(figsize=(8,8))
    # plot the data points
    plt.scatter(tail_emb[0], tail_emb[1], c=colors['tail'])
    plt.scatter(query_emb[0], query_emb[1], c=colors['query'])
    plt.scatter(values_embs[:,0], values_embs[:,1], c=colors['value'])
    plt.scatter(all_embs[:,0], all_embs[:,1], c=colors['all'], s=1)

    # plot the names
    fontsize = 15
    weight = 'medium'
    plt.annotate(names['tail'], (tail_emb[0], tail_emb[1]), fontsize=fontsize, weight=weight)
    plt.annotate(names['query'], (query_emb[0], query_emb[1]), fontsize=fontsize, weight=weight)
    for i in range(values_embs.shape[0]):
        plt.annotate(names['values'][i], (values_embs[i, 0], values_embs[i, 1]), fontsize=fontsize, weight=weight)
    plt.xlabel('TSNE Dim-1', weight=weight, fontsize=fontsize)
    plt.ylabel('TSNE Dim-2', weight=weight, fontsize=fontsize)
    plt.title(names['dataset'], weight=weight, fontsize=fontsize)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

def findTSNEProjections(bert_type_embs, tail_type_embs, pca=False):
    if pca:
        tsne = PCA(n_components=2)
    else:
        tsne = TSNE(n_components=2, n_jobs=mp.cpu_count()-1)
    proj = tsne.fit_transform(np.vstack([bert_type_embs, tail_type_embs]))
    bert = proj[:bert_type_embs.shape[0], :]
    care = proj[bert_type_embs.shape[0]:,:]
    return bert, care

def findTSNE(bert_type_embs, care_type_embs, fname=None):
    tsne_bert = TSNE(n_components=2)
    tsne_care = TSNE(n_components=2)
    bert = tsne_bert.fit_transform(bert_type_embs)
    care = tsne_care.fit_transform(care_type_embs)
    plt.figure(figsize=(8,8))
    plt.scatter(bert[:,0], bert[:,1], c='b')
    plt.scatter(care[:,0], care[:,1], c='c')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

def genErrorAnalysis(name, basedir, split):
    predsfile = os.path.join(basedir, 'preds', '%s.%s.txt' % (name,split))
    ranksfile = os.path.join(basedir, 'ranks', '%s.%s.npy' % (name,split))
    delim1 = "\t"
    ranks = np.load(ranksfile)[:,3]
    with open(predsfile, 'r') as fin:
        linenum = 0
        for line in fin:
            line = line.strip()
            if line:
                line = line.split(delim1)
                newline = line[:3] + [str(ranks[linenum])] + line[3:]
                linenum += 1
                print(delim1.join(newline))

def findTopEntsForTyps(typ_preds, data, id2typ, k=5):
    num_typs = typ_preds.shape[1]
    typ2ents = {}
    for typ_idx in range(num_typs):
        ent_idxs = np.argpartition(typ_preds[:,typ_idx], -k)[-k:]
        typ2ents[id2typ[str(typ_idx)]] = [data.id2ent[idx] for idx in ent_idxs] 
    return typ2ents


def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="name of the run", default="testrun_id")
    parser.add_argument("--basedir", type=str, help="results base directory", default="./results")
    parser.add_argument("--split", type=str, help="data split [valid/test]", default="valid")
    return parser

def main():
    parser = getParser()
    try:
        params = parser.parse_args()
    except:
        # parser.print_help()
        sys.exit(1)
    genErrorAnalysis(params.name, params.basedir, params.split)

if __name__ == "__main__":
    main()
