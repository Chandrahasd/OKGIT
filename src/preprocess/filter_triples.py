import sys
import os
from shutil import copyfile

import argparse


def read_relations(dataset):
    # read entity ids
    rel2id = {}
    id2rel = {}
    delim1 = "\t"
    delim2 = " "
    first_line = True
    with open(os.path.join(dataset, 'rel2id.txt')) as fin:
        for line in fin:
            line = line.strip()
            if line:
                if first_line:
                    first_line = False
                    continue
                x = line.split(delim1)
                ent_id = int(x[1])
                ent_name = x[0].strip()
                rel2id[ent_name] = ent_id
                id2rel[ent_id] = ent_name
    return rel2id, id2rel

def read_entites(dataset):
    # read entity ids
    ent2id = {}
    id2ent = {}
    id2len = {}
    delim1 = "\t"
    delim2 = " "
    first_line = True
    with open(os.path.join(dataset, 'ent2id.txt')) as fin:
        for line in fin:
            line = line.strip()
            if line:
                if first_line:
                    first_line = False
                    continue
                x = line.split(delim1)
                ent_id = int(x[1])
                ent_name = x[0].strip()
                ent2id[ent_name] = ent_id
                id2ent[ent_id] = ent_name
                id2len[ent_id] = len(ent_name.split(delim2))
    return ent2id, id2ent, id2len

def filter_single_token_triples(params):
    bert_vocab = set()
    # load bert vocab to filter non-bert entities
    with open(params.bert_vocab, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                bert_vocab.add(line)

    ent2id, id2ent, id2len = read_entites(params.dataset)
    rel2id, id2rel = read_relations(params.dataset)

    train_file = os.path.join(params.dataset, 'train_trip.txt')
    valid_file = os.path.join(params.dataset, 'valid_trip.txt')
    test_file = os.path.join(params.dataset, 'test_trip.txt')
    filtered_train_triples, num_train = get_single_token_triples(train_file, id2ent, bert_vocab)
    filtered_valid_triples, num_valid = get_single_token_triples(valid_file, id2ent, bert_vocab)
    filtered_test_triples, num_test = get_single_token_triples(test_file, id2ent, bert_vocab)
    return {'train': filtered_train_triples,
            'valid': filtered_valid_triples,
            'test': filtered_test_triples,
            'id2rel': id2rel,
            'id2ent': id2ent,
            'rel2id': rel2id,
            'ent2id': ent2id,
            }

def get_single_token_triples(filename, id2ent, bert_vocab):
    # read triples
    triples = []
    first_line = True
    delim1 = "\t"
    delim2 = " "
    num_total = 0
    with open(filename) as fin:
        for line in fin:
            line = line.strip()
            if line:
                if first_line:
                    num_total = int(line)
                    first_line = False
                    continue
                x = line.split(delim1)
                head_id = int(x[0])
                rel_id = int(x[1])
                tail_id = int(x[2])
                tail_ent = id2ent[tail_id]
                head_ent = id2ent[head_id]
                tail_len = len(tail_ent.split(delim2))
                head_len = len(tail_ent.split(delim2))
                if tail_len == 1 and tail_ent.lower() in bert_vocab and head_len == 1 and head_ent.lower() in bert_vocab:
                    triples.append((head_id, rel_id, tail_id))
    return triples, num_total

def save_files(params, triples):
    source_dir = params.dataset
    target_dir = "%sF" % params.dataset
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in ['cesi_npclust.txt', 'ent2id.txt', 'gold_npclust.txt', 'rel2id.txt']:
        copyfile(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
    for filename in ['train', 'valid', 'test']:
        num_triple = len(triples[filename])
        with open(os.path.join(target_dir, '%s_trip.txt'%filename), 'w') as fout:
            print(num_triple, file=fout)
            for head_id, rel_id, tail_id in triples[filename]:
                print('%d\t%d\t%d' % (head_id, rel_id, tail_id), file=fout)

def expand_clusters(id2ent, cluster_file):
    cluster = []
    delim1 = '\t'
    with open(cluster_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                line = list(map(int, line.split(delim1)))
                src = id2ent[line[0]]
                cluster_size = line[1]
                dests = [id2ent[eid] for eid in line[2:]]
                cluster.append([src, cluster_size]+dests)
    return cluster

def expand_triples(triples):
    enames = triples['id2ent']
    rnames = triples['id2rel']
    filtered_triples = {}
    for spl in ['train', 'valid', 'test']:
        cur_triples = triples[spl]
        cur_filtered_triples = []
        for head_id, rel_id, tail_id in cur_triples:
            head = enames[head_id]
            rel = rnames[rel_id]
            tail = enames[tail_id]
            cur_filtered_triples.append((head, rel, tail))
        filtered_triples[spl] = cur_filtered_triples
    return filtered_triples

class Triples(object):
    def __init__(self):
        self.ents_list = []
        self.ents_set = set()
        self.ent2id = {}
        self.rel2id = {}
        self.rels_list = []
        self.rels_set = set()

    def add_rel(self, rel):
        if rel in self.rels_set:
            relid = self.rel2id[rel]
        else:
            relid = len(self.rels_list)
            self.rel2id[rel] = relid
            self.rels_list.append(rel)
            self.rels_set.add(rel)
        return relid

    def add_ent(self, ent):
        if ent in self.ents_set:
            entid = self.ent2id[ent]
        else:
            entid = len(self.ents_list)
            self.ent2id[ent] = entid
            self.ents_list.append(ent)
            self.ents_set.add(ent)
        return entid

    def getEnt2id(self):
        return self.ent2id

    def getRel2id(self):
        return self.rel2id

    def triple2id(self, triples):
        triple_ids = []
        for head, rel, tail in triples:
            head_id = self.add_ent(head)
            rel_id = self.add_rel(rel)
            tail_id = self.add_ent(tail)
            triple_ids.append((head, rel, tail))
        return triple_ids

    def clust2id(self, clusters):
        cluster_ids = []
        for cluster in clusters:
            src_id = self.ent2id.get(cluster[0])
            if src_id is None:
                continue
            dest_ids = []
            for dest in cluster[2:]:
                dest_id = self.ent2id.get(dest)
                if dest_id is None:
                    continue
                dest_ids.append(dest_id)
            if len(dest_ids) == 0:
                continue
            cluster_size = len(dest_ids)
            cluster_ids.append([src_id, cluster_size] + dest_ids)
        return cluster_ids

    def reindexTriples(self, triples):
        self.triples = {}
        for spl in ['train', 'valid', 'test']:
            self.triples[spl] = self.triple2id(triples[spl])

    def reindexClusters(self, clusters):
        self.clusters = {}
        for spl in ['cesi', 'gold']:
            self.clusters[spl] = self.clust2id(clusters[spl])

    def serializeTriples(self, outdir):
        for spl in ['train', 'valid', 'test']:
            with open(os.path.join(outdir, '%s_trip.txt'%spl), 'w') as fout:
                num_triples = len(self.triples[spl])
                print(f'{num_triples}', file=fout)
                for head, rel, tail in self.triples[spl]:
                    head = self.ent2id[head]
                    rel = self.rel2id[rel]
                    tail = self.ent2id[tail]
                    print(f'{head}\t{rel}\t{tail}', file=fout)

    def serializeCluster(self, outdir):
        delim = "\t"
        for spl in ['gold', 'cesi']:
            with open(os.path.join(outdir, '%s_npclust.txt'%spl), 'w') as fout:
                for cluster in self.clusters[spl]:
                    outstr = delim.join([str(eid) for eid in cluster])
                    print(outstr, file=fout)

    def serializeIds(self, outdir):
        num_ents = len(self.ents_list)
        with open(os.path.join(outdir, 'ent2id.txt'), 'w') as fout:
            print(f'{num_ents}', file=fout)
            for ent, entid in self.ent2id.items():
                print(f'{ent}\t{entid}', file=fout)
        num_rels = len(self.rels_list)
        with open(os.path.join(outdir, 'rel2id.txt'), 'w') as fout:
            print(f'{num_rels}', file=fout)
            for rel, relid in self.rel2id.items():
                print(f'{rel}\t{relid}', file=fout)

    def serialize(self, outdir):
        self.serializeIds(outdir)
        self.serializeTriples(outdir)
        self.serializeCluster(outdir)

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, help="source dataset directory", required=True)
    parser.add_argument("--bert-vocab", type=str, help="file containing bert vocab", default="bert_vocab.txt")
    return parser

def main():
    parser = getParser()
    try:
        params = parser.parse_args()
    except:
        sys.exit(1)
    triples = filter_single_token_triples(params)
    filtered_triples = expand_triples(triples)
    cesi_npclust = expand_clusters(triples['id2ent'], os.path.join(params.dataset, 'cesi_npclust.txt'))
    gold_npclust = expand_clusters(triples['id2ent'], os.path.join(params.dataset, 'gold_npclust.txt'))
    triples_obj = Triples()
    triples_obj.reindexTriples(filtered_triples)
    triples_obj.reindexClusters({'cesi':cesi_npclust, 'gold': gold_npclust})
    outdir = "%sF" % params.dataset
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    triples_obj.serialize(outdir)

if __name__ == "__main__":
    main()

