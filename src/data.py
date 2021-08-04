import pathlib
import numpy as np
import torch

class load_data():
    def __init__(self, args):
        self.args = args
        self.data_files = self.args.data_files
        self.add_reverse = args.reverse
        self.inverse2rel_map = {}


        self.fetch_data()

    def get_relation_phrases(self,file_path):
        f = open(file_path,"r").readlines()
        phrase2id = {}
        id2phrase = {}
        word2id = []
        phrases = []
        seen_words = set()
        for line in f[1:]:
            phrase,ID = line.strip().split("\t")
            phrases.append(phrase)
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
            for word in phrase.split():
                if word not in seen_words:
                    word2id.append(word)
                    seen_words.add(word)
        cur_id = max(id2phrase.keys())
        if self.add_reverse:
            for phrase in phrases:
                newphrase =  "inverse of " + phrase
                cur_id += 1 
                phrase2id[newphrase] = cur_id
                id2phrase[cur_id] = newphrase
                self.inverse2rel_map[cur_id] = phrase2id[phrase]
        return phrase2id,id2phrase,word2id

    def get_phrases(self,file_path):
        f = open(file_path,"r").readlines()
        phrase2id = {}
        id2phrase = {}
        word2id = []
        seen_words = set()
        for line in f[1:]:
            phrase,ID = line.strip().split("\t")
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
            for word in phrase.split():
                if word not in seen_words:
                    word2id.append(word)
                    seen_words.add(word)
        return phrase2id,id2phrase,word2id

    def get_phrase2word(self,phrase2id,word2id):
        phrase2word = {}
        for phrase in phrase2id:
            words = []
            for word in phrase.split(): words.append(word2id[word])
            phrase2word[phrase2id[phrase]] = words
        return phrase2word


    def get_word_embed(self,word2id,GLOVE_PATH):
        word_embed = {}
        if pathlib.Path(GLOVE_PATH).is_file():
            print("utilizing pre-trained word embeddings")
            with open(GLOVE_PATH, encoding="utf8") as f:
                for line in f:
                    word, vec = line.split(' ', 1)
                    if word in word2id:
                        word_embed[word2id[word]] = np.fromstring(vec, sep=' ')
        else: print("word embeddings are randomly initialized")

        wordkeys = word2id.values()
        a = [words for words in wordkeys if words not in word_embed]
        for word in a:
            word_embed[word] = np.random.normal(size = 300)

        self.embed_matrix  = np.zeros((len(word_embed),300))
        for word in word_embed:
            self.embed_matrix[word] = word_embed[word]

    def get_train_triples(self,triples_path,entid2clustid,rel2id,id2rel):
        trip_list = []
        label_graph = {}
        self.label_filter = {}
        f = open(triples_path,"r").readlines()
        for trip in f[1:]:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            if self.add_reverse:
                r_inv = "inverse of " + id2rel[r]
                if r_inv not in rel2id:
                    import pdb; pdb.set_trace()
                    ID = len(rel2id)
                    rel2id[r_inv] = ID

                r_inv = rel2id[r_inv]

            if (e1,r) not in label_graph:
                label_graph[(e1,r)] = set()
            label_graph[(e1,r)].add(e2)

            if self.add_reverse:
                if (e2,r_inv) not in label_graph:
                    label_graph[(e2,r_inv)] = set()
                label_graph[(e2,r_inv)].add(e1)

            if (e1,r) not in self.label_filter:
                self.label_filter[(e1,r)] = set()
            self.label_filter[(e1,r)].add(entid2clustid[e2])

            if self.add_reverse:
                if (e2,r_inv) not in self.label_filter:
                    self.label_filter[(e2,r_inv)] = set()
                self.label_filter[(e2,r_inv)].add(entid2clustid[e1])

            trip_list.append([e1,r,e2])
            if self.add_reverse:
                trip_list.append([e2,r_inv,e1])
        return np.array(trip_list),rel2id,label_graph

    def get_test_triples(self, triples_path,rel2id,id2rel):
        trip_list = []
        f = open(triples_path,"r").readlines()
        for trip in f[1:]:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            trip_list.append([e1,r,e2])
            if self.add_reverse:
                r_inv = "inverse of " + id2rel[r]
                if r_inv not in rel2id:
                    import pdb; pdb.set_trace()
                    ID = len(rel2id)
                    rel2id[r_inv] = ID

                trip_list.append([e2,rel2id[r_inv],e1])
        return np.array(trip_list)

    def get_clusters(self,clust_path):
        ent_clusts = {}
        entid2clustid = {}
        ent_list = []
        unique_clusts = []
        ID = 0
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            clust = [int(ent) for ent in line[2:]]
            ent_clusts[int(line[0])] = clust
            if line[0] not in ent_list:
                ID+=1
                unique_clusts.append(clust)
                ent_list.extend(line[2:])
                for ent in clust: entid2clustid[ent] = ID
        return ent_clusts, entid2clustid, unique_clusts

    def get_edges(self,ent_clusts):
        head_list = []
        tail_list = []
        for ent in ent_clusts:
            if self.args.model_variant=='CaRe' and len(ent_clusts[ent])==1:
                head_list.append(ent)
                tail_list.append(ent)
            for neigh in ent_clusts[ent]:
                if neigh!=ent:
                    head_list.append(neigh)
                    tail_list.append(ent)

        head_list = np.array(head_list).reshape(1,-1)
        tail_list = np.array(tail_list).reshape(1,-1)

        self.edges = np.concatenate((np.array(head_list),np.array(tail_list)),axis = 0)


    def fetch_data(self):
        self.rel2id,self.id2rel,self.word2id = self.get_relation_phrases(self.data_files["rel2id_path"])
        # self.rel2id,self.id2rel,self.word2id = self.get_phrases(self.data_files["rel2id_path"])
        self.ent2id,self.id2ent,_ = self.get_phrases(self.data_files["ent2id_path"])

        self.canon_clusts,_,self.unique_clusts = self.get_clusters(self.data_files["cesi_npclust_path"])
        self.true_clusts, self.entid2clustid,_ = self.get_clusters(self.data_files["gold_npclust_path"])

        self.get_edges(self.canon_clusts)

        self.train_trips,self.rel2id,self.label_graph = self.get_train_triples(self.data_files["train_trip_path"],
                                                                               self.entid2clustid,self.rel2id,
                                                                               self.id2rel)

        self.test_trips = self.get_test_triples(self.data_files["test_trip_path"],self.rel2id, self.id2rel)
        self.valid_trips = self.get_test_triples(self.data_files["valid_trip_path"],self.rel2id, self.id2rel)


        # self.word2id.add("inverse")
        # self.word2id.add("of")
        # self.word2id = {word:index for index,word in enumerate(list(self.word2id))}
        if "inverse" not in self.word2id:
            self.word2id.append("inverse")
        if "of" not in self.word2id:
            self.word2id.append("of")
        self.word2id = {word:index for index,word in enumerate(self.word2id)}
        self.word2id['<PAD>'] = len(self.word2id)

        self.rel2word = self.get_phrase2word(self.rel2id,self.word2id)

        self.get_word_embed(self.word2id,self.data_files["glove_path"])

    def get_bert_embs(self, bert, samples):
        num_rels = len(self.id2rel)//2
        ents = samples[:,0]
        rels = samples[:,1]
        heads = ents[rels<num_rels]
        non_inverse_rels = rels[rels<num_rels]
        tails = ents[rels>=num_rels]
        inverse_rels = [self.inverse2rel_map[rel] for rel in rels[rels>=num_rels]]

        heads = [self.id2ent[head] for head in heads]
        tails = [self.id2ent[head] for head in tails]
        non_inverse_rels = [self.id2rel[rel] for rel in non_inverse_rels]
        inverse_rels = [self.id2rel[rel] for rel in inverse_rels]
        bert_tail_embs = bert.getBertTails(heads, non_inverse_rels)
        bert_head_embs = bert.getBertHeads(tails, inverse_rels)
        ret_embs = torch.zeros((samples.shape[0], bert_tail_embs.shape[1]), dtype=bert_tail_embs.dtype)
        ret_embs[(rels<num_rels).nonzero()] = bert_tail_embs
        ret_embs[(rels>=num_rels).nonzero()] = bert_head_embs
        return ret_embs

    def fetch_bert_embs(self, bert):
        self.bert_tail_embs = {}
        for sp, trips in zip(['train', 'valid', 'test'], [self.train_trips, self.valid_trips, self.test_trips]):
            # heads = [self.id2ent[head] for head in trips[:,0]]
            # relations = [self.id2rel[rel] for rel in trips[:,1]]
            if sp in ['train']:
                continue
            if self.add_reverse:
                non_reverse_trips = trips[::2]
            else:
                non_reverse_trips = trips
            heads = [self.id2ent[head] for head in non_reverse_trips[:,0]]
            relations = [self.id2rel[rel] for rel in non_reverse_trips[:,1]]
            tails = [self.id2ent[tail] for tail in non_reverse_trips[:,2]]
            if self.add_reverse:
                tail_embs = bert.getBertEmbs(heads, relations, tails, self.args.batch_size)
            else:
                tail_embs = bert.getBertTails(heads, relations, self.args.batch_size)
            if self.args.use_cuda:
                tail_embs = tail_embs.cuda()
            self.bert_tail_embs[sp] = tail_embs

    def get_all_tails(self):
        all_tails = np.concatenate([self.train_trips[:,2],
                               self.valid_trips[:,2],
                               self.test_trips[:,2]])
        return set(all_tails.tolist())

    def get_all_ents(self):
        all_tails = np.array(list(self.get_all_tails()), dtype=np.int32)
        all_heads = np.concatenate([self.train_trips[:,0],
                                    self.valid_trips[:,0],
                                    self.test_trips[:,0]])
        all_ents = np.concatenate([all_heads, all_tails])
        return set(all_ents.tolist())
