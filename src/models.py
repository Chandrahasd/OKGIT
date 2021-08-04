#### Import all the supporting classes
import os
import sys

from utils import *
from encoder import GRUEncoder
from cn_variants import LAN, GCN, GAT

class OKGIT(nn.Module):
    def __init__(self, args, embed_matrix, rel2words):
        super(OKGIT, self).__init__()
        self.args = args
        self.conve = CaRE(args, embed_matrix, rel2words)
        self.bert_type_projection = nn.Linear(args.bert_dim, args.type_dim, bias=True)
        self.care_type_projection = nn.Linear(args.nfeats, args.type_dim, bias=True)
        # if self.args.type_loss.lower() in ['mse']:
        #     self.type_loss = torch.nn.MSELoss()
        # else:
        #     self.type_loss = torch.nn.BCELoss()
        self.type_loss = torch.nn.BCELoss()
        self.inp_drop = torch.nn.Dropout(self.args.dropout)

    def forward(self):
        pass

    def get_scores(self, ent, rel, ent_embed, batch_size, bert_tail_embs, type_score=False):
        scores = self.conve.get_scores(ent, rel, ent_embed, batch_size)
        # ent = ent.view(-1, 1, 15, 20)
        # rel = rel.view(-1, 1, 15, 20)

        # stacked_inputs = torch.cat([ent, rel], 2)

        # stacked_inputs = self.bn0(stacked_inputs)
        # x = self.inp_drop(stacked_inputs)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = self.feature_map_drop(x)
        # x = x.view(batch_size, -1)
        # x = self.fc(x)
        # x = self.hidden_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = torch.mm(x, ent_embed.transpose(1,0))
        # x += self.b.expand_as(x)
        # return x
        bert_type = self.bert_type_projection(self.inp_drop(bert_tail_embs))
        care_type = self.care_type_projection(ent_embed)
        # type_compat = F.sigmoid(torch.matmul(bert_type, care_type.transpose(0,1)))
        if self.args.type_loss in ['mse']:
            type_compat = -1.0*torch.pow(torch.unsqueeze(bert_type,1)-torch.unsqueeze(care_type, 0), 2).sum(2)
        else:
            type_compat = torch.matmul(bert_type, care_type.transpose(0,1))

        # apply type score transformation
        if self.args.type_transform in ['sigmoid']:
            type_compat = F.sigmoid(type_compat)
        elif self.args.type_transform in ['inverse']:
            type_compat = -1.0/type_compat

        # apply triple and type score composition
        if self.args.type_composition in ['add']:
            final_scores = scores + self.args.type_composition_weight * type_compat
        else:
            final_scores = scores*type_compat

        if type_score:
            return final_scores, type_compat
        else:
            return final_scores

    def get_type_embed(self, bert_tail_embs, node_id, edges):
        bert_type_embed = self.bert_type_projection(bert_tail_embs)
        np_embed = self.conve.np_embeddings(node_id)
        if self.args.CN != 'Phi':
            np_embed = self.conve(np_embed, edges)
        care_type_embs = self.care_type_projection(np_embed)
        return bert_type_embed, care_type_embs

    def get_embed(self, edges, node_id, r):
        return self.conve.get_embed(edges, node_id, r)

    def get_loss(self, samples, labels, edges, node_id, bert_embed):

        np_embed = self.conve.np_embeddings(node_id)
        if self.args.CN != 'Phi':
            np_embed = self.conve(np_embed, edges)

        sub_embed = np_embed[samples[:,0]]
        r = samples[:,1]


        r_batch,r_len = seq_batch(r.cpu().numpy(), self.args, self.conve.rel2words)
        rel_embed = self.conve.phrase_embed_model(r_batch,r_len)

        # scores = self.conve.get_scores(sub_embed, rel_embed, np_embed, self.args.batch_size)
        # bert_type = self.bert_type_projection(bert_embed)
        # care_type = self.care_type_projection(np_embed)
        # type_scores = torch.matmul(bert_type, care_type.transpose(0,1))
        # scores = scores*type_scores

        scores, type_scores = self.get_scores(sub_embed, rel_embed, np_embed, self.args.batch_size, bert_embed, type_score=True)

        pred = F.sigmoid(scores)

        predict_loss = self.conve.loss(pred, labels)
        if self.args.type_weight > 0:
            typeloss = self.type_loss(F.sigmoid(type_scores), labels)
            # typeloss = torch.sum((F.sigmoid(type_scores) - labels)**2)
            # typeloss = typeloss/(labels.shape[0]*labels.shape[1])
            predict_loss = predict_loss + self.args.type_weight * typeloss 

        return predict_loss

class CaRE(nn.Module):
    def __init__(self, args, embed_matrix,rel2words):
        super(CaRE, self).__init__()
        self.args = args

        self.rel2words = rel2words
        self.phrase_embed_model = GRUEncoder(embed_matrix, self.args)

        if self.args.CN=='LAN':
            self.cn = LAN(self.args.nfeats, self.args.nfeats)
            # self.cn = CaRe(self.args.nfeats, self.args.nfeats)
        elif self.args.CN=='GCN':
            self.cn = CaReGCN(self.args.nfeats, self.args.nfeats)
        else:
            self.cn = CaReGAT(self.args.nfeats, self.args.nfeats//self.args.nheads, heads=self.args.nheads, dropout=self.args.dropout)


        self.np_embeddings = nn.Embedding(self.args.num_nodes, self.args.nfeats)
        nn.init.xavier_normal_(self.np_embeddings.weight.data)

        self.inp_drop = torch.nn.Dropout(self.args.dropout)
        self.hidden_drop = torch.nn.Dropout(self.args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.args.dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.args.nfeats)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.num_nodes)))
        self.fc = torch.nn.Linear(16128,self.args.nfeats)

    def forward(self, x, edges):
        return self.cn(x, edges)

    def get_scores(self, ent, rel, ent_embed, batch_size, ignore=None):

        ent = ent.view(-1, 1, 15, 20)
        rel = rel.view(-1, 1, 15, 20)

        stacked_inputs = torch.cat([ent, rel], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_embed.transpose(1,0))
        x += self.b.expand_as(x)
        return x


    def get_embed(self, edges, node_id, r):

        np_embed = self.np_embeddings(node_id)
        if self.args.CN != 'Phi':
            np_embed = self.forward(np_embed, edges)


        r,r_len = seq_batch(r,self.args,self.rel2words)
        r_embed = self.phrase_embed_model(r,r_len)

        return r_embed, np_embed


    def get_loss(self, samples, labels, edges, node_id, ignore=None):

        np_embed = self.np_embeddings(node_id)
        if self.args.CN != 'Phi':
            np_embed = self.forward(np_embed, edges)

        sub_embed = np_embed[samples[:,0]]
        r = samples[:,1]


        r_batch,r_len = seq_batch(r.cpu().numpy(), self.args, self.rel2words)
        rel_embed = self.phrase_embed_model(r_batch,r_len)

        scores = self.get_scores(sub_embed, rel_embed, np_embed, self.args.batch_size)
        pred = F.sigmoid(scores)

        predict_loss = self.loss(pred, labels)

        return predict_loss