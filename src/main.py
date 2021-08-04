#### Import all the supporting classes
import os
import sys
import time
import uuid
import getpass
import random

import json
import pickle
import argparse
from configparser import ConfigParser
from tabulate import tabulate

from comet_ml import Experiment
import numpy as np
import torch
from torch.autograd import Variable

from data import load_data
from bertutils import BertUtils
from lama.options import __add_bert_args, __add_roberta_args
import utils
import models
import analysis
from comet_api import CometAPI

def add_comet(args):
    if args.nocomet:
        cm_train = utils.dummy_context_manager
        cm_test = utils.dummy_context_manager
        cm_valid = utils.dummy_context_manager
        experiment = None
    else:
        experiment = Experiment(project_name=args.project_name)
        experiment.add_tags([args.model_variant, args.dataset])
        experiment.log_parameters(vars(args))
        cm_train = experiment.train
        cm_test = experiment.test
        cm_valid = experiment.validate
    return experiment, cm_train, cm_valid, cm_test

def main(args):
    data = load_data(args)
    args.pad_id = data.word2id['<PAD>']
    args.num_nodes = len(data.ent2id)
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    logger = utils.get_logger(args.logs_path, args.logs_dir, args.config_dir)
    experiment, cm_train, cm_valid, cm_test = add_comet(args)
    if args.model_name in ['Care']:
        model = models.CaRE(args,data.embed_matrix,data.rel2word)
        evalFunc = utils.evaluate
    elif args.model_name in ['OKGIT']:
        model = models.OKGIT(args, data.embed_matrix, data.rel2word)
        evalFunc = utils.evaluate
    else:
        raise "Model not implemented"

    if args.use_cuda:
        model.cuda()

    bert = BertUtils(args)
    data.fetch_bert_embs(bert)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max',factor = 0.5, patience = 2)

    train_pairs = list(data.label_graph.keys())

    train_id = np.arange(len(train_pairs))

    node_id = torch.arange(0, args.num_nodes, dtype=torch.long)
    edges = torch.tensor(data.edges,dtype=torch.long)
    if args.use_cuda:
        edges = edges.cuda()
        node_id = node_id.cuda()

    best_MR = 20000
    best_MRR = 0
    best_epoch = 0
    count = 0
    valid_perf = None
    valid_ranks = None
    valid_ranked_cands = None
    for epoch in range(args.n_epochs):
        model.train()
        if count >= args.early_stop: break
        epoch_loss = 0
        permute = np.random.permutation(train_id)
        train_id = train_id[permute]
        n_batches = train_id.shape[0]//args.batch_size

        t1 = time.time()
        for i in range(n_batches):
            id_list = train_id[i*args.batch_size:(i+1)*args.batch_size]
            samples,labels = utils.get_next_batch(id_list, data, args, train_pairs)
            #TODO: move this to preprocessing
            # This part can be moved to preprocessing rather than batch-wise
            # heads = [data.id2ent[sample[0]] for sample in samples]
            # relations = [data.id2rel[sample[1]] for sample in samples]
            # bert_tail_embs = bert.getBertTails(heads, relations)
            bert_tail_embs = data.get_bert_embs(bert, samples)

            samples = Variable(torch.from_numpy(samples))
            labels = Variable(torch.from_numpy(labels).float())
            if args.use_cuda:
                samples = samples.cuda()
                labels = labels.cuda()
                bert_tail_embs = bert_tail_embs.cuda()

            optimizer.zero_grad()
            loss = model.get_loss(samples,labels,edges,node_id, bert_tail_embs)
            # loss = model.get_loss(samples,labels,edges,node_id)
            loss.backward()
            # print("batch {}/{} batches, batch_loss: {}".format(i,n_batches,(loss.data).cpu().numpy()),end='\r')
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            epoch_loss += (loss.data).cpu().numpy()
        # print("epoch {}/{} total epochs, epoch_loss: {}".format(epoch+1,args.n_epochs,epoch_loss/n_batches))
        logger.info("epoch {}/{} total epochs, epoch_loss: {}".format(epoch+1,args.n_epochs,epoch_loss/n_batches))

        if (epoch + 1)%args.eval_epoch==0:
            with cm_valid():
                model.eval()
                perf, ranks, ranked_cands = evalFunc(model, args.num_nodes, data.valid_trips, args, data, data.bert_tail_embs['valid'])
                # perf, ranks, ranked_cands = evaluateBert(model, args.num_nodes, data.valid_trips, args, data, data.bert_tail_embs['valid'])
                perf = utils.flatten(perf)
                perf['epoch'] = epoch+1
                if not args.nocomet:
                    experiment.log_metrics(perf)

                MR = perf['mr']
                MRR = perf['mrr']
                if MRR>best_MRR or MR<best_MR:
                    count = 0
                    if MRR>best_MRR: best_MRR = MRR
                    if MR<best_MR: best_MR = MR
                    valid_perf = perf
                    valid_ranks = ranks
                    valid_ranked_cands = ranked_cands
                    best_epoch = epoch + 1
                    # torch.save(model.state_dict(), model_state_file)
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, args.model_path)
                    # torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'data': data},model_state_file)
                    with open(args.outdata_path, 'wb') as fout:
                        pickle.dump(data, fout)
                else:
                    count+=1
                # print("Best Valid MRR: {}, Best Valid MR: {}, Best Epoch: {}".format(best_MRR,best_MR,best_epoch))
                logger.info("Best Valid MRR: {}, Best Valid MR: {}, Best Epoch: {}".format(best_MRR,best_MR,best_epoch))
                scheduler.step(best_epoch)


    # log the best valid perf to comet
    if not args.nocomet:
        with cm_valid():
            # valid_perf = utils.flatten(valid_perf, 'best')
            experiment.log_metrics(valid_perf)
    # save valid ranks and predictions
    save_ranks(data.valid_trips, valid_ranks, args.ranks_path % "valid")
    save_preds(data.valid_trips, valid_ranked_cands, data, args.preds_path_npy % "valid", args.preds_path_txt % "valid")
    ### Get Embeddings
    # print("Test Set Evaluation ---")
    logger.info("Test Set Evaluation ---")
    checkpoint = torch.load(args.model_path)
    with cm_test():
        model.eval()
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        # data = checkpoint['data']
        # with open(model_state_file+"_data.pkl", 'rb') as fin:
        # loaded_data = pickle.load(fin)
        test_perf, test_ranks, test_ranked_cands = evalFunc(model, args.num_nodes, data.test_trips, args, data, data.bert_tail_embs['test'])
        # test_perf, test_ranks, test_ranked_cands = evaluateBert(model, args.num_nodes, data.test_trips, args, data, data.bert_tail_embs['test'])
        test_perf = utils.flatten(test_perf)
        test_perf['epoch'] = best_epoch
        if not args.nocomet:
            experiment.log_metrics(test_perf)
    save_ranks(data.test_trips, test_ranks, args.ranks_path % "test")
    save_preds(data.test_trips, test_ranked_cands, data, args.preds_path_npy % "test", args.preds_path_txt % "test")
    with open(args.perfs_path, 'w') as fout:
        json.dump({'test':test_perf, 'valid':valid_perf}, fout)

def save_ranks(test_triples, ranks, rankfile):
    num_ranks = len(ranks['head']) + len(ranks['tail'])
    tr = np.zeros((num_ranks, 4), dtype=np.int32)
    # tr = np.zeros((len(ranks), 4), dtype=np.int32)
    tr[:,:3] = test_triples
    head_ranks = np.array(ranks['head'])
    tail_ranks = np.array(ranks['tail'])
    all_ranks = np.vstack([tail_ranks, head_ranks]).T.reshape((num_ranks,))
    tr[:,3] = all_ranks
    np.save(rankfile, tr)

def save_preds(test_triples, ranked_cands, data, predsnpy, predstxt):
    preds = np.zeros((test_triples.shape[0], test_triples.shape[1]+ranked_cands.shape[1]), dtype=np.int32)
    preds[:,:3] = test_triples
    preds[:,3:] = ranked_cands
    np.save(predsnpy, preds)

    delim = "\t"
    K = ranked_cands.shape[1]
    with open(predstxt, 'w') as fout:
        for i in range(test_triples.shape[0]):
            head = data.id2ent[test_triples[i,0]]
            rel = data.id2rel[test_triples[i,1]]
            tail = data.id2ent[test_triples[i,2]]
            topk = [head, rel, tail]
            for j in range(K):
                topk.append(data.id2ent[ranked_cands[i,j]])
            print(delim.join(topk), file=fout)


def analyse2(args):
    print("Analysing....")

    print("Loading model....")
    checkpoint = torch.load(args.model_path)
    with open(args.outdata_path, 'rb') as fin:
        data = pickle.load(fin)
        # data = load_data(args)
    args.pad_id = data.word2id['<PAD>']
    args.num_nodes = len(data.ent2id)
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    if args.model_name in ['Care']:
        model = models.CaRE(args, data.embed_matrix, data.rel2word)
    elif args.model_name in ['OKGIT']:
        model = models.OKGIT(args, data.embed_matrix, data.rel2word)
    else:
        raise "Model not implemented"

    if args.use_cuda:
        model.cuda()

    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    if args.eval_test:
        test_triples = data.test_trips
        bert_tail_embs = data.bert_tail_embs['test']
        sp = "test"
    else:
        test_triples = data.valid_trips
        bert_tail_embs = data.bert_tail_embs['valid']
        sp = "valid"
    print("Done!")

    print("Loading type embeddings...")
    # perf, ranks, ranked_cands = evalFunc(model, args.num_nodes, test_triples, args, data, bert_tail_embs)
    node_id = torch.arange(0, args.num_nodes, dtype=torch.long)
    edges = torch.tensor(data.edges,dtype=torch.long)
    only_tails = False
    if only_tails:
        all_tails = data.get_all_tails()
    else:
        all_tails = data.get_all_ents()
    all_tails = list(all_tails)
    tail_indices = {tail:idx for idx, tail in enumerate(all_tails)}
    all_tails = np.array(all_tails, dtype=np.long)
    if args.use_cuda:
        edges = edges.cuda()
        node_id = node_id.cuda()
        bert_tail_embs = bert_tail_embs.cuda()

    # get np_embed
    if args.model_name.lower() in ['okgit']:
        np_embed = model.conve.np_embeddings(node_id)
        np_embed = model.conve.forward(np_embed, edges)
    else:
        np_embed = model.np_embeddings(node_id)
        np_embed = model.forward(np_embed, edges)
    np_embed = np_embed.detach().cpu().numpy()

    bert_type_embs, care_type_embs = model.get_type_embed(bert_tail_embs, node_id, edges)
    bert_type_embs = bert_type_embs.cpu().detach().numpy()
    care_type_embs = care_type_embs.cpu().detach().numpy()
    # remove unused entity embeddings from Care
    # care_type_embs = care_type_embs[all_tails, :]
    if args.dataset in ['ReVerb20K_filtered']:
        outputdir = "emb/r20kf"
    elif args.dataset in ['ReVerb20K']:
        outputdir = "emb/r20k"
    elif args.dataset in ['ReVerb45K_filtered']:
        outputdir = "emb/r45kf"
    elif args.dataset in ['ReVerb45K']:
        outputdir = "emb/r45k"
    np.save(os.path.join(outputdir, 'np_embed.npy'), np_embed)
    np.save(os.path.join(outputdir, 'type_embed.npy'), care_type_embs)
    with open(os.path.join(outputdir, 'ent2id.json'), 'w') as fout:
        json.dump(data.ent2id, fout)
    with open(os.path.join(outputdir, 'id2ent.json'), 'w') as fout:
        json.dump(data.id2ent, fout)

def analyse(args):
    print("Analysing....")

    print("Loading model....")
    checkpoint = torch.load(args.model_path)
    with open(args.outdata_path, 'rb') as fin:
        data = pickle.load(fin)
        # data = load_data(args)
    args.pad_id = data.word2id['<PAD>']
    args.num_nodes = len(data.ent2id)
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    if args.model_name in ['Care']:
        model = models.CaRE(args, data.embed_matrix, data.rel2word)
    elif args.model_name in ['OKGIT']:
        model = models.OKGIT(args, data.embed_matrix, data.rel2word)
    else:
        raise "Model not implemented"

    if args.use_cuda:
        model.cuda()

    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    if args.eval_test:
        test_triples = data.test_trips
        bert_tail_embs = data.bert_tail_embs['test']
        sp = "test"
    else:
        test_triples = data.valid_trips
        bert_tail_embs = data.bert_tail_embs['valid']
        sp = "valid"
    print("Done!")

    print("Loading type embeddings...")
    # perf, ranks, ranked_cands = evalFunc(model, args.num_nodes, test_triples, args, data, bert_tail_embs)
    node_id = torch.arange(0, args.num_nodes, dtype=torch.long)
    edges = torch.tensor(data.edges,dtype=torch.long)
    only_tails = False
    if only_tails:
        all_tails = data.get_all_tails()
    else:
        all_tails = data.get_all_ents()
    all_tails = list(all_tails)
    tail_indices = {tail:idx for idx, tail in enumerate(all_tails)}
    all_tails = np.array(all_tails, dtype=np.long)
    if args.use_cuda:
        edges = edges.cuda()
        node_id = node_id.cuda()
        bert_tail_embs = bert_tail_embs.cuda()

    bert_type_embs, care_type_embs = model.get_type_embed(bert_tail_embs, node_id, edges)
    bert_type_embs = bert_type_embs.cpu().detach().numpy()
    care_type_embs = care_type_embs.cpu().detach().numpy()
    # remove unused entity embeddings from Care
    care_type_embs = care_type_embs[all_tails, :]
    print("Done!")

    print("Finding TSNE projections...")
    tsne_filename = os.path.join(args.plots_dir, "%s.tsne.png"%args.name)
    bert_tsne_proj, care_tsne_proj = analysis.findTSNEProjections(bert_type_embs, care_type_embs)
    print("Done!")

    print("Finding KNNs...")
    n_neighbors = 10
    n_examples = 50
    # neighbors = analysis.findkNN(bert_tsne_proj, care_tsne_proj, k=n_neighbors, normalize=True)
    if args.type_loss in ['mse']:
        neighbors = analysis.findkNN(bert_type_embs, care_type_embs, k=n_neighbors, normalize=False)
    else:
        neighbors = analysis.findkNN(bert_type_embs, care_type_embs, k=n_neighbors, normalize=True)
    print("Done!")

    print("Plotting TSNE and KNNs...")
    indices = np.random.randint(neighbors.shape[0], size=(n_examples,))
    output = []
    outdir = os.path.join(args.plots_dir, args.name)
    tsnedir = os.path.join(outdir, 'tsne')
    if not os.path.exists(tsnedir):
        os.makedirs(tsnedir)
    for ii in range(n_examples):
        i = indices[ii]
        head = data.id2ent[test_triples[i,0]]
        rel = data.id2rel[test_triples[i,1]]
        tail_id = test_triples[i,2]
        tail = data.id2ent[tail_id]
        vals = [str(i), head, rel, tail]
        names = {'query': '%s, %s' % (head, rel), 'tail': tail, 'dataset': args.dataset}
        knn_ids = []
        for j in range(n_neighbors):
            cur_idx = neighbors[i,j]
            name = data.id2ent[all_tails[cur_idx]]
            knn_ids.append(tail_indices[all_tails[cur_idx]])
            vals.append(name)
        names['values'] = vals[4:]
        output.append(vals)
        tsne_filename = os.path.join(tsnedir, "%s.%d.tsne.png"%(sp, i))
        analysis.plotTSNE(care_tsne_proj[tail_indices[tail_id],:], bert_tsne_proj[i,:], care_tsne_proj[knn_ids,:], care_tsne_proj, names, tsne_filename)
        # analysis.plotTSNE(care_tsne_proj[tail_id,:], bert_tsne_proj[i,:], care_tsne_proj[knn_ids,:], care_tsne_proj[all_tails,:], names, tsne_filename)
    print("Done!")
    print(tabulate(output))
    with open(os.path.join(outdir, 'knn.txt'), 'w') as fout:
        print(tabulate(output), file=fout)
    # save rank profiles
    # rank_file = 
    # analysis.generate_rank_profile()

def evaluation(args):
    checkpoint = torch.load(args.model_path)
    with open(args.outdata_path, 'rb') as fin:
        data = pickle.load(fin)
        # data = load_data(args)
    args.pad_id = data.word2id['<PAD>']
    args.num_nodes = len(data.ent2id)
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    if args.model_name in ['Care']:
        model = models.CaRE(args, data.embed_matrix, data.rel2word)
        evalFunc = utils.evaluate
    elif args.model_name in ['OKGIT']:
        model = models.OKGIT(args, data.embed_matrix, data.rel2word)
        evalFunc = utils.evaluate
    else:
        raise "Model not implemented"

    # model = OKGIT(args, data.embed_matrix, data.rel2word)

    if args.use_cuda:
        model.cuda()

    ### Get Embeddings
    print("Test Set Evaluation ---")
    model.eval()
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    # data = checkpoint['data']
    if args.eval_test:
        test_triples = data.test_trips
        bert_tail_embs = data.bert_tail_embs['test']
        sp = "test"
    else:
        test_triples = data.valid_trips
        bert_tail_embs = data.bert_tail_embs['valid']
        sp = "valid"
    perf, ranks, ranked_cands = evalFunc(model, args.num_nodes, test_triples, args, data, bert_tail_embs)
    # perf, ranks, ranked_cands = evaluateBert(model, args.num_nodes, test_triples, args, data, bert_tail_embs)
    save_ranks(test_triples, ranks, args.ranks_path % sp)
    save_preds(test_triples, ranked_cands, data, args.preds_path_npy % sp, args.preds_path_txt % sp)

def getParser():
    parser = argparse.ArgumentParser(description='CaRe: Canonicalization Infused Representations for Open KGs')

    ### Model and Dataset choice
    parser.add_argument('--CN',   dest='CN', default='LAN', choices=['LAN','GCN','GAT','Phi'], help='Choice of Canonical Cluster Encoder Network')
    parser.add_argument('--dataset',         dest='dataset',         default='ReVerb45K', help='Dataset Choice')
    parser.add_argument('--model_variant',   dest='model_variant',   default='CaRe',         help='model variant')
    parser.add_argument('--model_name',      dest='model_name',      default='OKGIT',     help='model name [Care/OKGIT]')

    ### Directories Config
    parser.add_argument('--data_cfg',        dest='data_cfg',        default='config/data.cfg',      help='config file containing directory information')
    parser.add_argument('--seed',      dest='seed',       default=42,     type=int,       help='seed')

    #### Hyper-parameters
    parser.add_argument('--nfeats',      dest='nfeats',       default=300,   type=int,       help='Embedding Dimensions')
    parser.add_argument('--nheads',      dest='nheads',       default=3,     type=int,       help='multi-head attantion in GAT')
    parser.add_argument('--num_layers',  dest='num_layers',   default=1,     type=int,       help='No. of layers in encoder network')
    parser.add_argument('--bidirectional',  dest='bidirectional',   default=True,     type=bool,       help='type of encoder network')
    parser.add_argument('--poolType',    dest='poolType',     default='last',choices=['last','max','mean'], help='pooling operation for encoder network')
    parser.add_argument('--dropout',     dest='dropout',      default=0.5,   type=float,     help='Dropout')
    parser.add_argument('--reg_param',   dest='reg_param',    default=0.0,   type=float,     help='regularization parameter')
    parser.add_argument('--lr',          dest='lr',           default=0.001, type=float,     help='learning rate')
    parser.add_argument('--p_norm',      dest='p_norm',       default=1,     type=int,       help='TransE scoring function')
    parser.add_argument('--batch_size',  dest='batch_size',   default=128,   type=int,       help='batch size for training')
    parser.add_argument('--neg_samples', dest='neg_samples',  default=10,    type=int,       help='No of Negative Samples for TransE')
    parser.add_argument('--n_epochs',    dest='n_epochs',     default=500,   type=int,       help='maximum no. of epochs')
    parser.add_argument('--grad_norm',   dest='grad_norm',    default=1.0,   type=float,     help='gradient clipping')
    parser.add_argument('--eval_epoch',  dest='eval_epoch',   default=5,     type=int,       help='Interval for evaluating on validation dataset')
    parser.add_argument('--Hits',        dest='Hits',         default= [1,3,10,30,50],           help='Choice of n in Hits@n')
    parser.add_argument('--early_stop',  dest='early_stop',   default=10,    type=int,       help='Stopping training after validation performance stops improving')

    parser.add_argument('--lamafile',    dest='lamafile',    type=str,      default="",     help='file containing lama model')
    parser.add_argument('--lamaweight',  dest='lamaweight',  type=float,    default=0.0,    help='weight for LAMA models score')
    parser.add_argument('--careweight',  dest='careweight',  type=float,    default=1.0,    help='weight for CaRE models score')
    parser.add_argument('--reverse',     dest='reverse',     default=False,  action='store_true', help='whether to add inverse relation edges')
    parser.add_argument('--eval-test',   dest='eval_test',   default=False,  action='store_true', help='flag to evaluate on test split instead of valid split')

    # gpu options
    parser.add_argument('--gpu',        dest='gpu',   default=0,     type=int,       help='gpu to run the code')

    # logging options
    parser.add_argument("--name",           type=str,   default='testrun_'+str(uuid.uuid4())[:8], help="Set filename for saving or restoring models")

    # analysis options
    parser.add_argument("--analysis",   default=False,  action='store_true', help="use this flag to analyse existing models")

    # comet options
    parser.add_argument("--nocomet",        action="store_true",    default=False,  help="flag for avoiding comet logging")
    parser.add_argument("--project-name",   type=str,   default="<projectname>",   help="project name in comet.ml")

    # Bert options
    parser.add_argument("--language-models", "--lm", dest="models", help="comma separated list of language models", required=True,)
    parser.add_argument('--type-dim', dest='type_dim', default=300, type=int, help='Dimensions for the type vectors')
    parser.add_argument('--type-weight', dest='type_weight', default=0.0, type=float, help='weight for type loss')
    parser.add_argument('--type-loss', dest='type_loss', default='bce', type=str, help='loss function to use for type constraints')
    parser.add_argument('--type_composition_weight', dest='type_composition_weight', default=1.0, type=float, help='type score composition weight in case of for additive composition')
    parser.add_argument('--type_composition', dest='type_composition', default="mul", type=str, help='type score composition [add/mul]')
    parser.add_argument('--type_transform', dest='type_transform', default="sigmoid", type=str, help='type score transformation [identity/sigmoid/inverse]')
    parser.add_argument( "--max-sentence-length", dest="max_sentence_length", type=int, default=100, help="max sentence lenght",)
    __add_bert_args(parser)
    __add_roberta_args(parser)
    return parser

def set_params(args):
    cfg = ConfigParser()
    cfg.read(args.data_cfg)
    username = getpass.getuser()

    # set directories
    args.data_path = cfg.get(username, 'datadir')
    args.config_dir = cfg.get(username, 'configdir')
    args.results_dir = cfg.get(username, 'resultsdir')
    args.logs_dir    = cfg.get(username, 'logsdir')
    args.model_dir  = cfg.get(username, 'modeldir')
    args.preds_dir  = cfg.get(username, 'predsdir')
    args.ranks_dir  = cfg.get(username, 'ranksdir')
    args.perfs_dir  = cfg.get(username, 'perfsdir')
    args.plots_dir  = cfg.get(username, 'plotsdir')

    # set data files
    args.data_files = {
        'ent2id_path'       : args.data_path + '/' + args.dataset + '/ent2id.txt',
        'rel2id_path'       : args.data_path + '/' + args.dataset + '/rel2id.txt',
        'train_trip_path'   : args.data_path + '/' + args.dataset + '/train_trip.txt',
        'test_trip_path'    : args.data_path + '/' + args.dataset + '/test_trip.txt',
        'valid_trip_path'   : args.data_path + '/' + args.dataset + '/valid_trip.txt',
        'gold_npclust_path' : args.data_path + '/' + args.dataset + '/gold_npclust.txt',
        'cesi_npclust_path' : args.data_path + '/' + args.dataset + '/cesi_npclust.txt',
        'glove_path'        : args.data_path + '/' + 'glove/glove.6B.300d.txt'
    }


    # set output files
    if not args.analysis and args.n_epochs > 0:
        args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    args.model_path = os.path.join(args.model_dir, '%s.model.pth' % args.name)
    args.outdata_path = os.path.join(args.model_dir, '%s.data.pkl' % args.name)
    args.logs_path = '%s.log' % args.name
    args.preds_path_txt = os.path.join(args.preds_dir, f'{args.name}.%s.txt')
    args.preds_path_npy = os.path.join(args.preds_dir, f'{args.name}.%s.npy')
    args.ranks_path = os.path.join(args.ranks_dir, f'{args.name}.%s.npy')
    args.perfs_path = os.path.join(args.perfs_dir, '%s.json' % args.name)
    # args.model_path = "ConvE" + "-" + args.CN + "-" + args.dataset + "_modelpath.pth"

    # set bert dimensions
    if args.models in ['bert']:
        if args.bert_model_name.startswith('bert-base'):
            args.bert_dim = 768
        else:
            args.bert_dim = 1024
    elif args.models in ['roberta']:
        if args.roberta_model_dir is None:
            print("please specify --rmd (roberta model directory)")
        elif args.roberta_model_dir.endswith('base'):
            args.bert_dim = 768
        else:
            args.bert_dim = 1024


    return args

def copy_params(args):
    print("Getting parameters from Comet...")
    comet_api = CometAPI()
    params = comet_api.get_params(args.name)
    ignore = ['gpu', 'analysis', 'nocomet', 'n_epochs']
    for key, val in params.items():
        if key in ignore:
            continue
        else:
            args.__dict__[key] = val
    args.nocomet = True
    args.n_epochs = 0
    print("Done!")
    return args

def validate_params(args):
    if args.type_composition in ['mul'] and args.type_transform in ['identity']:
        print("Invalid parameter combination, can't use 'mul' composition with 'identity' transform")
        sys.exit(1)

if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    if args.analysis:
        args = copy_params(args)
    else:
        args = set_params(args)
    validate_params(args)
    seed = args.seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # print(np.random.randint(100))
    # sys.exit(1)

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)
    if args.analysis:
        # Run Analysis
        analyse2(args)
        # analyse(args)
    elif args.n_epochs > 0:
        # Run training
        main(args)
    else:
        # Run Evaluation
        evaluation(args)

