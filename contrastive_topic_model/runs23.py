import os
import copy
import argparse
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtools.optim import RangerLars
import itertools
from word_embedding_utils import *
from collections import OrderedDict
import pickle

from evaluation import evaluate_classification, evaluate_clustering


import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import CountVectorizer
from utils.miscellaneous import AverageMeter
from collections import OrderedDict

import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
from pytorch_transformers import *

from tqdm import tqdm
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from datetime import datetime
from scipy.linalg import qr
from data import *
from model import ContBertTopicExtractorAE
from evaluation import get_topic_qualities
import warnings
warnings.filterwarnings("ignore")
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from utils.config import _parse_args, save_config
import wandb
from utils import miscellaneous, log

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 

class Stage2Dataset(Dataset):
    def __init__(self, encoder, ds, basesim_matrix, word_candidates, k=1, lemmatize=True):
        self.lemmatize = lemmatize
        self.ds = ds
        self.org_list = self.ds.org_list
        self.nonempty_text = self.ds.nonempty_text
        english_stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords_list = set(english_stopwords)
        self.vectorizer = CountVectorizer(vocabulary=word_candidates)
        self.vectorizer.fit(self.preprocess_ctm(self.nonempty_text)) 
        self.bow_list = []
        for sent in tqdm(self.nonempty_text):
            self.bow_list.append(self.vectorize(sent))
            
        sim_weight, sim_indices = basesim_matrix.topk(k=k, dim=-1)
        zip_iterator = zip(np.arange(len(sim_weight)), sim_indices.squeeze().data.numpy())
        self.pos_dict = dict(zip_iterator)
        
        self.embedding_list = []
        encoder_device = next(encoder.parameters()).device
        for org_input in tqdm(self.org_list):
            org_input_ids = org_input['input_ids'].to(encoder_device).reshape(1, -1)
            org_attention_mask = org_input['attention_mask'].to(encoder_device).reshape(1, -1)
            embedding = encoder(input_ids = org_input_ids, attention_mask = org_attention_mask)
            self.embedding_list.append(embedding['pooler_output'].squeeze().detach().cpu())
            
    
    def __len__(self):
        return len(self.org_list)
        
    def preprocess_ctm(self, documents):
        preprocessed_docs_tmp = documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords_list])
                                 for doc in preprocessed_docs_tmp]
        if self.lemmatize:
            lemmatizer = WordNetLemmatizer()
            preprocessed_docs_tmp = [' '.join([lemmatizer.lemmatize(w) for w in doc.split()])
                                     for doc in preprocessed_docs_tmp]
        return preprocessed_docs_tmp
        
    def vectorize(self, text):
        text = self.preprocess_ctm([text])
        vectorized_input = self.vectorizer.transform(text)
        vectorized_input = vectorized_input.toarray().astype(np.float64)
#         vectorized_input = (vectorized_input != 0).astype(np.float64)

        # Get word distribution from BoW
        if vectorized_input.sum() == 0:
            vectorized_input += 1e-8
        vectorized_input = vectorized_input / vectorized_input.sum(axis=1, keepdims=True)
        assert abs(vectorized_input.sum() - vectorized_input.shape[0]) < 0.01
        
        vectorized_label = torch.tensor(vectorized_input, dtype=torch.float)
        return vectorized_label[0]
        
        
    def __getitem__(self, idx):
        pos_idx = self.pos_dict[idx]
        return self.embedding_list[idx], self.embedding_list[pos_idx], self.bow_list[idx], self.bow_list[pos_idx]

class Stage2TestDataset(Dataset):
    def __init__(self, encoder, ds, word_candidates, k=1, lemmatize=False):
        self.lemmatize = lemmatize
        self.ds = ds
        self.org_list = self.ds.org_list
        self.nonempty_text = self.ds.nonempty_text
        english_stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords_list = set(english_stopwords)
        self.vectorizer = CountVectorizer(vocabulary=word_candidates)
        self.vectorizer.fit(self.preprocess_ctm(self.nonempty_text)) 
        self.bow_list = []
        for sent in tqdm(self.nonempty_text):
            self.bow_list.append(self.vectorize(sent))
        
        self.embedding_list = []
        encoder_device = next(encoder.parameters()).device
        for org_input in tqdm(self.org_list):
            org_input_ids = org_input['input_ids'].to(encoder_device).reshape(1, -1)
            org_attention_mask = org_input['attention_mask'].to(encoder_device).reshape(1, -1)
            embedding = encoder(input_ids = org_input_ids, attention_mask = org_attention_mask)
            self.embedding_list.append(embedding['pooler_output'].squeeze().detach().cpu())
            
    
    def __len__(self):
        return len(self.org_list)
        
    def preprocess_ctm(self, documents):
        preprocessed_docs_tmp = documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords_list])
                                 for doc in preprocessed_docs_tmp]
        if self.lemmatize:
            lemmatizer = WordNetLemmatizer()
            preprocessed_docs_tmp = [' '.join([lemmatizer.lemmatize(w) for w in doc.split()])
                                     for doc in preprocessed_docs_tmp]
        return preprocessed_docs_tmp
        
    def vectorize(self, text):
        text = self.preprocess_ctm([text])
        vectorized_input = self.vectorizer.transform(text)
        vectorized_input = vectorized_input.toarray().astype(np.float64)
#         vectorized_input = (vectorized_input != 0).astype(np.float64)

        # Get word distribution from BoW
        if vectorized_input.sum() == 0:
            vectorized_input += 1e-8
        vectorized_input = vectorized_input / vectorized_input.sum(axis=1, keepdims=True)
        assert abs(vectorized_input.sum() - vectorized_input.shape[0]) < 0.01
        
        vectorized_label = torch.tensor(vectorized_input, dtype=torch.float)
        return vectorized_label[0]
        
        
    def __getitem__(self, idx):
        return self.embedding_list[idx], self.bow_list[idx]

def dist_match_loss(hiddens, alpha=1.0):
    device = hiddens.device
    hidden_dim = hiddens.shape[-1]
    H = np.random.randn(hidden_dim, hidden_dim)
    Q, R = qr(H) 
    rand_w = torch.Tensor(Q).to(device)
    loss_dist_match = get_swd_loss(hiddens, rand_w, alpha)
    return loss_dist_match

def get_swd_loss(states, rand_w, alpha=1.0):
    device = states.device
    states_shape = states.shape
    states = torch.matmul(states, rand_w)
    states_t, _ = torch.sort(states.t(), dim=1)

    # Random vector with length from normal distribution
    states_prior = torch.Tensor(np.random.dirichlet([alpha]*states_shape[1], states_shape[0])).to(device) # (bsz, dim)
    states_prior = torch.matmul(states_prior, rand_w) # (dim, dim)
    states_prior_t, _ = torch.sort(states_prior.t(), dim=1) # (dim, bsz)
    return torch.mean(torch.sum((states_prior_t - states_t)**2, axis=0))

def data_load(dataset_name):
    should_measure_hungarian = False
    if dataset_name == 'news':
        textData = newsData()
        should_measure_hungarian = True
    elif dataset_name == 'imdb':
        textData = IMDBData()
    elif dataset_name == 'agnews':
        textData = AGNewsData()
    elif dataset_name == 'yahoo':
        textData = YahooData()
    elif dataset_name == 'twitter':
        textData = twitterData('/home/data/topicmodel/twitter_covid19.tsv')
    elif dataset_name == 'wiki':
        textData = wikiData('/home/data/topicmodel/smplAbstracts/')
    elif dataset_name == 'nips':
        textData = nipsData('/home/data/topicmodel/papers.csv')
    elif dataset_name == 'stackoverflow':
        textData = stackoverflowData('/home/data/topicmodel/stack_overflow.csv')
    elif dataset_name == 'reuters':
        textData = reutersData('/home/data/topicmodel/reuters-21578.txt')
    elif dataset_name == 'r52':
        textData = r52Data('/home/data/topicmodel/r52/')
        should_measure_hungarian = True
    return textData, should_measure_hungarian

def remove_dup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

RESULT_DIR = 'results'
DATA_DIR = 'data'


if __name__ == "__main__":
    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)
    
    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))

    parser = _parse_args()
    args = parser.parse_args()
    save_config(args, os.path.join(current_run_dir, 'config.txt'))
    wandb.init(project="utopic-stage23", config=args)

    bsz = args.bsz
    epochs_1 = args.epochs_1
    epochs_2 = args.epochs_2

    n_cluster = args.n_cluster
    n_topic = args.n_topic if (args.n_topic is not None) else n_cluster
    args.n_topic = n_topic

    textData, should_measure_hungarian = data_load(args.dataset)

    ema_alpha = 0.99
    n_word = args.n_word
    if args.dirichlet_alpha_1 is None:
        dirichlet_alpha_1 = 1 / n_cluster
    else:
        dirichlet_alpha_1 = args.dirichlet_alpha_1
    if args.dirichlet_alpha_2 is None:
        dirichlet_alpha_2 = dirichlet_alpha_1
    else:
        dirichlet_alpha_2 = args.dirichlet_alpha_2
        
    bert_name = args.base_model
    bert_name_short = bert_name.split('/')[-1]
    gpu_ids = args.gpus

    # skip_stage_1 = (args.stage_1_ckpt is not None)

    # load stage 1 saved
    model_stage1_name = f'./results/stage_1/{args.dataset}_model_{bert_name_short}_stage1_{args.n_topic}t_bsz{args.bsz}_{args.n_word}w_{args.coeff_1_dist}s1dist_{args.epochs_1}e'
    miscellaneous.create_folder_if_not_exist(model_stage1_name)

    trainds = BertDataset(bert=bert_name, text_list=textData.data, N_word=n_word, vectorizer=None, lemmatize=True)

    word_candidates = []
    with open(os.path.join(model_stage1_name, 'word_candidates'), 'r') as file:
        word_candidates = file.read().splitlines()
    print(word_candidates)

    new_state_dict = OrderedDict()
    for k, v in torch.load(os.path.join(model_stage1_name, 'checkpoint.ckpt')).items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    model = ContBertTopicExtractorAE(N_topic=n_cluster, N_word=n_word, bert=bert_name, bert_dim=384)
    model.cuda(gpu_ids[0])
    model.load_state_dict(new_state_dict, strict=False)

    basesim_path = os.path.join(model_stage1_name, f'{args.dataset}_{bert_name_short}_basesim_matrix_full.pkl')
    basesim_matrix = torch.load(basesim_path)
    finetuneds = Stage2Dataset(model.encoder, trainds, basesim_matrix, word_candidates, lemmatize=True)    
    testds = BertDataset(bert=bert_name, text_list=textData.test_data, N_word=n_word, vectorizer=None, lemmatize=True)
    testds2 = Stage2TestDataset(model.encoder, testds, word_candidates, lemmatize=True)

    vocab_dict = finetuneds.vectorizer.vocabulary_
    vocab_dict_reverse = {i:v for v, i in vocab_dict.items()}
    print(n_word)

    total_score_cat = np.load(os.path.join(model_stage1_name, 'total_score_cat.npy'))
    print('total score cat shape: ', total_score_cat.shape)
    with open(os.path.join(model_stage1_name, 'words_to_idx.pkl'), 'rb') as file:
        words_to_idx = pickle.load(file)


    weight_candidates = {}
    for candidate in word_candidates:
        weight_candidates[candidate] = [total_score_cat[label, words_to_idx[candidate]] for label in range(n_cluster)]

    weight_cand_to_idx = {k: v for v, k in enumerate(list(weight_candidates.keys()))}
    weight_cand_matrix = np.array(list(weight_candidates.values()))

    # weight_cand_matrix = np.load(os.path.join(model_stage1_name, 'weight_cand_matrix.npy'))
    weight_cands = torch.tensor(weight_cand_matrix.max(axis=1)).cuda(gpu_ids[0]).float()

    results_list = []

    # run stage 2
    for i in range(args.stage_2_repeat):
        model = ContBertTopicExtractorAE(N_topic=n_topic, N_word=args.n_word, bert=bert_name, bert_dim=384)
        new_state_dict = OrderedDict()
        for k, v in torch.load(os.path.join(model_stage1_name, 'checkpoint.ckpt')).items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.beta = nn.Parameter(torch.Tensor(model.N_topic, n_word))
        nn.init.xavier_uniform_(model.beta)
        model.beta_batchnorm = nn.Sequential()
        model.cuda(gpu_ids[0])

        losses = AverageMeter()
        dlosses = AverageMeter() 
        rlosses = AverageMeter()
        closses = AverageMeter()
        distlosses = AverageMeter()
        trainloader = DataLoader(finetuneds, batch_size=bsz, shuffle=False, num_workers=0)
        testloader = DataLoader(testds2, batch_size=bsz, shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.stage_2_lr)

        memory_queue = F.softmax(torch.randn(512, n_topic).cuda(gpu_ids[0]), dim=1)
        print("Coeff   / regul: {:.5f} - recon: {:.5f} - c: {:.5f} - dist: {:.5f} ".format(args.coeff_2_regul, 
                                                                                            args.coeff_2_recon,
                                                                                            args.coeff_2_cons,
                                                                                            args.coeff_2_dist))
        logger.info("Coeff   / regul: {:.5f} - recon: {:.5f} - c: {:.5f} - dist: {:.5f} ".format(args.coeff_2_regul,
                                                                                                 args.coeff_2_recon,
                                                                                                 args.coeff_2_cons,
                                                                                                 args.coeff_2_dist))
        for epoch in range(args.epochs_2):
            model.train()
            model.encoder.eval()
            for batch_idx, batch in enumerate(trainloader):
                org_input, pos_input, org_bow, pos_bow = batch
                org_input = org_input.cuda(gpu_ids[0])
                org_bow = org_bow.cuda(gpu_ids[0])
                pos_input = pos_input.cuda(gpu_ids[0])
                pos_bow = pos_bow.cuda(gpu_ids[0])

                batch_size = org_input.size(0)

                org_dists, org_topic_logit = model.decode(org_input)
                pos_dists, pos_topic_logit = model.decode(pos_input)

                org_topic = F.softmax(org_topic_logit, dim=1)
                pos_topic = F.softmax(pos_topic_logit, dim=1)

                recons_loss = torch.mean(-torch.sum(torch.log(org_dists + 1E-10) * (org_bow * weight_cands), axis=1), axis=0)
                recons_loss += torch.mean(-torch.sum(torch.log((1-org_dists) + 1E-10) * ((1-org_bow) * weight_cands), axis=1), axis=0)
                recons_loss += torch.mean(-torch.sum(torch.log(pos_dists + 1E-10) * (pos_bow * weight_cands), axis=1), axis=0)
                recons_loss += torch.mean(-torch.sum(torch.log((1-pos_dists) + 1E-10) * ((1-pos_bow) * weight_cands), axis=1), axis=0)
                recons_loss *= 0.5

                # consistency loss
                pos_sim = torch.sum(org_topic * pos_topic, dim=-1)
                cons_loss = -pos_sim.mean()

                # distribution loss
                # batchmean
                distmatch_loss = dist_match_loss(torch.cat((org_topic, pos_topic), dim=0), dirichlet_alpha_2)


                loss = args.coeff_2_recon * recons_loss + \
                    args.coeff_2_cons * cons_loss + \
                    args.coeff_2_dist * distmatch_loss 

                losses.update(loss.item(), bsz)
                closses.update(cons_loss.item(), bsz)
                rlosses.update(recons_loss.item(), bsz)
                distlosses.update(distmatch_loss.item(), bsz)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch-{} / recon: {:.5f} - dist: {:.5f} - cons: {:.5f}".format(epoch, rlosses.avg, distlosses.avg, closses.avg))
            logger.info("Epoch-{} / recon: {:.5f} - dist: {:.5f} - cons: {:.5f}".format(epoch, rlosses.avg, distlosses.avg, closses.avg))
            wandb.log({"recon_loss": rlosses.avg, "dist_loss": distlosses.avg, "cons_loss": closses.avg})

        print("------- Evaluation results -------")
        logger.info("------- Evaluation results -------")
        all_list = {}
        for e, i in enumerate(model.beta.cpu().topk(15, dim=1).indices):
            word_list = []
            for j in i:
                word_list.append(vocab_dict_reverse[j.item()])
            all_list[e] = word_list
            print("topic-{}".format(e), word_list)
            logger.info("topic-{}".format(e))
            logger.info(word_list)

        topic_words_list = list(all_list.values())
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        results = get_topic_qualities(topic_words_list, palmetto_dir=args.palmetto_dir,
                                    reference_corpus=[doc.split() for doc in trainds.preprocess_ctm(trainds.nonempty_text)],
                                    filename=os.path.join(current_run_dir, f'{str(now)}.txt'))
        train_theta = []
        test_theta = []
        for batch_idx, batch in tqdm(enumerate(trainloader)):
            org_input, _, org_bow, _ = batch
            org_input = org_input.cuda(gpu_ids[0])
            org_bow = org_bow.cuda(gpu_ids[0])
            batch_size = org_input.size(0)
            org_dists, org_topic_logit = model.decode(org_input)
            org_topic = F.softmax(org_topic_logit, dim=1)

            train_theta.append(org_topic.detach().cpu())

        train_theta = np.concatenate(train_theta, axis=0)

        for batch_idx, batch in tqdm(enumerate(testloader)): 
            org_input, org_bow = batch
            batch_size = org_input.size(0)
            org_input = org_input.cuda(gpu_ids[0])
            org_bow = org_bow.cuda(gpu_ids[0])
            org_dists, org_topic_logit = model.decode(org_input)
            org_topic = F.softmax(org_topic_logit, dim=1)

            test_theta.append(org_topic.detach().cpu())

        test_theta = np.concatenate(test_theta, axis=0)

        classification_res = evaluate_classification(train_theta, test_theta, textData.target_filtered, textData.test_targets, tune=True, logger=logger)
        clustering_res = evaluate_clustering(test_theta, textData.test_targets)

        results.update(classification_res)
        results.update(clustering_res)


        if should_measure_hungarian:
            topic_dist = torch.empty((0, n_topic))
            model.eval()
            evalloader = DataLoader(finetuneds, batch_size=bsz, shuffle=False, num_workers=0)
            non_empty_text_index = [i for i, text in enumerate(textData.data) if len(text) != 0]
            assert len(finetuneds) == len(non_empty_text_index)
            with torch.no_grad():
                for batch in tqdm(evalloader):
                    org_input, _, org_bow, __ = batch
                    org_input = org_input.cuda(gpu_ids[0])
                    org_dists, org_topic_logit = model.decode(org_input)
                    org_topic = F.softmax(org_topic_logit, dim=1)
                    topic_dist = torch.cat((topic_dist, org_topic.detach().cpu()), 0)

        print(results)
        logger.info(results)
        wandb.log(results)
        print()
        results_list.append(results)

    results_df = pd.DataFrame(results_list)
    print(results_df)
    print(results_df['CV_wiki'])
    print(results_df['diversity'])
    print(results_df['irbo'])
    print(results_df['acc'])
    print(results_df['macro-F1'])
    print(results_df['Purity'])
    print(results_df['NMI'])
    
    table_df = wandb.Table(dataframe=results_df)
    wandb.log({'results_df': table_df})
    
    print('mean')
    print(results_df.mean())
    print('std')
    print(results_df.std())


    logger.info(results_df)
    logger.info('mean')
    logger.info(results_df.mean())
    logger.info('std')
    logger.info(results_df.std())

    wandb.log({'mean_acc': results_df.mean()['acc']})
    wandb.log({'mean_f1': results_df.mean()['macro-F1']})
    wandb.log({'mean_purity': results_df.mean()['Purity']})
    wandb.log({'mean_NMI': results_df.mean()['NMI']})
    wandb.log({'mean_CV': results_df.mean()['CV_wiki']})
    wandb.log({'mean_diversity': results_df.mean()['diversity']})
    wandb.log({'mean_NPMI': results_df.mean()['npmi_wiki']})
    wandb.log({'mean_cp': results_df.mean()['cp_wiki']})
    wandb.log({'mean_sim': results_df.mean()['sim_w2v']})
    wandb.log({'mean_irbo': results_df.mean()['irbo']})

    wandb.log({'std_acc': results_df.std()['acc']})
    wandb.log({'std_f1': results_df.std()['macro-F1']})
    wandb.log({'std_purity': results_df.std()['Purity']})
    wandb.log({'std_NMI': results_df.std()['NMI']})
    wandb.log({'std_CV': results_df.std()['CV_wiki']})
    wandb.log({'std_diversity': results_df.std()['diversity']})
    wandb.log({'std_NPMI': results_df.std()['npmi_wiki']})
    wandb.log({'std_cp': results_df.std()['cp_wiki']})
    wandb.log({'std_sim': results_df.std()['sim_w2v']})
    wandb.log({'std_irbo': results_df.std()['irbo']})

    if args.result_file is not None:
        result_filename = f'{current_run_dir}/{args.result_file}'
    else:
        result_filename = f'{current_run_dir}/{now}.tsv'

    results_df.to_csv(result_filename, sep='\t')
