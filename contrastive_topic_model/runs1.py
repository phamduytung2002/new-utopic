import os
import copy
import math
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtools.optim import RangerLars
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from gensim.utils import deaccent
from nltk.corpus import stopwords as stop_words

from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

import numpy as np
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from utils.miscellaneous import AverageMeter
from collections import OrderedDict

from sklearn.feature_extraction.text import CountVectorizer
from pytorch_transformers import *
from tqdm import tqdm
import scipy.sparse as sp
from data import *
from word_embedding_utils import *
from model import ContBertTopicExtractorAE
import warnings
import wandb
from scipy.linalg import qr

from utils import miscellaneous
from utils.config import _parse_args, save_config
from utils import log
import pickle, json

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

class WhiteSpacePreprocessing():
    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000):
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language))
        self.vocabulary_size = vocabulary_size

        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("WhiteSpacePreprocessing is deprecated and will be removed in future versions."
                      "Use WhiteSpacePreprocessingStopwords.")

    def preprocess(self):
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [deaccent(doc.lower()) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        temp_vocabulary = set(vectorizer.get_feature_names())

        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in temp_vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs, retained_indices = [], [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])
                retained_indices.append(i)

        vocabulary = list(set([item for doc in preprocessed_docs for item in doc.split()]))

        return preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices

class TopicModelDataPreparationNoNumber(TopicModelDataPreparation):
    def fit(self, text_for_contextual, text_for_bow, labels=None, wordlist=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model
        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param labels: list of labels associated with each document (optional).
        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z]{2,}\b', vocabulary=wordlist)

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(train_contextualized_embeddings, train_bow_embeddings, self.id2token, encoded_labels)

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

def remove_dup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

RESULT_DIR = 'results'
DATA_DIR = 'data'


if __name__=="__main__":
    # set up logger and args parser
    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)
    parser = _parse_args()
    args = parser.parse_args()
    wandb.init(project="utopic_stage1", config=args)
    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    save_config(args, os.path.join(current_run_dir, 'config.txt'))
    miscellaneous.seedEverything(args.seed)


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
    
    model_stage1_name = f'./results/stage_1/{args.dataset}_model_{bert_name_short}_stage1_{args.n_topic}t_bsz{args.bsz}_{args.n_word}w_{args.coeff_1_dist}s1dist_{args.epochs_1}e'
    if not os.path.exists(model_stage1_name):
        os.makedirs(model_stage1_name)
        print("Folder created:", model_stage1_name)
    else:
        print("Folder already exists:", model_stage1_name)
        exit(0)

    # miscellaneous.create_folder_if_not_exist(model_stage1_name)


    # data preparation
    trainds = BertDataset(bert=bert_name, text_list=textData.data, N_word=n_word, vectorizer=None, lemmatize=True)
    basesim_path = os.path.join(model_stage1_name, f'{args.dataset}_{bert_name_short}_basesim_matrix_full.pkl')
    if os.path.isfile(basesim_path) == False:
        model = SentenceTransformer(bert_name.split('/')[-1], device='cuda')
        base_result_list = []
        for text in tqdm(trainds.nonempty_text):
            base_result_list.append(model.encode(text))
            
        base_result_embedding = np.stack(base_result_list)
        basereduced_norm = F.normalize(torch.tensor(base_result_embedding), dim=-1)
        basesim_matrix = torch.mm(basereduced_norm, basereduced_norm.t())
        ind = np.diag_indices(basesim_matrix.shape[0])
        basesim_matrix[ind[0], ind[1]] = torch.ones(basesim_matrix.shape[0]) * -1
        torch.save(basesim_matrix, basesim_path)
    else:
        basesim_matrix = torch.load(basesim_path)

    model = ContBertTopicExtractorAE(N_topic=n_cluster, N_word=n_word, bert=bert_name, bert_dim=384)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda(gpu_ids[0])


    # training stage 1
    losses = AverageMeter()
    closses = AverageMeter() 
    dlosses = AverageMeter() 
    rlosses = AverageMeter() 
    criterion = nn.CrossEntropyLoss()

    temp_basesim_matrix = copy.deepcopy(basesim_matrix)
    finetuneds = FinetuneDataset(trainds, temp_basesim_matrix, ratio=1, k=1)
    trainloader = DataLoader(finetuneds, batch_size=bsz, shuffle=False, num_workers=0)
    memoryloader = DataLoader(finetuneds, batch_size=bsz * 2, shuffle=False, num_workers=0)

    optimizer = RangerLars(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    global_step = 0
    memory_queue = F.softmax(torch.randn(512, n_cluster).cuda(gpu_ids[0]), dim=1)
    for epoch in range(epochs_1):
        model.train()
        try:
            model.module.encoder.eval()
        except:
            model.encoder.eval()
        #ema_model.train()
        tbar = tqdm(trainloader)
        for batch_idx, batch in enumerate(tbar):       
            org_input, pos_input, _, _ = batch
            org_input_ids = org_input['input_ids'].cuda(gpu_ids[0])
            org_attention_mask = org_input['attention_mask'].cuda(gpu_ids[0])
            pos_input_ids = pos_input['input_ids'].cuda(gpu_ids[0])
            pos_attention_mask = pos_input['attention_mask'].cuda(gpu_ids[0])
            batch_size = org_input_ids.size(0)

            all_input_ids = torch.cat((org_input_ids, pos_input_ids), dim=0)
            all_attention_masks = torch.cat((org_attention_mask, pos_attention_mask), dim=0)
            all_topics, _ = model(all_input_ids, all_attention_masks, return_topic=True)

            orig_topic, pos_topic = torch.split(all_topics, len(all_topics) // 2)
            pos_sim = torch.sum(orig_topic * pos_topic, dim=-1)

            # consistency loss
            consistency_loss = -pos_sim.mean()

            # distribution matching loss
            memory_queue = torch.cat((memory_queue.detach(), all_topics), dim=0)[all_topics.size(0):]
            distmatch_loss = dist_match_loss(memory_queue, dirichlet_alpha_1)
            loss = args.coeff_1_sim * consistency_loss + \
                args.coeff_1_dist * distmatch_loss

            losses.update(loss.item(), bsz)
            closses.update(consistency_loss.item(), bsz)
            dlosses.update(distmatch_loss.item(), bsz)

            tbar.set_description("Epoch-{} / consistency: {:.5f} - dist: {:.5f}".format(epoch, 
                                                                                        closses.avg, 
                                                                                        dlosses.avg), refresh=True)
            tbar.refresh()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": losses.avg, "consistency_loss": closses.avg, "distmatch_loss": dlosses.avg})
            # global_step += 1
            # if global_step==10:
            #     break
        scheduler.step()
        logger.info("Epoch-{} / consistency: {:.5f} - dist: {:.5f}".format(epoch, closses.avg, dlosses.avg))

    torch.save(model.state_dict(), os.path.join(model_stage1_name, 'checkpoint.ckpt'))
    wandb.log({"stage_1_model": model_stage1_name})



    # new_state_dict = OrderedDict()
    # for k, v in torch.load(os.path.join(model_stage1_name, 'checkpoint.ckpt')).items():
    #     if k.startswith("module."):
    #         new_state_dict[k[7:]] = v  # Remove 'module.' prefix
    #     else:
    #         new_state_dict[k] = v
    # model = ContBertTopicExtractorAE(N_topic=n_cluster, N_word=n_word, bert=bert_name, bert_dim=384)
    # model.cuda(gpu_ids[0])
    # model.load_state_dict(new_state_dict, strict=True)



    # done training stage 1
    model.eval()
    result_list = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(memoryloader)):        
            org_input, _, _, _ = batch
            org_input_ids = org_input['input_ids'].cuda(gpu_ids[0])
            org_attention_mask = org_input['attention_mask'].cuda(gpu_ids[0])
            topic, _ = model(org_input_ids, org_attention_mask, return_topic = True)
            result_list.append(topic)
    result_embedding = torch.cat(result_list)
    _, result_topic = torch.max(result_embedding, 1)

    sp = WhiteSpacePreprocessing(textData.data, stopwords_language='english')
    preprocessed_documents, unpreprocessed_corpus, vocab, retained_labels = sp.preprocess()

    train_target_filtered = textData.targets.squeeze()[retained_labels]

    points = []
    for i in range(1, n_topic+1):
        points.append([math.cos(math.radians(360 / n_topic * i)), math.sin(math.radians(360 / n_topic * i))])
    points = torch.Tensor(np.array(points))
    numpy_points = torch.mm(result_embedding.detach().cpu(), points).numpy()

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    try:
        color_map = train_target_filtered.tolist()
        plt.scatter(numpy_points[:, 0], numpy_points[:, 1], c = color_map, cmap = plt.cm.rainbow, s=1)
    except:
        plt.scatter(numpy_points[:, 0], numpy_points[:, 1], cmap = plt.cm.rainbow, s=1)
    plt.savefig(os.path.join(model_stage1_name, 'fig.png'))
    image = wandb.Image(os.path.join(model_stage1_name, 'fig.png'))
    wandb.log({"topic_distribution": image})



    # training stage 2
    d = {'text': trainds.preprocess_ctm(trainds.nonempty_text), 
        'cluster_label': result_topic.cpu().numpy()}
    cluster_df = pd.DataFrame(data=d)

    docs_per_class = cluster_df.groupby(['cluster_label'], as_index=False).agg({'text': ' '.join})

    count_vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z]{2,}\b')
    # count_vectorizer = CountVectorizer()
    ctfidf_vectorizer = CTFIDFVectorizer()
    count = count_vectorizer.fit_transform(docs_per_class.text)
    ctfidf = ctfidf_vectorizer.fit_transform(count, n_samples=len(cluster_df)).toarray()
    words = count_vectorizer.get_feature_names()

    # transport to gensim
    (gensim_corpus, gensim_dict) = vect2gensim(count_vectorizer, count)
    vocab_list = set(gensim_dict.token2id.keys())
    stopwords = set(line.strip() for line in open(os.path.join('data','snowball_stopwords.txt')))

    normalized = [coherence_normalize(doc) for doc in trainds.nonempty_text]
    gensim_dict = Dictionary(normalized)
    resolution_score = (ctfidf - np.min(ctfidf, axis=1, keepdims=True)) / (np.max(ctfidf, axis=1, keepdims=True) - np.min(ctfidf, axis=1, keepdims=True))

    n_word = args.n_word
    n_topic_word = n_word #/ len(docs_per_class.cluster_label.index)

    words_to_idx = {k: v for v, k in enumerate(words)}
    topic_word_dict = {}
    topic_score_dict = {}
    total_score_cat = []
    for label in docs_per_class.cluster_label.index:
        total_score = resolution_score[label]
        score_higest = total_score.argsort()
        score_higest = score_higest[::-1]
        topic_word_list = [words[index] for index in score_higest]
        
        total_score_cat.append(total_score)
        topic_word_list = [word for word in topic_word_list if word not in stopwords]    
        topic_word_list = [word for word in topic_word_list if word in gensim_dict.token2id]
        topic_word_list = [word for word in topic_word_list if len(word) >= 3]    
        topic_word_dict[docs_per_class.cluster_label.iloc[label]] = topic_word_list[:int(n_topic_word)]
        topic_score_dict[docs_per_class.cluster_label.iloc[label]] = [total_score[words_to_idx[top_word]] for top_word in topic_word_list[:int(n_topic_word)]]
        # print(f"topic {docs_per_class.cluster_label.iloc[label]}: {topic_word_list[:int(n_topic_word)]},")
    total_score_cat = np.stack(total_score_cat, axis = 0)

    # for key in topic_word_dict:
    #     print(f"{key}: {topic_word_dict[key]},")
    #     logger.info(f"{key}: {topic_word_dict[key]},")

    topic_word_set = list(itertools.chain.from_iterable(pd.DataFrame.from_dict(topic_word_dict).values))
    print(f'topic_word_set: {len(topic_word_set)}')
    print(f'after remove dup: {len(remove_dup(topic_word_set))}')
    word_candidates = remove_dup(topic_word_set)[:n_word]
    print(f'word_candidates: {len(word_candidates)}')
    
    
    with open(os.path.join(model_stage1_name, 'word_candidates'), 'w') as file:
        for item in word_candidates:
            file.write(f"{item}\n")
    np.save(os.path.join(model_stage1_name, 'total_score_cat.npy'), total_score_cat)

    with open(os.path.join(model_stage1_name, 'words_to_idx.pkl'), 'wb') as file:
        pickle.dump(words_to_idx, file)

    weight_candidates = {}

    try:
        for candidate in word_candidates:
            weight_candidates[candidate] = [total_score_cat[label, words_to_idx[candidate]] for label in range(n_cluster)]

        weight_cand_to_idx = {k: v for v, k in enumerate(list(weight_candidates.keys()))}
        weight_cand_matrix = np.array(list(weight_candidates.values()))

        np.save(os.path.join(model_stage1_name, 'weight_cand_matrix.npy'), weight_cand_matrix)
    except:
        logger.info('docs are not spread enough, cant create weight_candidates matrix')
        print('docs are not spread enough, cant create weight_candidates matrix')
