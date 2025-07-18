import os
from datetime import datetime
from itertools import combinations
import gensim.downloader
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn import metrics
import wandb
from contextualized_topic_models.evaluation import InvertedRBO

# word2vec_google = gensim.downloader.load('word2vec-google-news-300')

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, preds):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results

def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)

def evaluate_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale', tune=False, logger=None):
    if tune:
        results = {
            'acc': 0,
            'macro-F1': 0
        }
        for C in [0.1, 1, 10, 100, 1000]:
            for gamma in ['scale', 'auto', 10, 1, 0.1, 0.01]:
                for kernel in ['rbf', 'linear']:
                    print(f'C: {C}, gamma: {gamma}, kernel: {kernel}')
                    if logger:
                        logger.info(f'C: {C}, gamma: {gamma}, kernel: {kernel}')
                    if classifier == 'SVM':
                        clf = SVC(C=C, kernel=kernel, gamma=gamma)
                    else:
                        raise NotImplementedError

                    clf.fit(train_theta, train_labels)
                    preds = clf.predict(test_theta)
                    this_results = {
                        'acc': accuracy_score(test_labels, preds),
                        'macro-F1': f1_score(test_labels, preds, average='macro')
                    }
                    results = {
                        key: max(results[key], this_results[key])
                        for key in results
                    }
                    print(f'Accuracy: {this_results["acc"]}, Macro-F1: {this_results["macro-F1"]}')
                    if logger:
                        logger.info(f'Accuracy: {this_results["acc"]}, Macro-F1: {this_results["macro-F1"]}')
                    wandb.log(this_results)
    else:
        if classifier == 'SVM':
            clf = SVC(gamma=gamma)
        else:
            raise NotImplementedError

        clf.fit(train_theta, train_labels)
        preds = clf.predict(test_theta)
        results = {
            'acc': accuracy_score(test_labels, preds),
            'macro-F1': f1_score(test_labels, preds, average='macro')
        }
    return results

def get_topic_qualities(topic_word_list, palmetto_dir, **kwargs):
    """Get topic coherece, similarity, and topic diversity
    arguments
    =========
    topic_word_list: list of list of str
                     or str
    palmetto_dir: str, directory where palmetto JAR and Wikipedia index are
    reference_corpus: list of list of str
    
    returns
    =========
    dict"""
    if isinstance(topic_word_list, list):
        assert isinstance(topic_word_list, list)
        assert isinstance(topic_word_list[0], list)
        assert isinstance(topic_word_list[0][0], str)
        
        if 'filename' in kwargs:
            filename = save_topic_top_keywords(topic_word_list, filename=kwargs['filename'])
        else:
            filename = save_topic_top_keywords(topic_word_list, filename=None)
    elif isinstance(topic_word_list, str):
        filename = topic_word_list
        with open(filename, 'r') as f:
            topic_word_list = [line.split() for line in f.readlines() if len(line) > 0]
    
    print(palmetto_dir)
    
    # os.system(f'java -jar {palmetto_dir}/palmetto.jar {palmetto_dir}/wikipedia/wikipedia_bd umass {filename} > {filename[:-4]}_umass.txt')
    # with open(f"{filename[:-4]}_umass.txt", 'r') as f:
    #     tmpstr = '\n'.join(f.readlines())
    # umass = read_palmetto_result(tmpstr)
    os.system(f'java -jar {palmetto_dir}/palmetto.jar {palmetto_dir}/wikipedia/wikipedia_bd npmi {filename} > {filename[:-4]}_npmi.txt')
    with open(f"{filename[:-4]}_npmi.txt", 'r') as f:
        tmpstr = '\n'.join(f.readlines())
    npmi = read_palmetto_result(tmpstr)
    os.system(f'java -jar {palmetto_dir}/palmetto.jar {palmetto_dir}/wikipedia/wikipedia_bd c_p {filename} > {filename[:-4]}_cp.txt')
    with open(f"{filename[:-4]}_cp.txt", 'r') as f:
        tmpstr = '\n'.join(f.readlines())
    cp = read_palmetto_result(tmpstr)
    # os.system(f'java -jar {palmetto_dir}/palmetto.jar {palmetto_dir}/wikipedia/wikipedia_bd uci {filename} > {filename[:-4]}_uci.txt')
    # with open(f"{filename[:-4]}_uci.txt", 'r') as f:
    #     tmpstr = '\n'.join(f.readlines())
    # uci = read_palmetto_result(tmpstr)
    os.system(f'java -jar {palmetto_dir}/palmetto.jar {palmetto_dir}/wikipedia/wikipedia_bd C_V {filename} > {filename[:-4]}_CV.txt')
    with open(f"{filename[:-4]}_CV.txt", 'r') as f:
        tmpstr = '\n'.join(f.readlines())
    CV = read_palmetto_result(tmpstr)
    
    irbo = InvertedRBO(topic_word_list).score()
    # npmi_in = None
    # uci_in = None
    # if 'reference_corpus' in kwargs: # reference_corpus: list of list of words
    #     reference_corpus = kwargs['reference_corpus']
    #     reference_dictionary = Dictionary(reference_corpus)
    #     reference_dictionary.add_documents(topic_word_list)
    #     cm = CoherenceModel(topics=topic_word_list, texts=reference_corpus, dictionary=reference_dictionary, coherence='c_npmi', topn=10)
    #     npmi_in = cm.get_coherence()
    #     cm = CoherenceModel(topics=topic_word_list, texts=reference_corpus, dictionary=reference_dictionary, coherence='c_uci', topn=10)
    #     uci_in = cm.get_coherence()
    # sim = get_average_word2vec_similarity(topic_word_list, word2vec_google)
    sim = 0
    all_word_set = set()
    all_word_list = []
    for word_list in topic_word_list:
        all_word_set.update(word_list)
        all_word_list += word_list
    diversity = len(all_word_set) / len(all_word_list)
    return {'topic_N': len(topic_word_list),
            # 'umass_wiki': umass,
           'npmi_wiki': npmi,
        #    'npmi_in': npmi_in,
        #    'uci_wiki': uci,
        #    'uci_in': uci_in,
           'CV_wiki': CV,
           'cp_wiki': cp,
           'sim_w2v': sim,
           'diversity': diversity,
           'filename': filename,
           'irbo': irbo}

    
def save_topic_top_keywords(top_keywords_list, filename=None):
    # Save the keywords, seperated with spaces
    if filename is None:
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        filename = f"{now}.txt"
    with open(filename, 'w') as f:
        for keywords in top_keywords_list:
            f.write(' '.join(keywords))
            f.write('\n')
    return filename

def read_palmetto_result(result_text):
    # summarize the result from palmetto JAR
    result_lines = result_text.split('\n')
    if 'org.aksw.palmetto.Palmetto' in result_lines[0]:
        result_lines = result_lines[1:]
    val_l = []
    for line in result_lines:
        if line == '':
            continue
        val = line.split('\t')[1]
        val_l.append(float(val))
    print(val_l)
    print(sum(val_l) / len(val_l))
    return sum(val_l) / len(val_l)

def get_average_word2vec_similarity(topic_word_list, model):
    similarity_list = []
    missing_word_count = 0
    for topic, word_list in enumerate(topic_word_list):
        word_list_filtered = [word for word in word_list if model.has_index_for(word)]
        missing_word_count += len(word_list) - len(word_list_filtered)
        for word1, word2 in combinations(word_list_filtered, 2):
            similarity = model.similarity(word1, word2)
            similarity_list.append(similarity)
    return sum(similarity_list) / len(similarity_list)
