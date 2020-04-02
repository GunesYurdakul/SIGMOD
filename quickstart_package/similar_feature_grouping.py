import nltk
from nltk import everygrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
import nltk

nltk.download('stopwords')


def prepare_text_for_word2vec(data_dict):
    stop_words=stopwords.words()
    all_text=[]
    i=0
    for id_, features in data_dict.items():
        for feature,feature_text in features.items():
            if type(feature_text)==list:
                feature_text=' '.join(feature_text)
            feature_text = ' '.join([ word for word in feature_text.lower().split() if not word.lower() in stop_words])
            tokens=[feature] + [' '.join(word_list) for word_list in list(everygrams(feature_text, 1, 3))]
            all_text.append(tokens)
        if i%1000==0:
            print(i/len(data_dict.items()))
        i+=1
    return all_text


def get_grouped_features(dataset_df,assigned_clusters,labels):
    NUM_CLUSTERS=max(assigned_clusters)+1
    cluster2features={}
    i=0
    for cluster in assigned_clusters:
        if cluster not in cluster2features.keys():
             cluster2features[cluster]=[]
        cluster2features[cluster].append(labels[i])
        i+=1

    for idx in range(NUM_CLUSTERS):
        dataset_df[idx]=dataset_df['spec_id'].apply(lambda x: '')

    for idx in range(NUM_CLUSTERS):
        dataset_df[idx]=dataset_df['all_features'].apply(lambda x: set_feature_value(idx,x,cluster2features))

    return dataset_df

def set_feature_value(idx,features,cluster2features):
    grouped_features=[]
    for feature,value in features.items():
        if feature in cluster2features[idx]:
            grouped_features.append(str(value))
    if len(grouped_features)==0:
        return ''
    return ' '.join(grouped_features)