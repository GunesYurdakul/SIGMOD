import os
import json
import pandas as pd
import re
import itertools
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk import everygrams
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
import main as sigmod
from itertools import combinations 
import helper as helper
class EntityResolutionEngine():


    def __init__(self, word2vec_model, dataset_df, labelled_df):
        self.word2vec_model = word2vec_model
        self.dataset_df = dataset_df
        self.grouped_df = self.dataset_df.groupby(['blocking_key'])
        self.labelled_df = labelled_df
        self.block_df=None
        self.output_df=pd.DataFrame(columns=['left_spec_id','right_spec_id','left_page_title','right_page_title'])
        self.product_clusters=list()

    def set_model_words_column(self):
        
        self.dataset_df['words_to_compare']=''
        for index, row in self.dataset_df.iterrows():
            x=row['page_title']
            brand_name = row['blocking_key']
            self.dataset_df.at[index,'words_to_compare']=' '.join([' '.join(x[idx:].split()[:3]) for idx in [x.start() for x in re.finditer(brand_name, x)]] + [' '.join(sigmod.extract_model_words(token)) for token in list(everygrams(x.split(),2,3))])                                       
    
    def get_similarity(self,str1,str2):
        try:
            similarity=helper.cosine_sim(helper.text_to_ngrams(str1,3,'chars'),helper.text_to_ngrams(str2,3,'chars'))
        except:
            return 0
        return similarity

    #get avg feature vector: concatanation of model name(dim 200) and model words(dim 200)
    def get_avg_vector(self,x,brand):
        avg=np.zeros((200,))
        x=x.replace(brand,'')
        x=x.replace('  ',' ')
        words = x.split() 
        model_vec=self.word2vec_model.wv[brand]
        if len(words)>1 and words[0] in self.word2vec_model.wv: 
            model_vec= self.word2vec_model.wv[words[0]] 
        if len(words)>2 and words[1] in self.word2vec_model.wv: 
            model_vec2= self.word2vec_model.wv[words[1]] 
        for i in range(len(words[1:])):
            word=words[i]
            if word in self.word2vec_model.wv:
                avg += self.word2vec_model.wv[word]  
        if len(words)>0:
            avg= avg/(len(words))
        return np.concatenate((model_vec,model_vec2,avg))
    
    #set feature vector of selected block
    def set_block(self,blocking_key):
        self.block_df = self.grouped_df.get_group(blocking_key)
        self.block_df['concat_wordvector']=''
        for index, row in self.block_df.iterrows():
            x=row['page_title']
            self.block_df.at[index,'concat_wordvector']=self.get_avg_vector(\
                ' '.join([' '.join(x[idx:].split()[:4]) for idx in [x.start() for x in re.finditer(blocking_key, x)]] + [' '.join(sigmod.extract_model_words(token)) for token in list(everygrams(x.split(),2,3))]),blocking_key)     
        return
    
    #for dimensionality reduction
    def run_TSNE(self,n_components_=2,n_iter_=1000, perplexity_=20):
        X=np.zeros((len(self.block_df),600))
        for i in range(len(self.block_df)):
            X[i]=self.block_df['concat_wordvector'][i]

        tsne = TSNE(n_components=n_components_, random_state=0, n_iter=n_iter_, perplexity=perplexity_)
        np.set_printoptions(suppress=True)
        self.T = tsne.fit_transform(X)
        self.labels = self.block_df['spec_number'].values
        return self.labels,self.T
    
    #clustering
    def run_DBSCAN(self,eps_=8, min_samples_=3):
        kclusterer = DBSCAN(eps=eps_, min_samples=min_samples_).fit(self.T)
        self.assigned_clusters = kclusterer.labels_
        self.block_df['inblock_cluster'] = self.assigned_clusters
        
        block_group_df = self.block_df.groupby('inblock_cluster')
        cluster_indices = list(set(self.assigned_clusters))
        for i in range(min(cluster_indices),max(cluster_indices)+1):
            self.product_clusters.append(list(block_group_df.get_group(i).index))
        return

    def get_pairs(self, threshold):
        i=0
        for product_group in self.product_clusters:
            i+=1
            if i%5==0:
                print(i/(len(self.product_clusters)))
            pairs=combinations(product_group,2)
            for pair in pairs:
                similarity = self.get_similarity(self.dataset_df.loc[pair[0]].words_to_compare,self.dataset_df.loc[pair[1]].words_to_compare)
                if similarity>threshold:
                    self.output_df = self.output_df.append({'left_spec_id': pair[0],'right_spec_id': pair[1], 'left_page_title': self.dataset_df.loc[pair[0]].page_title,'right_page_title':self.dataset_df.loc[pair[1]].page_title }, ignore_index=True)
        return 
    