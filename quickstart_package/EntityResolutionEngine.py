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
import numpy as np
import main as sigmod
from itertools import combinations 
import helper as helper
from fuzzywuzzy import  fuzz
from tqdm import tqdm


class EntityResolutionEngine():

    
    def __init__(self, dataset_df, labelled_df):
        self.dataset_df = dataset_df
        self.labelled_df = labelled_df
        self.block_df=None
        self.output_df=pd.DataFrame(columns=['left_spec_id','right_spec_id','left_page_title','right_page_title'])
        self.product_clusters=list()

    # Returns all model words of a given word list
    def extract_model_words (self,word_list):
        all_model_words = []
        is_model_word = re.compile('[0-9]+[^0-9]+|[^0-9]+[0-9]+')

        for i in range(len(word_list)): 
            if (contains(is_model_word, word_list[i])):
                all_model_words.append(word_list[i])

        return all_model_words

    #get partial fuzzy similarity using fuzzywuzzy library, which uses Levehstein distance to compute
    #cosine distance. using 3grams of strings
    def get_similarity(self,str1,str2):
        try:
            #helper.cosine_sim(helper.text_to_ngrams(str1,3,'spaces'),helper.text_to_ngrams(str2,3,'spaces'))
            similarity=fuzz.partial_ratio(str1,str2)
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
        model_vec2=self.word2vec_model.wv[brand]
        if len(words)>1 and words[0] in self.word2vec_model.wv: 
            model_vec= self.word2vec_model.wv[words[0]] 
        if len(words)>2 and words[1] in self.word2vec_model.wv: 
            model_vec2= self.word2vec_model.wv[words[1]] 
        return np.concatenate((model_vec,model_vec2))
    
    #set feature vector of selected block
    def set_block(self,blocking_key):
        self.block_df = self.grouped_df.get_group(blocking_key)
        self.block_df['concat_wordvector']=''  
        return
    
    #get pairs with similarity greater than the given threshold value
    def get_pairs(self, threshold,labelled_set=True):
        if labelled_set:
            labelled_index = list(set(list(self.labelled_df.left_spec_id.values)+list(self.labelled_df.right_spec_id.values)))   
            self.labelled_data_df = self.dataset_df.loc[labelled_index]
            dataset_df=self.labelled_data_df
        else:
            dataset_df=self.dataset_df
        model_names= set(dataset_df['model_name'])
        self.grouped_df = dataset_df.groupby(['model_name'])
        i=0
        for model_name in model_names:
            i+=1
            if i%10==0:
                print('% ',100*(i/len(list(model_names))))
            if model_name !='':
                self.block_df=self.grouped_df.get_group(model_name)
                self.product_clusters=[list(self.block_df.index)]
                if len(self.block_df)==1 or model_name =='':
                    continue

            for product_group in self.product_clusters:
                pairs=combinations(product_group,2)
                for pair in pairs:
                    similarity = self.get_similarity(dataset_df.loc[pair[0]].page_title,dataset_df.loc[pair[1]].page_title)
                    if similarity>threshold:
                        self.output_df = self.output_df.append({'left_spec_id': pair[0],'right_spec_id': pair[1], 'left_page_title': dataset_df.loc[pair[0]].page_title,'right_page_title':dataset_df.loc[pair[1]].page_title }, ignore_index=True)
        return 
    
    #extract_model_names
    def get_model_names(self):
        models={}
        grouped=self.dataset_df.groupby('blocking_key')
        for blocking_key in set(self.dataset_df.blocking_key):
            self.block_df=grouped.get_group(blocking_key)
            self.block_df['model']=self.block_df.page_title.apply(lambda x: [' '.join(x[idx:].split()[:1]) for idx in [x.start() for x in re.finditer(blocking_key+' ', x)]])
            self.block_df['model']=self.block_df['model'].apply(lambda x:x[0] if len(x)>0 else '')
            models.update(self.block_df.groupby('model').size().to_dict())

        del models['']

        counts = np.array(list(models.values()))
        limit=np.percentile(counts,50)
        keys=list(models.keys()).copy()
        skip_keys=['sigma','other','tamron','samsung nx','eos 5d','sony mavica','sony digital','sony ccd','hikvision ir','svp','canon ef','leica','vizio','enxun','hasselblad','disney','casio exilim']

        for key in keys:    
            
            if models[key]>limit and (key not in skip_keys) :
                self.block_df=grouped.get_group(key)
                if key=='hikvision':
                    key='ds '        
                    self.block_df['model']=self.block_df.page_title.apply(lambda x: [' '.join(x[idx:].split()[:2]) + ' ' for idx in [x.start() for x in re.finditer(key, x)]])
                    self.block_df['model']=self.block_df['model'].apply(lambda x:x[0] if len(x)>0 else '')
                    key='hikvision'
                    models[key]=self.block_df.groupby('model').size().to_dict()


                else:
                    self.block_df['model']=self.block_df.page_title.apply(lambda x: [' '.join(x[idx:].split()[:2]) + ' ' for idx in [x.start() for x in re.finditer(key+' ', x)]])
                    self.block_df['model']=self.block_df['model'].apply(lambda x:x[0] if len(x)>0 else '')
                    models[key]=self.block_df.groupby('model').size().to_dict()

        skip_keys=['sigma','tamron','samsung nx','eos 5d','sony mavica','sony digital','sony ccd','hikvision ir','svp','canon ef','leica','vizio','enxun','hasselblad','disney','casio exilim']

        for covering_key in [key for key in models.keys() if type(models[key])==dict]:  
            if  covering_key=='hikvision':
                continue
            for key,value in models[covering_key].items(): 
                if covering_key=='nikon' and key !='nikon 1 ':
                    continue

                if models[covering_key][key]>50 and key!='' and (key not in skip_keys):
                    self.block_df=grouped.get_group(covering_key)
                    self.block_df['model']=self.block_df.page_title.apply(lambda x: [' '.join(x[idx:].split()[:3]) + ' ' for idx in [x.start() for x in re.finditer(key, x)]])
                    self.block_df['model']=self.block_df['model'].apply(lambda x:x[0] if len(x)>0 else '')
                    models[covering_key][key]=self.block_df.groupby('model').size().to_dict()

                    del models[covering_key][key]['']

        models['nikon']

        skip_keys=['sigma','tamron','samsung nx','eos 5d','sony mavica','sony digital','sony ccd','hikvision ir','svp','canon ef','leica','vizio','enxun','hasselblad','disney','casio exilim']
        enter_sub_keys=['sony alpha slt ','sony alpha nex ','sony alpha ilce ','sony alpha 7 ','panasonic lumix dmc ' ,'sony cybershot dsc ','sony alpha dsc ']
        for covering_key in [key for key in models.keys() if type(models[key])==dict]:  
            if covering_key == 'nikon' or covering_key=='hikvision':
                continue
            for key,value in models[covering_key].items(): 
                if type(models[covering_key][key])==dict:
                    for sub_key,value_x in models[covering_key][key].items():
                        if sub_key in enter_sub_keys:
                            self.block_df=grouped.get_group(covering_key)
                            self.block_df['model']=self.block_df.page_title.apply(lambda x: [' '.join(x[idx:].split()[:4]) + ' ' for idx in [x.start() for x in re.finditer(sub_key, x)]])
                            self.block_df['model']=self.block_df['model'].apply(lambda x:x[0] if len(x)>0 else '')
                            models[covering_key][key][sub_key]=self.block_df.groupby('model').size().to_dict()
        self.model_names = self.clean(models)
        return 

    #clean model names
    def clean(self,models_):
        for key,value in list(models_.items()).copy():
            if (len(key.split())>len(list(set(key.split())))) or ('.' in key) or ('fujifilm finepix' in key and ('.' in key )) or key=='':
                del models_[key]
            elif type(value)==dict:
                self.clean(models_[key])
            elif ('fujifilm finepix' in key and ('.' in key )) or ('compact' in key) or key=='' or (('eos' in key ) and ('rebel' not in key) and len(key.split())>2) or  (('sony' in key ) and ('alpha' in key)): 
                del models_[key]
        return models_
    
    #grouping same products in the output pairs, returns an undirected graph
    def grouping_same_products(self,spec_to_idx,idx_to_spec,all_specs):

        graph=np.zeros((len(all_specs),len(all_specs)))

        for idx,row in self.output_df.iterrows():
            left_idx=spec_to_idx[row['left_spec_id']]
            right_idx=spec_to_idx[row['right_spec_id']]
            graph[left_idx,right_idx]=1
            graph[right_idx,left_idx]=1

        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        same_products={}

        for i in range(len(labels)):
            if labels[i] not in same_products.keys():
                same_products[labels[i]]=[]
            same_products[labels[i]].append(idx_to_spec[i])

        return same_products,graph
    
    #grouping same products labelled pairs, returns an undirected graph
    def grouping_same_products_from_labelled_set(self):
        correct_pairs = self.labelled_df[self.labelled_df['label']==1]
        all_specs = list(self.labelled_df['left_spec_id'])+list(self.labelled_df['right_spec_id'])
        all_specs = list(set(all_specs))

        spec_to_idx = dict(zip(all_specs,list(range(len(all_specs)))))
        idx_to_spec = dict(zip(list(range(len(all_specs))),all_specs))
        graph=np.zeros((len(all_specs),len(all_specs)))

        for idx,row in correct_pairs.iterrows():
            left_idx=spec_to_idx[row['left_spec_id']]
            right_idx=spec_to_idx[row['right_spec_id']]
            graph[left_idx,right_idx]=1
            graph[right_idx,left_idx]=1

        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        same_products={}

        for i in range(len(labels)):
            if labels[i] not in same_products.keys():
                same_products[labels[i]]=[]
            same_products[labels[i]].append(idx_to_spec[i])

        return same_products,graph,spec_to_idx,idx_to_spec,all_specs
    
    #assign each product to a model name group
    def compute_model_name_blocking(self):

        print('>>> Computing blocking...')
        subgroup_keys = ["coolpix", "powershot", "eos", "alpha"]

        self.dataset_df['model_name'] = ''
        group_df=self.dataset_df.groupby('blocking_key')
        for blocking_key,value in self.model_names.items():
            self.block_df = group_df.get_group(blocking_key)
            for index, row in tqdm(self.block_df.iterrows()):
                page_title = row['page_title']
                brand = row['blocking_key']            
                if (brand == "canon" or brand == "nikon" or brand == "sony"):
                    for i in subgroup_keys:
                        if i in page_title:
                            page_title=page_title.replace(brand,'')
                
                page_title=page_title.replace('general electric','ge ')

                if blocking_key in page_title:
                    self.dataset_df.at[index, 'model_name'] = blocking_key    

                    if type(value)==dict:
                        for sub_key,sub_value in value.items():

                            if brand=='hikvision':
                                if sub_key.replace(brand,'') in page_title:
                                    self.dataset_df.at[index, 'model_name'] = sub_key
                                    if type(sub_value)==dict:
                                        for sub_key2,sub_value2 in sub_value.items():
                                            if sub_key2.replace(brand,'') in page_title:
                                                self.dataset_df.at[index, 'model_name'] = sub_key2.replace(' '+sub_key,'')

                            if sub_key in page_title:
                                self.dataset_df.at[index, 'model_name'] = sub_key
                                if type(sub_value)==dict:
                                    for sub_key2,sub_value2 in sub_value.items():
                                        if sub_key2.replace(brand,'') in page_title or sub_key2.replace(' '+sub_key,'') in page_title:
                                            self.dataset_df.at[index, 'model_name'] = sub_key2.split()[0] + ' '+ sub_key2.split()[2]
                                            if type(sub_value2)==dict:
                                                for sub_key3,sub_value3 in sub_value2.items():
                                                    if sub_key3.replace(brand,'') in page_title:
                                                        self.dataset_df.at[index, 'model_name'] =  sub_key3.split()[0] + ' ' + sub_key3.split()[3]
        return

    #calculate f measure using the graphs of ground truth and our output pairs
    def calculate_f_measure(self):
        print('calculating precision, recall and f-measure ...')
        same_products,ground_truth,spec_to_idx,idx_to_spec, all_specs = self.grouping_same_products_from_labelled_set()
        our_same_products,our_truth = self.grouping_same_products(spec_to_idx,idx_to_spec,all_specs)
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        columns = ground_truth.shape[1]
        false_negatives=[]
        for j in range(columns):
            for i in range(columns):
                if our_truth[i][j] == 1 and ground_truth[i][j] == 1:
                    TP += 1
                elif our_truth[i][j] == 1 and ground_truth[i][j] == 0:
                    FP += 1
                elif our_truth[i][j] == 0 and ground_truth[i][j] == 1:
                    FN += 1
                    false_negatives.append([i,j])
                else:
                    TN += 1

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        f_measure = (2 * p * r) / (p + r)
        return p, r, f_measure, false_negatives
         