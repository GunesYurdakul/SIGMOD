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
import pickle
import helper as helper
import pandas as pd


# Returns true if search_str contains regex exp
def contains (exp, search_str):
    if re.search(exp, search_str):
        return True
    return False

def process_others(blocking_key, brand_list, df):
    for index, row in tqdm(df.iterrows()):
        key_list = []
        brand = row['brand']  
        brand = brand.lower()
        blocking_key = row['blocking_key']

        #print("key ",blocking_key,"brand",brand,"\n")
        # and (brand in brand_list)
        #print(brand)
        #print (brand[0])
        brand = brand.strip()
        for item in brand_list:
            if (blocking_key[0] == 'other' and brand != '' and ((item in brand) or (brand in item))):
                key_list.append(brand)
                #if(item == "vista"):
                 #   print("brand ", brand, "item ", item)
                df.at[index, 'blocking_key'] = item
                
       # elif (blocking_key[0] == 'other' and brand != ' ' and !(brand in brand_list))
            
    print("ended")
    return df


def create_brand_dataframe (dataset_path):
    stop_words=stopwords.words('english')
    print('Creating df')
    skip_words=['alibaba.com','buy',
             'sale',
             'digital',
             'product',
             'ebay',
             'instant',
             'type',
             'camera',
             'camra',
             'india',
             'prices',
             'reviews',
             'plum','black','white','gray','purple','pink','grey','red','silver','orange','gold',
             'advanced',
             'cameras',
             'others',
             'brand',
             'high',
             'resolution',
             'memory',
             'full',
             'bundle',
             'mirrorless',
             'price',
             'body',
             'series',
             'chinese',
             'cheap',
             'tough',
             'refurbished',
             'color']
    print('>>> Creating dataframe...\n')
    columns_df = ['source', 'spec_number', 'spec_id', 'page_title','brand']

    progressive_id = 0
    progressive_id2row_df = {}
    for source in tqdm(os.listdir(dataset_path)):
        for specification in os.listdir(os.path.join(dataset_path, source)):
            specification_number = specification.replace('.json', '')
            specification_id = '{}//{}'.format(source, specification_number)
            with open(os.path.join(dataset_path, source, specification)) as specification_file:
                specification_data = json.load(specification_file)
                brand =' '
                
                page_title = specification_data.get('<page title>').lower()
                model_words=['mega pixel','megapixel',' mp','mp ']
                for key,text in specification_data.items():
                    if type(text)==list:
                        text=' '.join(text)
                    text=text.lower()
                    mp_exists=False
                    for model_word in model_words:  
                        start_idx = text.find(model_word)
                        if start_idx!=-1:
                            mp_exists=True
                            if start_idx>4:
                                page_title+= ' ' + text[start_idx-4:start_idx] + 'mp'
                            else:
                                page_title+= ' ' + text[:start_idx]+ 'mp'
                    if mp_exists:
                        break
                
                if not (specification_data.get('model') is None):
                    model = specification_data.get('model')
                    
            
                    if(type(model)==str):   
                        
                        model = model.split()
                        for m in model:
                            m = m.lower()
                            if m not in page_title and m != "/":
                                page_title = m + " " + page_title + " "

                    else:
                        for m in model:
                            m = m.lower()
                            if m not in page_title :
                                if (m !="/"):
                                      page_title = m + " " + page_title+ " "
                
                

                page_title=page_title.replace(' - ',' ')
                page_title=page_title.replace('-',' ')
                page_title=page_title.replace(',',' ')
                page_title=page_title.replace(' /',' ')
                page_title=page_title.replace(' | ',' ')
                page_title=page_title.replace('|',' ')
                page_title=page_title.replace(' . ',' ')
                page_title=page_title.replace('"',' ')
                page_title=page_title.replace('photo smart','photosmart')
                page_title = page_title.replace('cannon', 'canon') 
                page_title = page_title.replace('ricoh', 'pentax')
                page_title=page_title.replace('minotla','minolta')
                page_title=page_title.replace('minolta','konica')
                page_title=page_title.replace(' dslr','')
                page_title=page_title.replace('dslr ','')
                page_title=page_title.replace(' slr','')
                page_title=page_title.replace('slr ','')
                page_title=page_title.replace(' black','')
                page_title=page_title.replace(' red','')
                page_title=page_title.replace(' white','')
                page_title=page_title.replace(' pink','')
                page_title=page_title.replace(' silver','')
                page_title=page_title.replace(' orange','')
                page_title=page_title.replace(' grey','')
                page_title=page_title.replace(' e ',' ')
                page_title=page_title.replace('compact ','ZZZ')
                page_title=page_title.replace('cyber shot','cybershot')
                page_title=page_title.replace('power shot','powershot')
                page_title=page_title.replace('poweshot','powershot')
                page_title=page_title.replace('*istdl','*ist dl')
                page_title=page_title.replace(' series','')
                page_title=page_title.replace('pentax k','pentax k ')
                page_title=page_title.replace('pentax k  ','pentax k ')
                page_title=page_title.replace('fine pix','finepix')
                page_title=page_title.replace('fuji ','fujifilm ')
                page_title=page_title.replace('fuijifilm','fujifilm')
                page_title=page_title.replace('exlim','exilim')
                page_title=page_title.replace('eos 1dx','eos 1d x')
                page_title=page_title.replace(' + ',' ')
                page_title=page_title.replace(' (',' ')
                page_title=page_title.replace(') ',' ')
                page_title=page_title.replace(' mp','mp')
                page_title=page_title.replace(' p ','p ')
                page_title=page_title.replace('megapixel','mp')
                page_title=page_title.replace('hikvisionip','hikvision')
                page_title=page_title.replace('hikvision1.3mp','hikvision 1.3mp')
                page_title=page_title.replace('hiksion','hikvision')
                page_title=page_title.replace('hiksision','hikvision')
                page_title=page_title.replace('sony alpha a','sony a')
                page_title=page_title.replace(' î','   ')
                
                
                
                page_title=' '.join([ word for word in page_title.lower().split() if (not (word.lower() in stop_words)) and (not(word.lower() in skip_words))])
                
                if not (specification_data.get('brand') is None):
                    brand = specification_data.get('brand')
                if(isinstance(brand, str)):    
                    row = (source, specification_number, specification_id, page_title, brand)
                else:
                    row = (source, specification_number, specification_id, page_title, brand[1])
                progressive_id2row_df.update({progressive_id: row})
                progressive_id += 1
    df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
  #  print(df)
    print('>>> Dataframe created successfully!\n')
    return df


def subgroup_blocking(title, brand):
    subgroup_keys = ["coolpix", "powershot", "eos", "alpha"]
    if (brand == "canon" or brand == "nikon" or brand == "sony"):
        for i in subgroup_keys:
            if i in title:
                brand = i
                return brand
    return brand


def brand_blocking_keys(df):
    myList = df.loc[(df['source'] != "cammarkt.com") 
                    & (df['source'] != "www.alibaba.com") 
                    & (df['source'] != "www.buzzillions.com") 
                    & (df['source'] != "www.canon-europe.com")
                    & (df['source'] != "www.ebay.com")
                    & (df['source'] != "www.gosale.com"), ['page_title']] 
    #blackmagic
    #ion camera ??? 
    myList  = myList['page_title'].tolist()
    myList = [i.split(' ')[0] for i in myList]
    mySet = set(myList)
    #mySet.remove("buy")
    mySet.remove("video")
    mySet.remove("get")
    mySet.remove("purchase")
    mySet.remove("tokina")
    mySet.remove("samyang")
    mySet.remove("lowepro")
    mySet.remove("canonpowershot")
    mySet.remove("pov")
    mySet.remove("ion") #new
    #mySet.remove("bell+howell")
    mySet.update(["emerson","enxun", "lg", "svp", "vizio", "vista", "philips", "toshiba","aigo", "phase", "advert", "thomson", "medion", "minox", "vageeswari", 
                  "memoto", "hasselblad", "bell", "epson", "dahua", "minolta", "konica", "hikvision", "sanyo","pioneer", "shimano","sj4000", "vibe",
                  "keedox", "blackmagic", "rieter", "wopson",
                  "croco", "g-cover","besnfoto", "eirmai", "wetrans", "crayola","sealife", "fvanor", "concepts", "sj4000", "vibe", "dxg", "fotopix","keedox","acorn",
                  "brinno", "lowrance", "barbie","kitty","apple"])
    mySet.update(["hikvisionip", "hikvision1.3mp", "hiksion", "hiksision"])  #edgecases
    
    #hikvisionip, hikvision1.3mp hiksion hiksision 
    return mySet  

#Onur
def compute_brand_blocking(df):
    """Function used to compute blocks before the matching phase

    Gets a set of blocking keys and assigns to each specification the first blocking key that will match in the
    corresponding page title.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles

    Returns:
        df (pd.DataFrame): The Pandas DataFrame containing specifications, page titles and blocking keys
    """

    print('>>> Computing blocking...')
    blocking_keys = brand_blocking_keys(df)

    df['blocking_key'] = ''
    for index, row in tqdm(df.iterrows()):
        page_title = row['page_title']
        page_title_list = page_title.split(" ")
        key_list = []
        for blocking_key in blocking_keys:
            if blocking_key in page_title_list:
                #df.at[index, 'blocking_key'] = blocking_key   ##multiple groups ?
                #break
                if(blocking_key =="general"):
                    blocking_key = "ge"

                blocking_key = subgroup_blocking(page_title, blocking_key)
                key_list.append(blocking_key)
        if not key_list:
            key_list.append("other")
        key_list=set(key_list)
        
        #edge cases for products belonging to 2 groups
        if(len(key_list) > 1 and (df.at[index,'brand'] == ' ') and ("leica" not in key_list) and ("tamron" not in key_list) and ("lowepro" not in key_list)and  ("sigma" not in key_list) and  ("polaroid" not in key_list) ):
            if("fuji" not in key_list and "fujifilm" not in key_list):
                if("konica" not in key_list and "minolta" not in key_list):
                    key_list = ['accessory']
        if("sandisk" in key_list and (df.at[index,'brand'] != ' ') and len(key_list) > 1 ):
            key_list.remove("sandisk")
        if("other" in key_list and "lowepro" in (page_title_list) ):
            key_list.add("lowepro")
        if("other" in key_list and "vistaquest" in (page_title_list) ):
            key_list=["vista"]
            #vistaquest 
        if("other" in key_list and "coolpix" in (page_title_list) and "nokia" not in key_list ):
            key_list=["coolpix"]
        if("ricoh" in key_list):
            key_list.remove('ricoh')
            key_list=['pentax']
         

        key_list = list(key_list)
        df.at[index, 'blocking_key'] = key_list      
    df = process_others(blocking_key, blocking_keys, df)
    print('>>> Blocking computed successfully!\n')
    df = df.explode('blocking_key')
    df.loc[df['blocking_key'] == 'fuji', 'blocking_key'] = 'fujifilm'
    df.loc[df['blocking_key'] == 'bell+howell', 'blocking_key'] = 'bell'
    return df
