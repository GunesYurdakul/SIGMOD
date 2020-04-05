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


def get_all_keys_values(dataset_path):
    data_dict={}
    progressive_id = 0
    progressive_id2row_df = {}
    for source in tqdm(os.listdir(dataset_path)):
        for specification in os.listdir(os.path.join(dataset_path, source)):
            specification_number = specification.replace('.json', '')
            specification_id = '{}//{}'.format(source, specification_number)
            with open(os.path.join(dataset_path, source, specification)) as specification_file:
                specification_data = json.load(specification_file)
                data_dict[specification_id]=specification_data
    return data_dict

def grouping_same_products_from_labelled_set(labelled_df):
    correct_pairs = labelled_df[labelled_df['label']==1]
    all_specs = list(labelled_df['left_spec_id'])+list(labelled_df['right_spec_id'])
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
        
    return same_products

# Returns true if search_str contains regex exp
def contains (exp, search_str):
    if re.search(exp, search_str):
        return True
    return False

# Returns all model words of a given matrix
def extract_model_words (matrix):
    all_model_words = []
    is_model_word = re.compile('[0-9]+[^0-9]+|[^0-9]+[0-9]+')

    for i in range(len(matrix)): 
        if (contains(is_model_word, matrix[i])):
            all_model_words.append(matrix[i])

    return all_model_words

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
#pairs_df = sigmod.get_block_pairs_df(dataset_df)
#blocking_keys = brand_blocking_keys(df)



#Creates dataframe for brand blocking
def create_brand_dataframe (dataset_path):
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


def create_dataframe(dataset_path):
    """Function used to create a Pandas DataFrame containing specifications page titles

    Reads products specifications from the file system ("dataset_path" variable in the main function) and creates a Pandas DataFrame where each row is a
    specification. The columns are 'source' (e.g. www.sourceA.com), 'spec_number' (e.g. 1) and the 'page title'. Note that this script will consider only
    the page title attribute for simplicity.

    Args:
        dataset_path (str): The path to the dataset

    Returns:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles
    """

    print('>>> Creating dataframe...\n')
    columns_df = ['source', 'spec_number', 'spec_id', 'page_title']

    progressive_id = 0
    progressive_id2row_df = {}
    for source in tqdm(os.listdir(dataset_path)):
        for specification in os.listdir(os.path.join(dataset_path, source)):
            specification_number = specification.replace('.json', '')
            specification_id = '{}//{}'.format(source, specification_number)
            with open(os.path.join(dataset_path, source, specification)) as specification_file:
                specification_data = json.load(specification_file)                    
                page_title = specification_data.get('<page title>').lower()
                row = (source, specification_number, specification_id, page_title)
                progressive_id2row_df.update({progressive_id: row})
                progressive_id += 1
    df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    print(df)
    print('>>> Dataframe created successfully!\n')
    return df


def __get_blocking_keys(df):
    """Private function used to calculate a set of blocking keys

    Calculates the blocking keys simply using the first three characters of the page titles. Each 3-gram extracted in
    this way is a blocking key.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles
    Returns:
        blocking_keys (set): The set of blocking keys calculated
    """

    blocking_keys = set()
    for _, row in df.iterrows():
        page_title = row['page_title']
        blocking_key = page_title[:3]
        blocking_keys.add(blocking_key)
    return blocking_keys



def compute_blocking(df):
    """Function used to compute blocks before the matching phase

    Gets a set of blocking keys and assigns to each specification the first blocking key that will match in the
    corresponding page title.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles

    Returns:
        df (pd.DataFrame): The Pandas DataFrame containing specifications, page titles and blocking keys
    """

    print('>>> Computing blocking...')
    blocking_keys = __get_blocking_keys(df)
    df['blocking_key'] = ''
    for index, row in tqdm(df.iterrows()):
        page_title = row['page_title']
        for blocking_key in blocking_keys:
            if blocking_key in page_title:
                df.at[index, 'blocking_key'] = blocking_key
                break
    print(df)
    print('>>> Blocking computed successfully!\n')
    return df

def subgroup_blocking(title, brand):
    subgroup_keys = ["coolpix", "powershot", "eon", "eos", "alpha"]
    if (brand == "canon" or brand == "nikon" or brand == "sony"):
        for i in subgroup_keys:
            if i in title:
                brand = i
                return brand
    return brand


#Onur
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
    mySet.remove("buy")
    mySet.remove("video")
    mySet.remove("get")
    mySet.remove("purchase")
    #mySet.remove("bell+howell")
    mySet.update(["nylon", "leather", "metal","emerson","enxun", "lg", "svp", "vizio", "vista", 
                  "philips", "toshiba","aigo", "phase", "advert", "thomson", "medion", "minox", "vageeswari", 
                  "memoto", "hasselblad", "bell", "epson", "dahua", "minolta", "konica", "hikvision", "sanyo", 
                  "plastic", "security", "aluminum" ])
    #mySet.add("svp")
    

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
                if(blocking_key =="fuji" or blocking_key == "fuijifilm"):
                    blocking_key = "fujifilm"
                if(blocking_key =="general"):
                    blocking_key = "ge"
                if(blocking_key =="minolta"):
                    blocking_key = "konica"
                blocking_key = subgroup_blocking(page_title, blocking_key)
                key_list.append(blocking_key)
        if not key_list:
            key_list.append("other")
        key_list=set(key_list)
        key_list = list(key_list)
        df.at[index, 'blocking_key'] = key_list      
    df = process_others(blocking_key, blocking_keys, df)
    print('>>> Blocking computed successfully!\n')
    return df
	
	

def get_block_pairs_df(df):
    """Function used to get a Pandas DataFrame containing pairs of specifications based on the blocking keys

    Creates a Pandas DataFrame where each row is a pair of specifications. It will create one row for every possible
    pair of specifications inside a block.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing specifications, page titles and blocking keys

    Returns:
        pairs_df (pd.DataFrame): A Pandas DataFrame containing pairs of specifications
    """
    print('>>> Creating pairs dataframe...\n')
    df = df.explode('blocking_key')
    grouped_df 	= df.groupby('blocking_key')
    index_pairs = []
    for _, block in grouped_df:
        block_indexes = list(block.index)
        index_pairs.extend(list(itertools.combinations(block_indexes, 2)))

    progressive_id = 0
    progressive_id2row_df = {}
    for index_pair in tqdm(index_pairs):
        left_index, right_index = index_pair
        left_spec_id = df.loc[left_index, 'spec_id']
        right_spec_id = df.loc[right_index, 'spec_id']
        left_spec_title = df.loc[left_index, 'page_title']
        right_spec_title = df.loc[right_index, 'page_title']
        row = (left_spec_id, right_spec_id, left_spec_title, right_spec_title)
        progressive_id2row_df.update({progressive_id: row})
        progressive_id += 1

    columns_df = ['left_spec_id', 'right_spec_id', 'left_spec_title', 'right_spec_title']
    pairs_df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    print(pairs_df)
    print('>>> Pairs dataframe created successfully!\n')
    return pairs_df


def compute_matching(pairs_df):
    """Function used to actually compute the matching specifications

    Iterates over the pairs DataFrame and uses a matching function to decide if they represent the same real-world
    product or not. Two specifications are matching if they share at least 2 tokens in the page title.
    The tokenization is made by simply splitting strings by using blank character as separator.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing pairs of specifications

    Returns:
        matching_pairs_df (pd.DataFrame): The Pandas DataFrame containing the matching pairs
    """

    print('>>> Computing matching...\n')
    columns_df = ['left_spec_id', 'right_spec_id']
    matching_pairs_df = pd.DataFrame(columns=columns_df)
    for index, row in tqdm(pairs_df.iterrows()):
        left_spec_title = row['left_spec_title']
        right_spec_title = row['right_spec_title']
        left_tokens = set(left_spec_title.split())
        right_tokens = set(right_spec_title.split())

        if len(left_tokens.intersection(right_tokens)) >= 2:
            left_spec_id = row['left_spec_id']
            right_spec_id = row['right_spec_id']
            matching_pair_row = pd.Series([left_spec_id, right_spec_id], columns_df)
            matching_pairs_df = matching_pairs_df.append(matching_pair_row, ignore_index=True)
    print(matching_pairs_df.head(5))
    print('>>> Matching computed successfully!\n')
    return matching_pairs_df


"""
    This script will:
    1. create a Pandas DataFrame for the dataset. Note that only the <page title> attribute is considered (for example purposes);
    2. partition the rows of the Pandas DataFrame in different blocks, accordingly with a blocking function;
    3. create a Pandas DataFrame for all the pairs computed inside each block;
    4. create a Pandas DataFrame containing all the matching pairs accordingly with a matching function;
    5. export the Pandas DataFrame containing all the matching pairs in the "outputh_path" folder.
"""
if __name__ == '__main__':
    dataset_path = './dataset'
    outputh_path = './output'

    dataset_df = create_dataframe(dataset_path)
    dataset_df = compute_blocking(dataset_df)
    pairs_df = get_block_pairs_df(dataset_df)
    matching_pairs_df = compute_matching(pairs_df)
    # Save the submission as CSV file in the outputh_path
    matching_pairs_df.to_csv(outputh_path + '/submission.csv', index=False)
    print('>>> Submission file created in {} directory.'.format(outputh_path))
