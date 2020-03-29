import os
import json
import pandas as pd
import itertools
from tqdm import tqdm


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
                row = (source, specification_number, specification_id, page_title, brand)
                progressive_id2row_df.update({progressive_id: row})
                progressive_id += 1
    df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    print(df)
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
    mySet.add("sanyo")
    print(mySet)
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
                if(blocking_key =="fuji"):
                    blocking_key = "fujifilm"
                key_list.append(blocking_key)
        if not key_list:
            key_list.append("other")
        df.at[index, 'blocking_key'] = key_list 
        
    print(df)
   # df.loc[df['blocking_key'] == [], 'blocking_key'] = 'other'
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
