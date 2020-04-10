import warnings
import sys
from distutils.util import strtobool
import preprocessing as preprocessor
import pandas as pd
import EntityResolutionEngine as EntityResolutionEngine 

warnings.filterwarnings('ignore')

"""
    This script will:
       1-create a dataframe 
       2- first blocking by brand names
       3 - extracting model names for each brand group
       3- then blocking by model names
       4- finding pairs by comparing the page title and some features of the products for each block
       5- generating output file
"""
if __name__ == '__main__':
       
    if len(sys.argv)>=6:
        dataset_path=sys.argv[1]
        labelled_path=sys.argv[2]
        output_file_name=sys.argv[3]
        use_labelled_set=strtobool(sys.argv[4])
        similarity_threshold=int(sys.argv[5])

        dataset_df = preprocessor.create_brand_dataframe(dataset_path)
        dataset_df = dataset_df.set_index('spec_id')
        labelled_df = pd.read_csv(labelled_path)

        dataset_df = preprocessor.compute_brand_blocking(dataset_df)

        entity_resolution_engine = EntityResolutionEngine.EntityResolutionEngine(dataset_df,labelled_df)
        entity_resolution_engine.get_model_names()
        entity_resolution_engine.compute_model_name_blocking()

        entity_resolution_engine.get_pairs(threshold=similarity_threshold,labelled_set=use_labelled_set)

        if use_labelled_set:
            precision,recall,f_measure,incorrect_pairs = entity_resolution_engine.calculate_f_measure()
            print('Precision:', precision,'Recall:', recall,'F-measure: ',f_measure)
        
        entity_resolution_engine.output_df[['left_spec_id','right_spec_id']].to_csv(output_file_name+'.csv',index=False)
    
    else:
        print('You have entered',len(sys.argv), 'command line arguments')
        print('Correct order of args: dataset_path labelled_dataset_path output_file_path_with_name(will be save as path+.csv) similarity_threshold use_labelled_set(boolean True or False)')
        