# SIGMOD20 Programming Contest Write-up Team Seguon

## Introduction
This year's SIGMOD Programming Contest subject is solving the Entity Resolution problem of finding duplicate real-world objects in given dataset. 
Challenge here is to develop an Entity Resolution System to to identify and match which product spesification from multiple represenets the same real-world camera or camera accessories objects.

## Dataset
Test dataset consists of almost 30k camera and camera accessory spesifications in JSON format from multiple e-commerce websites. All spesifications include <Page title> attribute. 
Other than that, dataset does not have a predefined structure and the information does not necessarily obey grammatical rules.
There are also two validation set that consist of 6k rows of matching and unmatching objects with labels 1 and 0 respectively. 


## Methodology

### Data Preprocessing

#### Extracting Useful Information
Before checking for similarities, it was needed to decide which data attributes to use since there is no structure in the provided data. <Page title> is used since it is present in all data. In addition, we also tried to extract megapixel, brand name and model name
attributes if they exist and added them to the page title in order to increase the similarity of possibly matching objects. 

#### Cleaning
As first step all of the spesifications are converted to lowercase letters. Then all the stopwords in English are eliminated. Additionally, unnecessary punctuations such as "-" and "/"  are deleted. 
Some words such as "buy", "purchase", "promotion" that decrease the similarity but does not have any effect to identify an object are also eliminated after manual inspection of data.

#### Blocking
We have used a blocking strategy that consists of two different phases. 

In first phase, all the data is blocked using camera and accessory brands as blocking keys. Since most of the data spesification does not have an explicit brand name attribute, a list of brand names is generated
the by using the first word of the page title in some of the e-commerce website data. After the list is generated, brand names are searched in the <page title> of all dataset.
Almost all the entities were assigned a blocking key while the rest were marked as "others" where it was not possible to identify the brand. It was also possible that some entities were assigned to multiptle blocks.

In the second phase of blocking, we applied another blocking inside the same blocks. Our aim was to categorize each block with respect to their model_name.
First, the location of the blocking_key in the page title is found. After finding the exact location, we include the first consecutive word inside our model_name. Then, this model_name is searched inside the page title.
If model_name is shared among some entities this model_name represents the subblock of a given block. This process is applied 4 times for 4 consecutive words to define the model of a brand and block them respectively.
   
### Similarity Checking
We have tried multiple methods for similarity checking such as Cosine, Jaccard, Levenshtein and Dice. All similarity checks are done by cleaning the object data, extracting the model words and generating its trigrams. 
*** According to our experiments on validation set, cosine and Levensthein had similar results but latter was slightly more effective. As a result, Levensthein similarity function of fuzzywuzzy library is utilized to check similarities of objects
after blocking. Thresold for similarity is chosen as 10 / 100. A relatively low threshold is chosen since using high threshold values resulted in increasing of precision but decreasing of recall.

### Pairing and Output
The objects that were in the same blocks and found similar are paired. As a result, our output consisted of pairs of objects that are categorized as duplicates by our system.
We have represeneted the matching objects of our output and labelled dataset in a graph where objects are vertices and matching condition is an edge.

### Validation
After generating an output, the output is compared to the labelled dataset.  
Output graph is converted into a matrix to check the precision, recall and F-measure of our Entity Resolution system.


## Results
Current results on secret dataset: 
- Precision: 0.83
- Recall: 0.84
- F-Measure: 0.84

Current results on labelled dataset:
- Precision: 0.96
- Recall: 0.89
- F-Measure: 0.92

## Prerequisites

- Python 3.*
- Pip
- pip install -r requirements.txt

## Running

Run the project:
```
$ python main.py [DATASET PATH] [LABELLED DATASET PATH] [OUTPUT DATASET PATH]  [True:Test on labelled dataset, returns F-Measure | False: Generates output using whole dataset ] [similarity threshold 0-100]

- The following command can be used to test results on labelled dataset:

$ python main.py ../datasets/2013_camera_specs/ ../datasets/sigmod_large_labelled_dataset.csv ../datasets/output  True 10
```

This command will produce a CSV file (the submission) in the output directory ("outputh_path") and will print intermediate results in the shell.
