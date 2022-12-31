import re 
import nltk 
import spacy 
import numpy as np 
import pandas as pd 
import seaborn as sns
from tqdm import tqdm  
from typing import List 
from tqdm.notebook import tqdm 
from transformers import AutoTokenizer
from nltk.stem import WordNetLemmatizer


import tensorflow as tf 
from tensorflow.keras.layers import Layer #type: ignore 

class TensorFlowExperimentUtils:
    def __init__(self):
        tqdm.pandas()
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        lemm = WordNetLemmatizer()
        sns.set_style("darkgrid")
        #init_notebook_mode(connected=True)
        spacy_eng = spacy.load("en_core_web_sm")

        self.model_checkpoint = 'bert-base-uncased'
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.model_checkpoint)

    def load_dfs(self, names : List):
        dfs = []
        for name in names:
            df = pd.read_csv(name)
            df.name = name 
            dfs.append(df)
        return dfs

    def train_valid_split(self, df, test_split=0.2):
        train_length = int(len(df) * (1 - test_split))
        train_data = pd.DataFrame(df.iloc[:train_length, :])
        valid_data = pd.DataFrame(df.iloc[train_length:, :])
        return (train_data, valid_data)


    def text_cleaning(self, x):
        questions = re.sub('\s+\n+', ' ', x)
        questions = re.sub('[^a-zA-Z0-9]', ' ', questions)
        questions = questions.lower()
        return questions


    def to_string(self, x): return str(x)

    def text_clean_df_cols(self, dfs : List, cols : List):
        for df in dfs:
            print(f"=> For df: {str(df.name)}")
            df = pd.DataFrame(df)
            for col in cols:
                try:
                    df[col] = df[col].progress_apply(self.to_string)
                    df[col] = df[col].progress_apply(self.text_cleaning)
                except:
                    continue 
        print("Finished")
    

    def encode_text_as_one(self, texts, reasons, tokenizer):

        sentences = []
        for text, reason in zip(texts, reasons):
            sentences.append(
                '[CLS]' + text + '[SEP]' + reason + '[SEP]'
            )

        encoded = tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True, 
            max_length=50, 
            truncation=True, 
            padding='max_length', 
            return_attention_mask=True,
            return_tensors="tf",
        )

        return {
            'input_ids' : np.array(encoded["input_ids"], dtype="int32"), 
            'attention_mask' : np.array(encoded["attention_mask"], dtype="int32")
        }

    def encode_text(self, text, tokenizer):
        encoded = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf",
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks
        }


    def get_encoded_items(self, df, approach='single'):
        '''
        Options for approach: single/pair 
        '''
        return [
            self.encode_text(df['text'].tolist(), self.TOKENIZER), 
            self.encode_text(df['reason'].tolist(), self.TOKENIZER),
            np.array(df['label'].tolist())
        ] if approach == 'pair' else [
            self.encode_text_as_one(
                df['text'].tolist(),
                df['reason'].tolist(),
                self.TOKENIZER
            ), 
            np.array(df['label'].tolist())
        ]



# Different L1 and L2 Distance layers (non parametric)

class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
        
    def call(self,embedding1,embedding2):
        return tf.math.abs(embedding1 - embedding2)

class L2Dist(Layer):
    def __init__(self,**kwargs):
        super(L2Dist, self).__init__()
    
    def call(self, embedding1, embedding2):
        return tf.math.rms(embedding1 - embedding2)