import os
import warnings 
import re 
import nltk
import spacy 
import numpy as np  
import pandas as pd
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt

from typing import List, Dict 
from tqdm.notebook import tqdm 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from plotly.offline import init_notebook_mode
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import tensorflow as tf
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          TFAutoModel,DistilBertConfig,
                          TFDistilBertModel, 
                          BertConfig, TFBertModel, TFRobertaModel)
from tensorflow.keras.layers import (Embedding, 
                                     Layer, 
                                     Dense, 
                                     Dropout, 
                                     MultiHeadAttention, 
                                     LayerNormalization, 
                                     Input, GlobalAveragePooling1D)
from tensorflow.keras.models import Sequential, Model


class TensorFlowExperimentUtils:
    def __init__(self):
        tqdm.pandas()
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        lemm = WordNetLemmatizer()
        sns.set_style("darkgrid")
        init_notebook_mode(connected=True)
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
            "attention_masks": attention_masks
        }


    def get_encoded_items(self, df):
        return [
            self.encode_text(df['text'].tolist(), self.TOKENIZER), 
            self.encode_text(df['reason'].tolist(), self.TOKENIZER),
            df['label'].tolist()
        ]


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


class TensorFlowExperiment(TensorFlowExperimentUtils):
    def __init__(self, exp_name, model_type, loss):
        super(TensorFlowExperimentUtils, self).__init__()
        self.exp_name = exp_name
        self.model_type = model_type
        self.loss = loss
        self.strategy = tf.distribute.get_strategy()

    def create_model_bert_classification(self):
        with self.strategy.scope():
            transformer_model = TFBertModel(self.model_checkpoint) 
            input_ids1 = Input(shape=(None, ), name='input_ids1', dtype='int32')
            input_ids2 = Input(shape=(None, ), name='input_ids2', dtype='int32')
            input_mask1 = Input(shape=(None, ), name='attention_mask1', dtype='int32')
            input_mask2 = Input(shape=(None, ), name='attention_mask2', dtype='int32')

            embedding1 = transformer_model(
                input_ids1, attention_mask=input_mask1
            ).last_hidden_state

            embedding2 = transformer_model(
                input_ids2, attention_mask=input_mask2
            ).last_hidden_state

            

def TensorFlowExperiment(train_df, test_df, batch_size=16):
    """
    This function is used to train a TensorFlow model.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        test_df (pd.DataFrame): The testing dataframe.

    Returns:
        None
    """

    text_clean_df_cols(
        [train_df, test_df], 
        ['text', 'reason']
    )

    strategy = tf.distribute.get_strategy()
    BATCH_SIZE=batch_size

