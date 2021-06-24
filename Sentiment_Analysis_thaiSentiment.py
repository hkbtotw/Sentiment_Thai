## ref: https://colab.research.google.com/drive/1Kbk6sBspZLwcnOE61adAQo30xxqOQ9ko?usp=sharing#scrollTo=_oq4vnrz9aa8     (wangchanBerta)
## ref: https://github.com/cstorm125/thai_sentiment  (thai_sentiment)

import numpy as np
from tqdm.auto import tqdm
import torch

#datasets
from datasets import load_dataset

#transformers
from transformers import (
    CamembertTokenizer,
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

#thai2transformers
import thai2transformers
from thai2transformers.preprocess import process_transformers
from thai2transformers.metrics import (
    classification_metrics, 
    multilabel_classification_metrics,
)
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
    SEFR_SPLIT_TOKEN
)

from thai_sentiment import get_sentiment
import pandas as pd

################################################################################

def GetSentimentDf(x):
    return get_sentiment(x)


def GetSentimentDf_wangchanBerta(x):
    input_text = process_transformers(x)
    #infer
    return (classify_multiclass(input_text))


###############################################################################
#################### Wangchanmodel Initialization

model_names = [
    'wangchanberta-base-att-spm-uncased',
    'xlm-roberta-base',
    'bert-base-multilingual-cased',
    'wangchanberta-base-wiki-newmm',
    'wangchanberta-base-wiki-ssg',
    'wangchanberta-base-wiki-sefr',
    'wangchanberta-base-wiki-spm',
]

tokenizers = {
    'wangchanberta-base-att-spm-uncased': AutoTokenizer,
    'xlm-roberta-base': AutoTokenizer,
    'bert-base-multilingual-cased': AutoTokenizer,
    'wangchanberta-base-wiki-newmm': ThaiWordsNewmmTokenizer,
    'wangchanberta-base-wiki-ssg': ThaiWordsSyllableTokenizer,
    'wangchanberta-base-wiki-sefr': FakeSefrCutTokenizer,
    'wangchanberta-base-wiki-spm': ThaiRobertaTokenizer,
}
public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased'] 
#@title Choose Pretrained Model
model_name = "wangchanberta-base-att-spm-uncased" #@param ["wangchanberta-base-att-spm-uncased", "xlm-roberta-base", "bert-base-multilingual-cased", "wangchanberta-base-wiki-newmm", "wangchanberta-base-wiki-syllable", "wangchanberta-base-wiki-sefr", "wangchanberta-base-wiki-spm"]

#create tokenizer
tokenizer = tokenizers[model_name].from_pretrained(
                f'airesearch/{model_name}' if model_name not in public_models else f'{model_name}',
                revision='main',
                model_max_length=416,)


#@title Choose Multi-class Classification Dataset
dataset_name = "wisesight_sentiment" #@param ['wisesight_sentiment','wongnai_reviews']
#pipeline
classify_multiclass = pipeline(task='sentiment-analysis',
         tokenizer=tokenizer,
         model = f'airesearch/{model_name}' if model_name not in public_models else f'{model_name}',
         revision = f'finetuned@{dataset_name}')


###############################################################################
file_path=r'C:\\Users\\70018928\Documents\\Project2021\\Experiment\\NLP\\sentiment\\'
file_name='TBPoint_Transaction_TC.xlsx'
spreadsheet_sheet='Transaction_NLP_no1'
###############################################################################

##### read input
df1=pd.read_excel(file_path+file_name,  sheet_name=spreadsheet_sheet, engine='openpyxl')


#### Select only identifier and textinput columns   
df2_Transaction = df1[['Id','OtherReason']].copy()
############################################################################################


df2_Transaction['sentiment_thai']=df2_Transaction.apply(lambda x:GetSentimentDf(x['OtherReason']),axis=1)
df2_Transaction['sentiment_wangchan']=df2_Transaction.apply(lambda x:GetSentimentDf_wangchanBerta(x['OtherReason']),axis=1)
print(' sentiment : ',df2_Transaction.head(10))
df2_Transaction.to_excel(file_path+'sentiment.xlsx')

