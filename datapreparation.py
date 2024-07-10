# Import libraries
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from scipy.linalg import triu
from numpy.linalg import norm
#from termcolor import colored
import pandas as pd
import numpy as np
#import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk

nltk.download('punkt')

# Load data
def prepareData():
    df = pd.read_csv('./dataset/nyc-jobs-1.csv')
# Check data
#df.head()

# Show column name
#print(df.columns)

    df =df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]
#df.head()

# Create a new column called 'data' and merge the values of the other columns into it
    df['data'] = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
# Drop the individual columns if you no longer need them
    df.drop(['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills'], axis=1, inplace=True)
# Preview the updated dataframe
    print(df.head())

# Tag data
    data = list(df['data'])
    tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(data)]
    return tagged_data

