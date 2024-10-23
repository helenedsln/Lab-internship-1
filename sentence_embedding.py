from german_generation import save_dataframe
from utils import lemmatize
from sentence_transformers import SentenceTransformer
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import fasttext
import fasttext.util

FASTTEXT_MODEL = "cc.de.300.bin"
nltk.download('stopwords')


def get_most_recent_csv(prefix):

  directory = "/home/ubuntu/helene/results"

  # List all files in the directory
  files = os.listdir(directory)
  
  # Filter out only CSV files
  csv_files = [f for f in files if f.endswith('.csv') and f.startswith(prefix)] 
  
  # Get the full path of the files
  full_paths = [os.path.join(directory, f) for f in csv_files]
  
  # Sort files by modification time in descending order
  most_recent_file = max(full_paths, key=os.path.getmtime)
  
  return most_recent_file


def load_df():
  results = pd.read_csv(get_most_recent_csv('german_results'))
  results_df = results.copy()
  
  return results_df


def remove_and_tokenize(results_df):
  #remove narratives with problematic_tokens
  df2 = results_df['result'][~results_df['result'].str.contains('http')]
  df2 = df2[~df2.str.contains('www.')]
  df2 = df2[~df2.str.contains('/n')]
  df2 = df2[~df2.str.contains('UESPWiki')]
  df2 = df2[~df2.str.contains('\n\n')]
  df2 = df2[~df2.str.contains('<|endoftext|>')] 
  df2=df2.reset_index(drop=True)

  df2 = df2.str.replace(r'([a-z])\.([A-Z])', r'\1. \2', regex=True)
  sentences = [sent_tokenize(x) for x in df2]
  sentences1 = sentences

  #Tokenize to words
  tokenizer11 = RegexpTokenizer(r"\w+")
  words = [[tokenizer11.tokenize(x) for x in sent] for sent in sentences1]

  stop_words = stopwords.words('german')
  word_removeStop = [[[x.lower() for x in y if (x.lower() not in stop_words)] for y in z] for z in words] #Remove stop-words
  word_removeStop = [[x for x in y if len(x) > 1] for y in word_removeStop] #Remove words of length 1
  word_lemma=[lemmatize(x) for x in word_removeStop] #Lemmatize words

  sentences2 = [[x for x in y if len(x.split()) > 1] for y in sentences1] #Remove sentences that include single words (indicating a problem in the original tokenization)

  return sentences2, word_lemma



def compute_sentence_distances(embeddingsST):
  #Cosine distances between sentences 
  Sent_dist_AVG = [np.matrix([[sp.spatial.distance.cosine(samp[s1].cpu(), samp[s2].cpu()) if s2 > s1 else np.nan 
                               for s2 in range(samp.shape[0])] 
                               for s1 in range(samp.shape[0])]) 
                               for samp in tqdm(embeddingsST, position=0, leave=True)]

  #Only distances between consecutive sentences
  Sent_AVG_sequential = [[sim[s, s+1] for s in range(sim.shape[0] - 1)] for sim in Sent_dist_AVG]

  #Average distance between consecutive sentences 
  Sent_AVG_sequential_m = [np.nanmean(np.array(sim)[np.array(sim) > 0]) for sim in Sent_AVG_sequential]
  Sent_AVG_sequential_min = [np.nanmin(np.array(sim)[np.array(sim) > 0]) if np.any(np.array(sim) > 0) else np.nan for sim in Sent_AVG_sequential]
  Sent_AVG_sequential_max = [np.nanmax(np.array(sim)[np.array(sim) > 0]) if np.any(np.array(sim) > 0) else np.nan for sim in Sent_AVG_sequential]
  Sent_AVG_sequential_std = [np.nanstd(np.array(sim)[np.array(sim) > 0]) for sim in Sent_AVG_sequential]

  #Average distance between all sentences
  Sent_AVG_all_m = [np.nanmean(np.array(sim)[np.array(sim) > 0]) for sim in Sent_dist_AVG]

  sentence_embedding_df_init = pd.DataFrame()
  sentence_embedding_df_init["sent_ST_seq_dist"] = [np.nanmean(x) for x in Sent_AVG_sequential_m]
  sentence_embedding_df_init["sent_ST_seq_dist_min"] = [np.nanmean(x) for x in Sent_AVG_sequential_min]
  sentence_embedding_df_init["sent_ST_seq_dist_max"] = [np.nanmean(x) for x in Sent_AVG_sequential_max]
  sentence_embedding_df_init["sent_ST_seq_dist_std"] = [np.nanmean(x) for x in Sent_AVG_sequential_std]
  sentence_embedding_df_init["sent_ST_all_dist"] = [np.nanmean(x) for x in Sent_AVG_all_m]

  return sentence_embedding_df_init


def sentence_embedding(results_df) :
  
  sentences2, word_lemma=remove_and_tokenize(results_df)
  
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  embeddingsST = [model.encode(x, convert_to_tensor=True) for x in tqdm(sentences2, position=0, leave=True)]

  sentence_embedding_df_init = compute_sentence_distances(embeddingsST)

  results_df2=results_df[['repetition', 'prompt', 'sampling_method', 'p', 'temperature', 'backward_attention_length']]
  sentence_embedding_df=results_df2.merge(sentence_embedding_df_init, left_index=True, right_index=True, how='left')

  save_dataframe(sentence_embedding_df, 'sentence_embedding')

  return sentence_embedding_df



def main_embedding():
 
  final= load_df()
  sentence_embedding_df= sentence_embedding(final)

  return sentence_embedding_df


if __name__ == "__main__":
    
    main_embedding()


