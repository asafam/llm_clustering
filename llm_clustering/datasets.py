from datasets import load_dataset
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from enum import Enum


class DatasetName(Enum):
    CLINC = "CLINC"
    BANKING77 = "BANKING77"
    
    
class EmbeddingModelName(Enum):
    all_MiniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    intfloat__e5_large_v2 = "intfloat/e5-large-v2"
    Alibaba_NLP__gte_large_en_v1_5 = "Alibaba-NLP/gte-large-en-v1.5"
    McGill_NLP__LLM2Vec_Mistral_7B_Instruct_v2_mntp = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
    McGill_NLP__LLM2Vec_Meta_Llama_3_8B_Instruct_mntp = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"


def load_dataset_by_name(dataset_name):
    dataset = None
    if dataset_name == 'CLINC':
        dataset = load_dataset('clinc_oos', 'small')
    elif dataset_name == 'BANKING77':
        dataset = load_dataset('banking77')
    return dataset['test']


def get_label_column(dataset_name):
    label_column = None
    if dataset_name == 'CLINC':
        label_column = 'intent'
    elif dataset_name == 'BANKING77':
        label_column = 'label'
    return label_column


def sample_text(
    df: pd.DataFrame, 
    label_column: str, 
    k: int = 0, 
    min_cluster_size: int = 0, 
    total_sample_size: int = 0, 
    seed: int = 42
) -> pd.DataFrame:
    if k * min_cluster_size > total_sample_size:
        raise ValueError(f"The requested sample with {k} clusters and a min cluster size of {min_cluster_size} is greated than the target sample size of {total_sample_size}")
    
    if total_sample_size == 0:
        result_df = df.sample(frac=1, random_state=seed)
        return result_df
        
    unique_labels = df[label_column].unique()
    
    random.seed(seed)
    selected_labels = random.sample(list(unique_labels), k) if k > 0 else list(unique_labels)

    # Initialize list to hold sampled documents
    sampled_documents = []

    # Sample min_cluster_size documents from each selected intent class
    for label in selected_labels:
        label_df = df[df[label_column] == label]
        if min_cluster_size > 0:
            sampled_label_df = label_df.sample(n=min_cluster_size, random_state=seed)
        else:
            sampled_label_df = label_df.sample(frac=1, random_state=seed)
        sampled_documents.append(sampled_label_df)

    # Concatenate all sampled documents into a single DataFrame
    result_df = pd.concat(sampled_documents)
    
    if len(result_df) >= total_sample_size:
        return result_df

    # Calculate the number of additional samples needed to reach total_sample_size documents
    additional_samples_needed = total_sample_size - len(result_df)

    # Sample the remaining documents from the combined DataFrame
    remaining_df = df[~df.index.isin(result_df.index)]
    additional_samples = remaining_df.sample(n=additional_samples_needed, random_state=seed)

    # Concatenate the additional samples to the result DataFrame
    result_df = pd.concat([result_df, additional_samples])
    return result_df


class TextClusterDataset(Dataset):
    def __init__(self, texts, clusters):
        self.texts = texts
        self.clusters = clusters

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.clusters[idx]