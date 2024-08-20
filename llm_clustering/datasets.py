from datasets import load_dataset
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from enum import Enum
import math


class DatasetName(Enum):
    CLINC = "CLINC"
    BANKING77 = "BANKING77"


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
    n: int = 0, 
    min_cluster_size: int = 0, 
    seed: int = 42
) -> pd.DataFrame:
    result_df = pd.DataFrame()

    # Seed the random generator to repeat results
    random.seed(seed)
    
    # Returned a shuffled copy of the data in case n = 0
    if n == 0:
        result_df = df.sample(frac=1, random_state=seed)
        return result_df

    # sample labels
    if min_cluster_size > 0:
        unique_labels = df[label_column].unique()
        selected_labels = random.sample(list(unique_labels), k) if k > 0 else []
        k = len(selected_labels)

        # Sample min_cluster_size documents from each selected intent class
        if k > 0:
            # Initialize list to hold sampled documents
            sampled_documents = []
            
            min_cluster_size = min(min_cluster_size, math.floor(n / k))
            for label in selected_labels:
                label_df = df[df[label_column] == label]
                if min_cluster_size > 0:
                    sampled_label_df = label_df.sample(n=min(min_cluster_size, len(label_df)), random_state=seed)
                else:
                    sampled_label_df = df.DataFrame()
                sampled_documents.append(sampled_label_df)

            # Concatenate all sampled documents into a single DataFrame
            result_df = pd.concat(sampled_documents)
    
    if len(result_df) >= n:
        return result_df

    # Calculate the number of additional samples needed to reach total_sample_size documents
    additional_samples_needed = n - len(result_df)

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