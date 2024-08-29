from typing import *
from datasets import load_dataset
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from enum import Enum
import math
import logging


class DatasetName(Enum):
    CLINC = "CLINC"
    BANKING77 = "BANKING77"


class TextLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_dataset_by_name(dataset_name: DatasetName, subset: str = 'test') -> TextLabelDataset:
    dataset = None
    if dataset_name == DatasetName.CLINC:
        dataset = load_dataset('clinc_oos', 'small')
    elif dataset_name == DatasetName.BANKING77:
        dataset = load_dataset('banking77')
    
    text_column = get_text_column(dataset_name=dataset_name)
    label_column = get_label_column(dataset_name=dataset_name)

    texts = dataset[subset][text_column]
    labels = dataset[subset][label_column]

    text_label_dataset = TextLabelDataset(texts, labels)
    return text_label_dataset


def get_dataset_from_df(df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label') -> TextLabelDataset:
    texts =df[text_column]
    labels = df[label_column]

    text_label_dataset = TextLabelDataset(texts, labels)
    return text_label_dataset


def get_text_column(dataset_name: DatasetName):
    label_column = None
    if dataset_name == dataset_name.CLINC:
        label_column = 'text'
    elif dataset_name == dataset_name.BANKING77:
        label_column = 'text'
    return label_column


def get_label_column(dataset_name: DatasetName):
    label_column = None
    if dataset_name == dataset_name.CLINC:
        label_column = 'intent'
    elif dataset_name == dataset_name.BANKING77:
        label_column = 'label'
    return label_column


def sample_dataset(
    dataset: TextLabelDataset,
    n: int = 0, 
    k: Optional[int] = None, 
    min_cluster_size: int = 1, 
    random_state: int = 42
) -> pd.DataFrame:
    result_df = pd.DataFrame()
    df = pd.DataFrame({
        'text': [item[0] for item in dataset],
        'label': [item[1] for item in dataset]
    })

    # Seed the random generator to repeat results
    random.seed(random_state)
    
    # Returned a shuffled copy of the data in case n = 0
    if n == 0:
        result_df = df.sample(frac=1, random_state=random_state)
        return result_df

    # sample labels
    unique_labels = df['label'].unique()

    if k is None:
        selected_labels = []
    elif k > 0:
        selected_labels = random.sample(list(unique_labels), k)
    elif k == 0:
        selected_labels = list(unique_labels)
    else:
        selected_labels = []

    k = len(selected_labels)

    # Sample min_cluster_size documents from each selected intent class
    if k > 0:
        # Initialize list to hold sampled documents
        sampled_documents = []

        min_cluster_size = min(min_cluster_size, math.floor(n / k))
        for label in selected_labels:
            label_df = df[df['label'] == label]
            if min_cluster_size > 0:
                sampled_label_df = label_df.sample(n=min(min_cluster_size, len(label_df)), random_state=random_state)
            else:
                sampled_label_df = pd.DataFrame()
            sampled_documents.append(sampled_label_df)

        # Concatenate all sampled documents into a single DataFrame
        result_df = pd.concat(sampled_documents)

    if len(result_df) >= n:
        return result_df

    # Calculate the number of additional samples needed to reach total_sample_size documents
    additional_samples_needed = n - len(result_df)

    # Sample the remaining documents from the combined DataFrame
    remaining_df = df[~df.index.isin(result_df.index)]
    additional_samples = remaining_df.sample(n=additional_samples_needed, random_state=random_state)

    # Concatenate the additional samples to the result DataFrame
    result_df = pd.concat([result_df, additional_samples])
    return result_df

