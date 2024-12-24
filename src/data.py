from typing import *
from datasets import load_dataset
import pandas as pd
import random
from torch.utils.data import Dataset
from enum import Enum
import math
from sklearn.preprocessing import LabelEncoder
import logging


class DatasetName(Enum):
    CLINC = "CLINC"
    CLINC_TOY = "CLINC_TOY"
    BANKING77 = "BANKING77"
    TOPV2 = "TOPV2"
    MASSIVE = "MASSIVE"
    MTOP_D = "MTOP_DOMAIN"
    MTOP_I = "MTOP_INTENT"
    AGNEWS = "AGNEWS"
    REUTERS21578 = "REUTERS21578"
    MEDRXIV = "MEDRXIV"
    BIORXIV = "BIORXIV"


class TextLabelDataset(Dataset):
    def __init__(self, texts, labels, label_names: Optional[list] = None, ids: Optional[list] = None, shuffle: bool = True, **kwargs):
        ids = ids if ids is not None else list(range(len(texts)))

        if shuffle:
            joined = list(zip(ids, texts, labels, label_names))
            random.seed(42)
            random.shuffle(joined)
            # Unpack the shuffled pairs back into labels and texts
            ids, texts, labels, label_names = zip(*joined)
            # Convert back to lists
            ids = list(ids)
            texts = list(texts)
            labels = list(labels)
            label_names = list(label_names)
            
        self.ids = list(range(len(ids)))
        self.orig_ids = ids
        self.texts = texts
        self.labels = labels
        self.label_names = label_names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx], self.labels[idx], self.label_names[idx]
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{len(self.texts)}"


def load_dataset_by_name(dataset_name: DatasetName, split: str = 'test', **kwargs) -> TextLabelDataset:
    dataset = None
    if dataset_name == DatasetName.CLINC:
        dataset = load_dataset('clinc_oos', 'small')
    elif dataset_name == DatasetName.BANKING77:
        dataset = load_dataset('mteb/banking77')
    elif dataset_name == DatasetName.AGNEWS:
        dataset = load_dataset('fancyzhx/ag_news')
    elif dataset_name == DatasetName.MASSIVE:
        dataset = load_dataset('mteb/amazon_massive_intent', "en", trust_remote_code=True)
    elif dataset_name == DatasetName.MTOP_D:
        dataset = load_dataset('mteb/mtop_domain', "en", trust_remote_code=True)
    elif dataset_name == DatasetName.MTOP_I:
        dataset = load_dataset('mteb/mtop_intent', "en", trust_remote_code=True)
    elif dataset_name == DatasetName.TOPV2:
        dataset = load_dataset('WillHeld/top_v2')
    elif dataset_name == DatasetName.MEDRXIV:
        dataset = load_dataset('mteb/medrxiv-clustering-s2s')
    elif dataset_name == DatasetName.BIORXIV:
        dataset = load_dataset('mteb/biorxiv-clustering-s2s')
    elif dataset_name == DatasetName.REUTERS21578:
        dataset = load_dataset('yangwang825/reuters-21578')
    else:
        raise ValueError(f"No supported dataset {dataset_name}")
    
    text_column = get_text_column(dataset_name=dataset_name)
    label_column = get_label_column(dataset_name=dataset_name)
    label_name_column = get_label_name_column(dataset_name=dataset_name)

    flattened_labels = [(label[0] if type(label) == list else label) for label in dataset[split][label_column]]
    texts = [s for s, label in zip(dataset[split][text_column], flattened_labels) if (dataset_name != DatasetName.CLINC or label != 42)]
    labels = [label for label in dataset[split][label_column] if (dataset_name != DatasetName.CLINC or label != 42)]
    if dataset_name != DatasetName.CLINC:
        label_names = [label for label in dataset[split][label_name_column]]
    else:
        intent_mapping = dataset[split].features[label_name_column]
        id_to_name = intent_mapping.names
        label_names = [id_to_name[label] for label in labels]

    if type(labels[0] == str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    text_label_dataset = TextLabelDataset(texts, labels, label_names=label_names, **kwargs)
    return text_label_dataset


def get_dataset_from_df(
        df: pd.DataFrame, 
        id_column: str = 'id', 
        text_column: str = 'text', 
        label_column: str = 'label', 
        label_name_column: str = 'name',
        **kwargs
    ) -> TextLabelDataset:
    ids = df[id_column].tolist() if 'id' in df else None
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    label_names = df[label_name_column].tolist()

    text_label_dataset = TextLabelDataset(texts, labels, ids, label_names, **kwargs)
    return text_label_dataset


def get_text_column(dataset_name: DatasetName):
    label_column = 'text'
    if dataset_name == DatasetName.TOPV2:
        label_column = 'utterance'
    elif dataset_name in [DatasetName.MEDRXIV, DatasetName.BIORXIV]:
        label_column = 'sentences'
    return label_column


def get_label_column(dataset_name: DatasetName):
    label_column = 'label'
    if dataset_name == DatasetName.CLINC:
        label_column = 'intent'
    elif dataset_name == DatasetName.TOPV2:
        label_column = 'domain'
    elif dataset_name in [DatasetName.MEDRXIV, DatasetName.BIORXIV]:
        label_column = 'labels'
    return label_column

def get_label_name_column(dataset_name: DatasetName):
    if dataset_name in [DatasetName.BANKING77, DatasetName.MTOP_D, DatasetName.MTOP_I]:
        label_name_column = 'label_text'
    elif dataset_name == DatasetName.TOPV2:
        label_name_column = 'domain'
    else:
        label_name_column = get_label_column(dataset_name=dataset_name)
    return label_name_column


def sample_dataset(
    dataset: TextLabelDataset,
    n: int = 0, 
    k: Optional[int|float] = None, 
    min_cluster_size: int = 1, 
    exclude_ids: list = [],
    random_state: int = 42
) -> pd.DataFrame:
    result_df = pd.DataFrame()
    df = pd.DataFrame({
        'id': [item[0] for item in dataset],
        'text': [item[1] for item in dataset],
        'label': [item[2] for item in dataset],
    })

    # Seed the random generator to repeat results
    random.seed(random_state)

    # Apply exclude filters
    df = df[~df['id'].isin(exclude_ids)]

    # sample labels
    unique_labels = df['label'].unique()

    if k is None:
        selected_labels = list(unique_labels)
    elif 0 < k < 1:
        selected_labels = random.sample(list(unique_labels), math.ceil(k * len(unique_labels)))
    elif k > 0:
        selected_labels = random.sample(list(unique_labels), k)
    else:
        selected_labels = list(unique_labels)

    df = df[df['label'].isin(selected_labels)]

    # Returned a shuffled copy of the data in case n = 0
    if n == 0:
        result_df = df.sample(frac=1, random_state=random_state)
        return result_df

    k = len(selected_labels)

    # Sample min_cluster_size documents from each selected intent class
    min_cluster_size = min(min_cluster_size, math.floor(n / k))
    if min_cluster_size > 0:
        # Initialize list to hold sampled documents
        sampled_documents = []

        for label in selected_labels:
            label_df = df[df['label'] == label]
            sampled_label_df = label_df.sample(n=min(min_cluster_size, len(label_df)), random_state=random_state)
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
    result_df = result_df.sample(frac=1, random_state=random_state)
    return result_df

