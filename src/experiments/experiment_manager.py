from typing import *
from datetime import datetime
import inspect
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data import DatasetName, load_dataset_by_name, sample_dataset
from clustering.constraints_manager import ConstraintsType, KInformationType
from clustering.models import ClusteringModel
from clustering.optimizations import KOptimization
from clustering.utils import evaluate_clustering
from embedding.models import TextEmbeddingModel
from llms.models import LLM
from experiments.constrained_models import BaseConstrainedLLM


def run_experiments(
        dataset_name: DatasetName,
        sample_n: int, 
        llm: LLM,
        constraint_type: ConstraintsType,
        text_embedding_model: TextEmbeddingModel,
        clustering_model: ClusteringModel,
        oracle_k_information_type: KInformationType = KInformationType.UnknownK,
        k_optimization: Optional[KOptimization] = None,
        cluster_k_information_type: KInformationType = KInformationType.UnknownK, 
        max_clusters: Optional[int] = 10,
        batch_size: int = 128,
        random_state: int = 42
    ):
    # get data
    dataset = load_dataset_by_name(dataset_name=dataset_name)

    # sample subset
    k = 0 if oracle_k_information_type == KInformationType.GroundTruthK else None
    sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, random_state=random_state)

    # run the LLM predictions to create the constraints
    sample_texts = sample_df['text'].tolist()
    constraint_model = BaseConstrainedLLM(llm=llm, constraint_type=constraint_type)
    constraint = constraint_model.create_constraint(texts=sample_texts, k=k)

    # embed the dataset for clustering
    all_embeddings = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_texts, batch_labels in dataloader:
        batch_embeddings = text_embedding_model.embed(batch_texts)
        all_embeddings.append(batch_embeddings)
        all_labels.extend(batch_labels)
    X = np.vstack(all_embeddings)
    labels_true = [tensor.item() for tensor in all_labels]
    
    # cluster the dataset using the constraint
    if cluster_k_information_type == KInformationType.UnknownK:
        # if number of clusters is unknown then optimize it
        labels_pred, best_k = clustering_model.cluster(X, constraints=constraint, k_optimization=k_optimization, max_k=max_clusters, random_state=random_state)
    elif cluster_k_information_type == KInformationType.GroundTruthK:
        # otherwise, provide the true cluster number to the clustering model
        n_clusters = len(set(labels_true))
        labels_pred = clustering_model.cluster(X, constraint, n_clusters=n_clusters, random_state=random_state)
    elif cluster_k_information_type == KInformationType.OracleK:
        # or else, provide the predicted cluster number (if we can extract it from the constraint) to the clustering model
        if constraint.labels is None:
            logger.error(f"No labels were correctly predicted running with constraint {constraint_type.value}")
        else:
            labels_oracle_pred = constraint.labels
            n_clusters = len(set(labels_oracle_pred))
            labels_pred = clustering_model.cluster(X, constraint, n_clusters=n_clusters, random_state=random_state)

    # compute score for the clustering
    scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true, X=X)

    results = dict(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        labels_true=labels_true,
        labels_pred=labels_pred,
    )
    results.update(scores)

    # get the arguments of the current execution
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arguments = {}
    for arg in args:
        arguments[arg] = values[arg]

    results.update(arguments)

    return results