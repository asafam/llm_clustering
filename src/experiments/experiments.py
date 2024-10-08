from typing import *
from datetime import datetime
import inspect
import traceback
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data import DatasetName, TextLabelDataset, load_dataset_by_name, sample_dataset, get_dataset_from_df
from clustering.constraints_manager import ConstraintsType, KInformationType
from clustering.models.base_models import ClusteringModel
from clustering.optimizations import KOptimization
from clustering.utils import evaluate_clustering
from embedding.models import TextEmbeddingModel
from llms.models import LLM
from llms.utils import PromptType, generate_prompt, get_formatter
from experiments.constrained_models import BaseConstrainedLLM
from experiments.utils import get_experiment_results_item_value
import logging


class BaseExperiment:
    def run(self):
        raise NotImplementedError()
    
    def run_safe(self, **kwargs):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # get the arguments of the current execution
        arguments = {}
        for key, value in kwargs.items():
            if key != 'self':
                arguments[key] = get_experiment_results_item_value(value)
        logger.info(f"Running experiment with arguments: {arguments}")
    
        # run the experiment
        try:
            results = self.run(**kwargs)
            results['success'] = True
            logger.debug(f"Experiment status: success")
        except Exception as e:
            results = dict()
            results['success'] = False
            results['error'] = traceback.format_exc()
            logger.error(f"Experiment status: Failure.")
            logger.error(f"Exception:\n{traceback.format_exc()}")

        results.update(arguments)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))
        
        return results
    

class SimpleClusteringExperiment(BaseExperiment):
    def run(
            self,
            dataset_name: DatasetName,
            sample_n: int, 
            text_embedding_model: TextEmbeddingModel,
            clustering_model: ClusteringModel,
            k_optimization: Optional[KOptimization] = None,
            cluster_k_information_type: KInformationType = KInformationType.UnknownK,
            min_clusters: int = 2,
            max_clusters: int = 10,
            min_cluster_size: int = 0,
            batch_size: int = 128,
            random_state: int = 42
    ):
        start_datetime = datetime.now()
        
        # get data
        dataset = load_dataset_by_name(dataset_name=dataset_name)

        # sample subset
        k = 0 if cluster_k_information_type == KInformationType.GroundTruthK else None
        sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, min_cluster_size=min_cluster_size, random_state=random_state)
        sampled_dataset = get_dataset_from_df(sample_df)

        # embed the dataset for clustering
        all_ids = []
        all_embeddings = []
        all_labels = []
        dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False)
        for batch_ids, batch_texts, batch_labels in dataloader:
            batch_embeddings = text_embedding_model.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
            all_labels.extend(batch_labels)
            all_ids.extend(batch_ids)
        X = np.vstack(all_embeddings)
        labels_true = [tensor.item() for tensor in all_labels]

        # cluster the dataset using the constraint
        if cluster_k_information_type == KInformationType.UnknownK:
            # if number of clusters is unknown then optimize it
            cluster_results = clustering_model.cluster(X, k_optimization=k_optimization, min_k=min_clusters, max_k=max_clusters, random_state=random_state)
        elif cluster_k_information_type == KInformationType.GroundTruthK:
            # otherwise, provide the true cluster number to the clustering model
            n_clusters = len(set(labels_true))
            cluster_results = clustering_model.cluster(X, n_clusters=n_clusters, k_optimization=k_optimization, random_state=random_state)
        else:
            raise ValueError(f"Illegal cluster_k_information_type given in BaseClustering: {cluster_k_information_type.value}")
        
        labels_pred = cluster_results['labels']

        # compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true)

        results = dict(
            labels_true=labels_true,
            labels_pred=labels_pred,
            X=X,
            cluster_results=cluster_results
        )
        results.update(scores)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))

        return results
    

class LLMClusteringExperiment(BaseExperiment):
    def run(
            self,
            dataset_name: Optional[DatasetName],
            dataset: Optional[TextLabelDataset],
            sample_n: int, 
            llm: LLM,
            llm_k_information_type: KInformationType = KInformationType.UnknownK,
            prompt_type: PromptType = PromptType.SimpleClusteringPrompt,
            min_cluster_size: int = 0,
            llm_max_tokens: int = 8096,
            random_state: int = 42
    ):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # get data
        if (dataset_name is not None) and (dataset is not None):
            raise ValueError("Only one of 'dataset_name' or 'dataset' can be provided, not both.")
        
        if (dataset_name is None) and (dataset is None):
            raise ValueError("You must provide exactly one of 'dataset_name' or 'dataset'.")
        
        dataset = dataset or load_dataset_by_name(dataset_name=dataset_name)

        # sample subset
        k = 0 if llm_k_information_type == KInformationType.GroundTruthK else None
        sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, min_cluster_size=min_cluster_size, random_state=random_state)
        logger.debug(f"Sample a dataset of size {sample_df.shape[0]}.")

        # embed the dataset for clustering
        texts = sample_df['text'].tolist()
        labels_true = sample_df['label'].tolist()
        k = len(set(labels_true)) if llm_k_information_type == KInformationType.GroundTruthK else None
        prompt = generate_prompt(prompt_type=prompt_type, texts=texts, k=k)
        data = llm.create_messages(prompt, max_tokens=llm_max_tokens)
        formatter_func = get_formatter(prompt_type=prompt_type)
        labels_pred = formatter_func(data=data, size=len(labels_true))
        raise ValueError("Fix code here formatter does not return what is expected")
        logger.debug(f"LLM generated {len(set(labels_pred))} labels predictions.")

        # compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true)
        logger.debug(f"Computed scores.")

        results = dict(
            labels_true=labels_true,
            labels_pred=labels_pred,
        )
        results.update(scores)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))

        return results
    

class LLMConstraintedClusteringExperiment(BaseExperiment):
    def run(
            self,
            dataset_name: DatasetName,
            sample_n: int, 
            llm: LLM,
            constraint_type: ConstraintsType,
            prompt_type: PromptType,
            text_embedding_model: TextEmbeddingModel,
            clustering_model: ClusteringModel,
            llm_k_information_type: KInformationType = KInformationType.UnknownK,
            cluster_k_information_type: KInformationType = KInformationType.UnknownK,
            k_optimization: Optional[KOptimization] = None,
            min_clusters: int = 2,
            max_clusters: int = 10,
            min_cluster_size: int = 0,
            batch_size: int = 128,
            random_state: int = 42,
            **kwargs
    ):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # get data
        dataset = load_dataset_by_name(dataset_name=dataset_name)

        # sample subset
        k = 0 if llm_k_information_type == KInformationType.GroundTruthK else None
        sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, min_cluster_size=min_cluster_size, random_state=random_state)

        # run the LLM predictions to create the constraints
        sample_ids = sample_df['id'].tolist()
        sample_texts = sample_df['text'].tolist()
        sample_labels = sample_df['label'].tolist()
        constraint_model = BaseConstrainedLLM(llm=llm, constraint_type=constraint_type)
        k = len(set(sample_labels)) if llm_k_information_type == KInformationType.GroundTruthK else None
        constraint_result = constraint_model.create_constraint(prompt_type=prompt_type, ids=sample_ids, texts=sample_texts, labels=sample_labels, k=k, **kwargs)
        constraint = constraint_result.get('constraint')

        # embed the dataset for clustering
        all_ids = []
        all_embeddings = []
        all_labels = []
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_ids, batch_texts, batch_labels in dataloader:
            batch_embeddings = text_embedding_model.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
            all_labels.extend(batch_labels)
            all_ids.extend(batch_ids)
        X = np.vstack(all_embeddings)
        labels_true = [tensor.item() for tensor in all_labels]
        
        # Cluster the dataset using the constraint
        if cluster_k_information_type == KInformationType.UnknownK:
            # If number of clusters is unknown then optimize it
            max_k = max(max_clusters, len(constraint.get_labels()))
            cluster_results = clustering_model.cluster(X, constraint=constraint, k_optimization=k_optimization, min_k=min_clusters, max_k=max_k, random_state=random_state)
        elif cluster_k_information_type == KInformationType.GroundTruthK:
            # Otherwise, provide the true cluster number to the clustering model
            n_clusters = len(set(labels_true))
            cluster_results = clustering_model.cluster(X, constraint=constraint,  k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state)
        elif cluster_k_information_type == KInformationType.OracleK:
            # Or else, provide the predicted cluster number (if we can extract it from the constraint) to the clustering model
            if constraint.labels is None:
                logger.error(f"No labels were correctly predicted running with constraint {constraint_type.value}")
            else:
                labels_oracle_pred = constraint.labels
                n_clusters = len(set(labels_oracle_pred))
                cluster_results = clustering_model.cluster(X, constraint=constraint,  k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state)
        
        labels_pred = cluster_results['labels']

        # Compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true, X=X)

        results = dict(
            labels_true=labels_true,
            labels_pred=labels_pred,
            X=X,
        )
        results.update(constraint_result)
        results.update(scores)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))

        return results


class LLMConstraintedClusteringQualityExperiment(BaseExperiment):
    def run(
            self,
            dataset_name: DatasetName,
            sample_n: int, 
            llm: LLM,
            constraint_type: ConstraintsType,
            prompt_type: Optional[PromptType] = None,
            k: int = 0,
            min_cluster_size: int = 0,
            random_state: int = 42,
            **kwargs
    ):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # get data
        dataset = load_dataset_by_name(dataset_name=dataset_name)

        # sample subset
        sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, min_cluster_size=min_cluster_size, random_state=random_state)

        # run the LLM predictions to create the constraints
        sample_ids = sample_df['id'].tolist()
        sample_texts = sample_df['text'].tolist()
        sample_labels = sample_df['label'].tolist()
        constraint_model = BaseConstrainedLLM(llm=llm, constraint_type=constraint_type)
        constraint_result = constraint_model.create_constraint(prompt_type=prompt_type, ids=sample_ids, texts=sample_texts, labels=sample_labels, **kwargs)
        constraint = constraint_result.get('constraint')

        results = dict(
            k_true=len(set(sample_labels)),
            k_pred=constraint.get_k(),
        )
        results.update(constraint_result)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))

        return results
