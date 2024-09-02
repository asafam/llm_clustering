from typing import *
from datetime import datetime
import inspect
import traceback
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data import DatasetName, load_dataset_by_name, sample_dataset, get_dataset_from_df
from clustering.constraints_manager import ConstraintsType, KInformationType
from clustering.models import ClusteringModel
from clustering.optimizations import KOptimization
from clustering.utils import evaluate_clustering
from embedding.models import TextEmbeddingModel
from llms.models import LLM
from llms.utils import PromptType, generate_prompt, format_response_as_dictionary_of_clusters
from experiments.constrained_models import BaseConstrainedLLM
from experiments.utils import get_experiment_results_item_value
import logging


class BaseExperiment:
    def run(self):
        raise NotImplementedError()
    
    def run_safe(self, **kwargs):
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

        results['arguments'] = arguments
        
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
        all_embeddings = []
        all_labels = []
        dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False)
        for batch_texts, batch_labels in dataloader:
            batch_embeddings = text_embedding_model.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
            all_labels.extend(batch_labels)
        X = np.vstack(all_embeddings)
        labels_true = [tensor.item() for tensor in all_labels]

        # cluster the dataset using the constraint
        if cluster_k_information_type == KInformationType.UnknownK:
            # if number of clusters is unknown then optimize it
            labels_pred, best_k = clustering_model.cluster(X, k_optimization=k_optimization, min_k=min_clusters, max_k=max_clusters, random_state=random_state)
            n_clusters = best_k
        elif cluster_k_information_type == KInformationType.GroundTruthK:
            # otherwise, provide the true cluster number to the clustering model
            n_clusters = len(set(labels_true))
            labels_pred = clustering_model.cluster(X, n_clusters=n_clusters, random_state=random_state)
        else:
            raise ValueError(f"Illegal cluster_k_information_type given in BaseClustering: {cluster_k_information_type.value}")

        # compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true)

        results = dict(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            labels_true=labels_true,
            labels_pred=labels_pred,
            n_clusters=n_clusters
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
            dataset_name: DatasetName,
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
        dataset = load_dataset_by_name(dataset_name=dataset_name)

        # sample subset
        k = 0 if llm_k_information_type == KInformationType.GroundTruthK else None
        sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, min_cluster_size=min_cluster_size, random_state=random_state)
        logger.debug(f"Sample a dataset of size {sample_df.shape[0]}.")

        # embed the dataset for clustering
        texts = sample_df['text'].tolist()
        labels_true = sample_df['label'].tolist()
        k = len(set(labels_true)) if llm_k_information_type == KInformationType.GroundTruthK else None
        prompt = generate_prompt(prompt_type=prompt_type, texts=texts, k=k)
        result = llm.create_messages(prompt, max_tokens=llm_max_tokens)
        labels_pred = format_response_as_dictionary_of_clusters(data=result, size=len(labels_true))
        logger.debug(f"LLM generated {len(set(labels_pred))} labels predictions.")

        # compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true)
        logger.debug(f"Computed scores.")

        results = dict(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            text_embedding_model: TextEmbeddingModel,
            clustering_model: ClusteringModel,
            llm_k_information_type: KInformationType = KInformationType.UnknownK,
            k_optimization: Optional[KOptimization] = None,
            cluster_k_information_type: KInformationType = KInformationType.UnknownK, 
            min_clusters: int = 2,
            max_clusters: int = 10,
            min_cluster_size: int = 0,
            batch_size: int = 128,
            random_state: int = 42
    ):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # get data
        dataset = load_dataset_by_name(dataset_name=dataset_name)

        # sample subset
        k = 0 if llm_k_information_type == KInformationType.GroundTruthK else None
        sample_df = sample_dataset(dataset=dataset, n=sample_n, k=k, min_cluster_size=min_cluster_size, random_state=random_state)

        # run the LLM predictions to create the constraints
        sample_texts = sample_df['text'].tolist()
        sample_labels = sample_df['label'].tolist()
        constraint_model = BaseConstrainedLLM(llm=llm, constraint_type=constraint_type)
        k = len(set(sample_labels)) if llm_k_information_type == KInformationType.GroundTruthK else None
        constraint = constraint_model.create_constraint(texts=sample_texts, k=k)

        # embed the dataset for clustering
        all_embeddings = []
        all_labels = []
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_texts, batch_labels in dataloader:
            batch_embeddings = text_embedding_model.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
            all_labels.extend(batch_labels)
        X = np.vstack(all_embeddings)
        labels_true = [tensor.item() for tensor in all_labels]
        
        # cluster the dataset using the constraint
        if cluster_k_information_type == KInformationType.UnknownK:
            # if number of clusters is unknown then optimize it
            labels_pred, best_k = clustering_model.cluster(X, constraint=constraint, k_optimization=k_optimization, min_k=min_clusters, max_k=max_clusters, random_state=random_state)
        elif cluster_k_information_type == KInformationType.GroundTruthK:
            # otherwise, provide the true cluster number to the clustering model
            n_clusters = len(set(labels_true))
            labels_pred = clustering_model.cluster(X, constraint=constraint, n_clusters=n_clusters, random_state=random_state)
        elif cluster_k_information_type == KInformationType.OracleK:
            # or else, provide the predicted cluster number (if we can extract it from the constraint) to the clustering model
            if constraint.labels is None:
                logger.error(f"No labels were correctly predicted running with constraint {constraint_type.value}")
            else:
                labels_oracle_pred = constraint.labels
                n_clusters = len(set(labels_oracle_pred))
                labels_pred = clustering_model.cluster(X, constraint=constraint, n_clusters=n_clusters, random_state=random_state)

        # compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true, X=X)

        results = dict(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            labels_true=labels_true,
            labels_pred=labels_pred,
        )
        results.update(scores)
        results.update(arguments)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))

        return results
