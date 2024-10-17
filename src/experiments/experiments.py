from typing import *
from datetime import datetime
import traceback
from torch.utils.data import DataLoader
import numpy as np
from data import DatasetName, TextLabelDataset, load_dataset_by_name, sample_dataset, get_dataset_from_df
from clustering.constraints_manager import ConstraintsType, KInformationType
from clustering.constraints import *
from clustering.models.base_models import ClusteringModel
from clustering.optimizations import KOptimization
from clustering.utils import evaluate_clustering
from experiments.experiment_manager import *
from experiments.utils import *
from embedding.models import TextEmbeddingModel
from llms.models import LLM
from llms.utils import PromptType, generate_prompt, get_formatter
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
            dataset_k: Optional[int] = None,
            constraint: Optional[ClusteringConstraints] = None,
            min_clusters: int = 2,
            max_clusters: int = 10,
            min_cluster_size: int = 0,
            batch_size: int = 128,
            random_state: int = 42,
            **kwargs
    ):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # Load the dataset
        start_datetime2 = datetime.now()
        dataset = load_dataset_by_name(dataset_name=dataset_name)
        if dataset_k is not None:
            sample_df = sample_dataset(dataset=dataset, k=dataset_k, random_state=random_state)
            dataset = get_dataset_from_df(sample_df)
        logger.debug(f"Dataset loading completed in {(datetime.now() - start_datetime2).total_seconds()} seconds")

        # Create the constraint
        if constraint is None:
            start_datetime2 = datetime.now()
            k = 0 if llm_k_information_type == KInformationType.GroundTruthK else None
            constraint_results = create_constraint(
                dataset,
                sample_n,
                llm,
                constraint_type,
                prompt_type,
                k,
                min_cluster_size,
                random_state,
                **kwargs
            )
            constraint = constraint_results['constraint']
            logger.debug(f"Constraint evaluation returned {constraint_results['constraint_quality']}")
            logger.debug(f"Constraint creation completed in {(datetime.now() - start_datetime2).total_seconds()} seconds")
        else:
            constraint_results = {'constraint': constraint}
            logger.debug("Constraing provided in parameters. No need to create it.")

        # Get the dataset embeddings for clustering
        start_datetime2 = datetime.now()
        X, ids, labels_true = encode_dataset(dataset=dataset, model=text_embedding_model, batch_size=batch_size)
        logger.debug(f"Dataset embedding completed in {(datetime.now() - start_datetime2).total_seconds()} seconds")

        # Cluster the dataset using the constraint
        start_datetime2 = datetime.now()
        cluster_results = cluster_with_constraint(
            X,
            ids,
            labels_true,
            constraint,
            clustering_model,
            cluster_k_information_type,
            k_optimization,
            min_clusters,
            max_clusters,
            random_state,
            **kwargs
        )
        labels_pred = cluster_results['labels']
        logger.debug(f"Clustering completed in {(datetime.now() - start_datetime2).total_seconds()} seconds")

        # Compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true, X=X)

        # Prepare the results 
        results = dict(
            ids=ids,
            labels_true=labels_true,
            labels_pred=labels_pred,
            X=X,
        )
        results.update(constraint_results)
        results.update(scores)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))
        logger.debug(f"Experiment completed in {(end_datetime - start_datetime).total_seconds()} seconds")

        return results


class LLMConstraintedClusteringQualityExperiment(BaseExperiment):
    def run(
            self,
            dataset_name: DatasetName,
            sample_n: int|list, 
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

        constraint_model = BaseConstrainedLLM()
        constraint_results = constraint_model.create_constraint(
            dataset,
            sample_n,
            llm,
            constraint_type,
            prompt_type,
            k,
            min_cluster_size,
            random_state,
            **kwargs
        )
        logger.debug(f"Constraint evaluation returned {constraint_results['constraint_quality']}")

        end_datetime = datetime.now()
        results = constraint_results
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))

        return results
