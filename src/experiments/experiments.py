from typing import *
from datetime import datetime
import traceback
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
            text_embedding_model: Optional[TextEmbeddingModel],
            text_embedding_path: Optional[str],
            clustering_model: ClusteringModel,
            cluster_k_information_type: KInformationType = KInformationType.UnknownK,
            n_clusters: Optional[int] = None,
            k_optimization: Optional[KOptimization] = None,
            min_clusters: int = 2,
            max_clusters: int = 500,
            batch_size: int = 128,
            random_state: int = 42,
            **kwargs
    ):
        start_datetime = datetime.now()
        logger = logging.getLogger('default')

        # Load the dataset
        start_datetime2 = datetime.now()
        dataset = load_dataset_by_name(dataset_name=dataset_name)
        logger.debug(f"Dataset loading completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")

        # Get the dataset embeddings for clustering
        start_datetime2 = datetime.now()
        text_embedding_file_path = os.path.join(text_embedding_path, f"{dataset_name.value}.npy")
        X, ids, labels_true = encode_dataset(dataset=dataset, model=text_embedding_model, file_path=text_embedding_file_path, batch_size=batch_size)
        logger.debug(f"Dataset embedding completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")

        # Cluster the dataset using the constraint
        start_datetime2 = datetime.now()
        # cluster the dataset using the constraint
        if n_clusters is not None:
            # Using provided number of clusters
            logger.debug(f"K is known, using provided k of {n_clusters}")
            cluster_results = clustering_model.cluster(X, ids=ids, k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state, **kwargs)
        elif cluster_k_information_type == KInformationType.UnknownK:
            # If number of clusters is unknown then optimize it
            logger.debug(f"K is unknown, using k in range [{min_clusters}, {max_clusters}]")
            cluster_results = clustering_model.cluster(X, ids=ids, k_optimization=k_optimization, min_k=min_clusters, max_k=max_clusters, random_state=random_state, **kwargs)
        elif cluster_k_information_type == KInformationType.GroundTruthK:
            # Otherwise, provide the true cluster number to the clustering model
            n_clusters = len(set(labels_true))
            logger.debug(f"K is known, using true k of {n_clusters}")
            cluster_results = clustering_model.cluster(X, ids=ids, k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state, **kwargs)
        else:
            cluster_results = None
            raise ValueError(f"Illegal cluster_k_information_type given in BaseClustering: {cluster_k_information_type.value}")
        logger.debug(f"Clustering completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")

        # Compute score for the clustering
        labels_pred = cluster_results['labels']
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true, X=X)

        # Prepare the results 
        results = dict(
            ids=ids,
            labels_true=labels_true,
            labels_pred=labels_pred,
        )
        results['cluster_results'] = cluster_results
        results.update(scores)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))
        logger.debug(f"Experiment completed in {(end_datetime - start_datetime).total_seconds():.2f} seconds")

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
            text_embedding_model: Optional[TextEmbeddingModel],
            text_embedding_path: Optional[str],
            clustering_model: ClusteringModel,
            llm_k_information_type: KInformationType = KInformationType.UnknownK,
            cluster_k_information_type: KInformationType = KInformationType.UnknownK,
            k_optimization: Optional[KOptimization] = None,
            llm_predicted_k_with_min_cluster_size: int = 1,
            llm_keep_context: bool = True,
            llm_messages_window: int = 0,
            dataset_k: Optional[int] = None,
            constraint: Optional[ClusteringConstraints] = None,
            n_clusters: Optional[int] = None, 
            min_clusters: int = 2,
            max_clusters: int = 10,
            min_cluster_size: int = 0,
            batch_size: int = 128,
            max_constraint_iterations: int = 1,
            exhaustive_step: bool = False,
            sample_until_convergence: bool = False,
            sample_same: bool = False,
            max_steps: int = 10,
            iteration_sleep_time: int = 5,
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
        logger.debug(f"Dataset loading completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")

        # Create the constraint
        if constraint is None:
            start_datetime2 = datetime.now()
            i = 0
            sampled_all_dataset = False
            while not sampled_all_dataset and i < max_constraint_iterations:
                logger.debug(f"Creating constraint: iteration {i + 1} / {max_constraint_iterations}")
                
                prompt_examples_file = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR', ''), 'prompts', 'examples.yaml')
                with open(prompt_examples_file, "r") as file:
                    examples = yaml.safe_load(file)
                kwargs["prompt_examples"] = examples.get(dataset_name.value, examples.get('default'))
                
                constraint_results = create_constraint(
                    dataset=dataset,
                    sample_n=sample_n,
                    llm=llm,
                    constraint_type=constraint_type,
                    prompt_type=prompt_type,
                    k=kwargs.get("sample_k"),
                    k_information_type=llm_k_information_type,
                    min_sample_cluster_size=min_cluster_size,
                    min_constraint_cluster_size=llm_predicted_k_with_min_cluster_size,
                    constraint=constraint,
                    start_step = 0 if i == 0 else -1,
                    max_steps = max_steps,
                    llm_keep_context=llm_keep_context,
                    llm_messages_window=llm_messages_window,
                    exhaustive_step=exhaustive_step,
                    sample_until_convergence=sample_until_convergence,
                    sample_same = sample_same,
                    iteration_sleep_time=iteration_sleep_time,
                    random_state=random_state,
                    **kwargs
                )
                constraint = constraint_results['constraint']
                sampled_all_dataset = constraint_results['sampled_all_dataset']
                logger.debug(f"Constraint evaluation returned {constraint_results['constraint_quality']}")
                
                if constraint_results['sample_size'] == 0 and llm_predicted_k_with_min_cluster_size > 1:
                    logger.debug(f"Completed creating constraint after {i + 1} iterations")
                    break
                
                i += 1

            if sampled_all_dataset:
                logger.debug(f"Completed iterating over the entire dataset after {i + 1} iterations")
            elif i == max_constraint_iterations and max_constraint_iterations > 1:
                logger.debug(f"Completed creating constraint after {i + 1} max iterations")

            logger.debug(f"Constraint creation completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")
        else:
            constraint_results = {'constraint': constraint}
            logger.debug("Constraing provided in parameters. No need to create it.")

        # Get the dataset embeddings for clustering
        start_datetime2 = datetime.now()
        text_embedding_file_path = os.path.join(text_embedding_path, f"{dataset_name.value}.npy")
        X, ids, labels_true = encode_dataset(dataset=dataset, model=text_embedding_model, file_path=text_embedding_file_path, batch_size=batch_size)
        logger.debug(f"Dataset embedding completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")

        # Cluster the dataset using the constraint
        start_datetime2 = datetime.now()
        cluster_results = cluster_with_constraint(
            X=X,
            ids=ids,
            labels_true=labels_true,
            constraint=constraint,
            clustering_model=clustering_model,
            cluster_k_information_type=cluster_k_information_type,
            k_optimization=k_optimization,
            n_clusters=n_clusters,
            min_cluster_size=llm_predicted_k_with_min_cluster_size,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_state=random_state,
            **kwargs
        )
        labels_pred = cluster_results['labels']
        logger.debug(f"Clustering completed in {(datetime.now() - start_datetime2).total_seconds():.2f} seconds")

        # Compute score for the clustering
        scores = evaluate_clustering(labels_pred=labels_pred, labels_true=labels_true, X=X)

        # Prepare the results 
        results = dict(
            ids=ids,
            labels_true=labels_true,
            labels_pred=labels_pred,
        )
        results['cluster_results'] = cluster_results
        results.update(constraint_results)
        results.update(scores)

        end_datetime = datetime.now()
        results.update(dict(
            start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            time_in_seconds=(end_datetime - start_datetime).total_seconds()
        ))
        logger.debug(f"Experiment completed in {(end_datetime - start_datetime).total_seconds():.2f} seconds")

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

