from typing import *
from data import sample_dataset
from llms.llm_manager import get_prompt_builder
from llms.models import *
from llms.utils import *
from experiments.utils import get_prompt_type
from clustering.constraints import *
from clustering.constraints_manager import ConstraintsType, KInformationType, generate_constraint
from clustering.models.base_models import ClusteringModel
from clustering.optimizations import KOptimization
import logging

def create_constraint(
        dataset,
        sample_n: int|list, 
        llm: LLM,
        constraint_type: ConstraintsType,
        prompt_type: Optional[PromptType] = None,
        k: int = 0,
        min_cluster_size: int = 0,
        random_state: int = 42,
        **kwargs
) -> dict:
    logger = logging.getLogger('default')
    ids = []
    labels_true = []
    
    prompt_type = prompt_type or get_prompt_type(constraint_type=constraint_type)
    prompt_builder = get_prompt_builder(prompt_type=prompt_type)
    logger.debug(f"Using prompt builder: {prompt_builder}")
    messages = []
    data = None
    for step in range(prompt_builder.get_steps_count()):
        # sample subset
        n = sample_n[min(step, len(sample_n) - 1)] if type(sample_n) == list else sample_n
        sample_df = sample_dataset(dataset=dataset, n=n, k=k, min_cluster_size=min_cluster_size, exclude_ids=ids, random_state=(random_state + step * 1000))
        sample_ids = sample_df['id'].tolist()
        sample_texts = sample_df['text'].tolist()
        sample_labels = sample_df['label'].tolist()

        ids += sample_ids
        labels_true += sample_labels
        # run the LLM predictions to create the constraints

        # Generate the prompt
        prompt = prompt_builder.build_prompt(step=step, context=data, ids=sample_ids, texts=sample_texts, **kwargs) # Build the prompt

        # Send prompt to LLM
        messages += [
            {
                "role": "user",
                "content": prompt
            }
        ]
        data_raw, data_raw_text = llm.create_messages(messages=messages) # Call the LLM with the prompt
        messages += [
            {
                "role": "assistant",
                "content": data_raw_text
            }
        ]
        # Format the output
        format_func = get_formatter(prompt_type=prompt_type, step=step)
        data = format_func(data_raw, context=data) if format_func else data_raw

    # Generate the constraints
    constraint = generate_constraint(data=data, constraint_type=constraint_type, ids=sample_ids, texts=sample_texts, labels=sample_labels, **kwargs)
    constraint_quality = constraint.evaluate(ids_true=ids, labels_true=labels_true)
    
    constraint_result = dict(
        constraint=constraint,
        prompt=prompt,
        constraint_quality = constraint_quality
    )

    # Build the result
    results = dict(
        ids=ids,
        labels_true=labels_true
    )
    results.update(constraint_result)

    return results

            
def cluster_with_constraint(
        X,
        ids: list,
        labels_true: list,
        constraint: ClusteringConstraints,
        clustering_model: ClusteringModel,
        cluster_k_information_type: KInformationType = KInformationType.UnknownK,
        k_optimization: Optional[KOptimization] = None,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
        **kwargs
):
    logger = logging.getLogger('default')
    if cluster_k_information_type == KInformationType.UnknownK:
        # If number of clusters is unknown then optimize it
        unique_labels = constraint.get_unique_labels()
        max_k = max(max_clusters, len(unique_labels) if unique_labels is not None else 0)
        logger.debug(f"K is unknown, using max_k of {max_k}")
        cluster_results = clustering_model.cluster(X, ids=ids, constraint=constraint, k_optimization=k_optimization, min_k=min_clusters, max_k=max_k, random_state=random_state, **kwargs)
    elif cluster_k_information_type == KInformationType.GroundTruthK:
        # Otherwise, provide the true cluster number to the clustering model
        n_clusters = len(set(labels_true))
        logger.debug(f"K is known, using true k of {n_clusters}")
        cluster_results = clustering_model.cluster(X, ids=ids, constraint=constraint, k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif cluster_k_information_type == KInformationType.OracleK:
        # Or else, provide the predicted cluster number (if we can extract it from the constraint) to the clustering model
        if constraint.unique_labels is None:
            raise ValueError(f"No labels were correctly predicted running with constraint {constraint}")
        else:
            n_clusters = len(constraint.unique_labels)
            logger.debug(f"K is predicted, using constraint predicted k of {n_clusters}")
            cluster_results = clustering_model.cluster(X, ids=ids, constraint=constraint,  k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state, **kwargs)

    return cluster_results
