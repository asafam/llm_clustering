from typing import *
import time
import pandas as pd
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
        k_information_type: KInformationType = KInformationType.UnknownK,
        min_sample_cluster_size: int = 0,
        min_constraint_cluster_size: int = 0,
        constraint: Optional[ClusteringConstraints] = None,
        start_step: int = 0,
        max_steps: int = 5,
        llm_keep_context: bool = True,
        llm_messages_window: int = 0,
        exhaustive_step: bool = False,
        sample_until_convergence: bool = False,
        sample_same: bool = False,
        iteration_sleep_time: int = 0,
        random_state: int = 42,
        **kwargs
) -> dict:
    # Initialize a logger for debugging
    logger = logging.getLogger('default')

    # Create a dictionary mapping IDs to the true labels
    df = sample_dataset(dataset=dataset)
    id_to_label = df.set_index('id')['label'].to_dict()
    id_to_text = df.set_index('id')['text'].to_dict()
    ids = []
    
    # Determine prompt type, falling back to a prompt by constraint if not provided
    prompt_type = prompt_type or get_prompt_type(constraint_type=constraint_type)
    system_prompt, user_prompt = None, None
    
    # Initialize a prompt builder based on the prompt type
    prompt_builder = get_prompt_builder(prompt_type=prompt_type)
    logger.debug(f"Using prompt builder: {prompt_builder}")
    
    # List to accumulate messages sent to the LLM
    messages = []
    
    # Ensure sample_n is a list, even if passed as a single integer
    sample_ns = sample_n if type(sample_n) == list else [sample_n]
    sample_size = None # Placeholder for this var
    sampled_all_dataset = False

    # Normalize the start_step index
    start_step = start_step if (start_step != -1 and start_step < len(sample_ns)) else (len(sample_ns) - 1)

    # Iterate over steps to sample subsets and generate constraints
    steps = sample_ns[start_step:].copy()
    unclustered_sample_df = pd.DataFrame()
    constraints = []
    for step, n in enumerate(steps, start=start_step):
        logger.debug(f"Create contraints: step {step+1} / {len(steps)}")
        
        # Sample a subset of the dataset, excluding already sampled IDs
        sample_df = sample_dataset(
            dataset=dataset, 
            n=n, 
            k=k, 
            min_cluster_size=min_sample_cluster_size, 
            exclude_ids=ids if (not sample_same) else [], 
            random_state=random_state if (sample_same or sample_until_convergence) else (random_state + step * 1000)
        )
        orig_sample_size = len(sample_df)

        # Add the previously unclustered sentences to the task
        if exhaustive_step:
            sample_df = pd.concat([sample_df, unclustered_sample_df], ignore_index=True)
        
        if constraint is not None:
            # Exclude ids that have already been assinged with a cluster label
            sample_df = sample_df[~sample_df["id"].isin(constraint.get_ids())]
        
        # Break loop if we labeled all sentences
        sample_size = len(sample_df)
        logger.debug(f"sample_size = {sample_size}, (orig_sample_size = {orig_sample_size})")
        if sample_size == 0:
            sampled_all_dataset = True
            logger.debug(f"Nothing left to sample. Breaking the loop...")
            break

        # Extract IDs, texts, and labels from the sampled subset
        sample_ids = sample_df['id'].tolist()
        sample_texts = sample_df['text'].tolist()
        sample_labels = sample_df['label'].tolist()

        # Append sampled IDs and labels to the accumulated lists
        constraint_ids = constraint.get_ids() if constraint is not None else []
        logger.debug(f"Using a constraint with {len(constraint_ids)} clustered labels")
        ids = list(set(constraint_ids + sample_ids))
        labels_true = [id_to_label[id] for id in ids]

        # Get the context from the current constraint
        context = dict(sentences_labels=constraint.get_ids_labels(), ids_texts=constraint.get_ids_texts(), explanations=constraint.get_explanations()) if constraint is not None else None
        
        # Generate the prompt using the builder
        # Select the user prompt to build
        prompts = prompt_builder.get_prompts(step=step)
        system_prompt, user_prompt = prompt_builder.build_prompt(
            system_prompt = prompts.get("system_prompt"),
            user_prompt = prompts.get("user_prompt"),
            step=step, 
            context=context, 
            ids=sample_ids, 
            texts=sample_texts, 
            n_clusters=len(set(id_to_label.values())) if k_information_type == KInformationType.GroundTruthK else None,
            **kwargs
        ) # Build the prompt

        # Check whether to reset the context messages list 
        if not llm_keep_context:
            messages = []

        # Add the prompt as a user message for the LLM
        messages += [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # logger.debug(f"System prompt = {system_prompt}")
        # logger.debug(f"User prompt = {user_prompt}")
        # logger.debug(f"messages = {messages}")
        
        # Call the LLM to get predictions based on the messages
        messages_window = llm_messages_window + (len(messages) % 2)
        data_raw, data_raw_text = llm.create_messages(system_prompt=system_prompt, messages=messages[-messages_window:]) # LLM output
        
        # Add the LLM's response to messages
        messages += [
            {
                "role": "assistant",
                "content": data_raw_text
            }
        ]

        # Format the raw data using the appropriate formatter function
        format_func = get_formatter(prompt_type=prompt_type, step=step)
        data = format_func(data_raw, context=context) if format_func else data_raw

        # Generate constraints based on the accumulated data and sampled subset
        constraint = generate_constraint(
            data=data, 
            constraint_type=constraint_type, 
            ids=sample_ids, 
            texts=sample_texts, 
            labels=sample_labels, 
            id_to_text=id_to_text,
            min_cluster_size=min_constraint_cluster_size,
            **kwargs
        )
        constraints.append(constraint)

        unclustered_sample_df = sample_df[~sample_df["id"].isin(constraint.get_ids())]

        if sample_until_convergence and sample_size != 0 and step < max_steps and (step == len(steps) - 1):
            logger.debug(f"Sampling until convergence: step {step+1}, sample_size = {sample_size} therefore adding another step")
            steps.append(steps[-1])

        logger.debug(f"Sleep {iteration_sleep_time} seconds")
        time.sleep(iteration_sleep_time)
        logger.debug(f"Woke up. Continue iteration")

    # Evaluate the quality of the final constraint against true labels
    constraint_quality = constraint.evaluate(ids_to_labels_true=id_to_label)
    
    # Store constraint, prompt, and its quality in a dictionary
    constraint_result = dict(
        constraint=constraint,
        constraints=constraints,
        user_prompt=user_prompt, 
        system_prompt=system_prompt,
        constraint_quality=constraint_quality,
        sample_size = sample_size,
        sampled_ids = ids,
        sampled_all_dataset = sampled_all_dataset
    )

    # Compile final results including all sampled IDs, true labels, and constraint data
    results = dict(
        ids=ids,
        labels_true=labels_true
    )
    results.update(constraint_result)

    return results  # Return the complete result dictionary


            
def cluster_with_constraint(
        X,
        ids: list,
        labels_true: list,
        constraint: ClusteringConstraints,
        clustering_model: ClusteringModel,
        cluster_k_information_type: KInformationType = KInformationType.UnknownK,
        k_optimization: Optional[KOptimization] = None,
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 1,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
        **kwargs
):
    logger = logging.getLogger('default')
    if n_clusters is not None:
        # Using provided number of clusters
        logger.debug(f"K is known, using provided k of {n_clusters}")
        cluster_results = clustering_model.cluster(X, ids=ids, k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif cluster_k_information_type == KInformationType.UnknownK:
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
        if constraint.get_labels_count() is None:
            raise ValueError(f"No labels were correctly predicted running with constraint {constraint}")
        else:
            n_clusters = len([_ for _, count in constraint.get_labels_count().items() if count >= min_cluster_size])
            logger.debug(f"K is predicted, using constraint predicted k of {n_clusters} (using threshold of {min_cluster_size}, originally predicted {len([label for label in constraint.get_labels_count().keys()])})")
            cluster_results = clustering_model.cluster(X, ids=ids, constraint=constraint,  k_optimization=k_optimization, n_clusters=n_clusters, random_state=random_state, **kwargs)

    return cluster_results
