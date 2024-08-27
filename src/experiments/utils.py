import os
import pickle
import logging
from clustering.constraints_manager import ConstraintsType
from llms.utils import PromptType


def get_logger():
    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_experiments(experiments, file_path):
    logger = logging.getLogger('default')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as file:
        pickle.dump(experiments, file)
    logger.debug(f"{len(experiments)} experiments saved to {file_path}")

    
def load_experiments(file_path):
    logger = logging.getLogger('default')
    if not os.path.exists(file_path):
        logger.error(f"Could not find experiments file in {file_path}")
        return None
    
    with open(file_path, 'rb') as file:
        experiments = pickle.load(file)
    logger.debug(f"{len(experiments)} experiments loaded from {file_path}")
    return experiments


def is_experiment_completed(experiments, **kwargs):
    for experiment in experiments:
        if all(item in experiment.items() for item in kwargs.items()):
            return True
    return False


def get_prompt_type(constraints_type: ConstraintsType):
    if constraints_type == ConstraintsType.HardLabelsConstraints:
        return PromptType.HardLabelsClusteringPrompt
    elif constraints_type == ConstraintsType.FuzzyLabelsConstraints:
        return PromptType.FuzzyLabelsClusteringPrompt
    elif constraints_type == ConstraintsType.MustLinkCannotLinkConstraints:
        return PromptType.MustLinkCannotLinkClusteringPrompt

