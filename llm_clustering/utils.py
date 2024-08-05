import os
import pickle
import logging


def save_experiments(experiments, file_path):
    logger = logging.getLogger()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as file:
        pickle.dump(experiments, file)
    logger.debug(f"{len(experiments)} experiments saved to {file_path}")

    
def load_experiments(file_path):
    logger = logging.getLogger()
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