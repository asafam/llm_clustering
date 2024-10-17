from llms.prompt_builders.base_prompt_builder import BasePromptBuilder
from llms.utils import *
import os
import importlib.util
import inspect

def get_prompt_builder(
        prompt_type: str, 
        folder_path: str = os.path.join(os.getenv('LLM_CLUSTERING_BASE_DIR'), 'src/llms/prompt_builders')
    ):
    """
    Loads all classes that are either BasePromptBuilder or extend it from the given folder, 
    instantiates them with the given prompt_type, checks if they match the prompt_type, 
    and returns the one with the highest priority.
    
    :param prompt_type: The type of prompt to match
    :param folder_path: The path to the folder containing the Python files with prompt builders
    :return: The class instance with the highest priority that matches the prompt_type
    """
    # Get all Python files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.py')]

    # List to hold instantiated classes
    prompt_builders = []

    # Loop through the files and load classes that are BasePromptBuilder or extend it
    for file in files:
        module_path = os.path.join(folder_path, file)

        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(file[:-3], module_path)  # Remove .py extension
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Iterate over all members of the module to find classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class is BasePromptBuilder or extends it
            if issubclass(obj, BasePromptBuilder):
                try:
                    # Instantiate the class with the given prompt_type
                    prompt_builder = obj(prompt_type=prompt_type)
                    prompt_builders.append(prompt_builder)
                except Exception as e:
                    print(f"Failed to instantiate {name}: {e}")

    # Filter classes by whether they match the prompt_type
    matching_builders = [pb for pb in prompt_builders if pb.is_match(prompt_type)]

    if not matching_builders:
        return None

    # Sort by priority (higher priority first)
    matching_builders.sort(key=lambda pb: pb.get_priority(), reverse=True)

    return matching_builders[0]

