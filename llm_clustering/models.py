import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from llm2vec import LLM2Vec
from peft import PeftModel


def load_model(model_name, device):
    if 'LLM2Vec' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Loading MNTP (Masked Next Token Prediction) model.
        model = PeftModel.from_pretrained(
            model,
            model_name,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    return tokenizer, model