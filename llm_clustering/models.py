import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from enum import Enum


class EmbeddingModelName(Enum):
    all_MiniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    intfloat__e5_large_v2 = "intfloat/e5-large-v2"
    Alibaba_NLP__gte_large_en_v1_5 = "Alibaba-NLP/gte-large-en-v1.5"
    McGill_NLP__LLM2Vec_Mistral_7B_Instruct_v2_mntp = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
    McGill_NLP__LLM2Vec_Meta_Llama_3_8B_Instruct_mntp = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"


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
