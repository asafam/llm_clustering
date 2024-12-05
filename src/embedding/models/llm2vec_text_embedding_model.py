import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec
from models import TextEmbeddingModel, EmbeddingModelName
import logging 

class LLM2VecTextEmbeddingModel(TextEmbeddingModel):
    def __init__(self, model_name: EmbeddingModelName) -> None:
        super().__init__()
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

        self.l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    def embed(self, texts, **kwargs):
        embeddings = self.l2v.encode(texts)
        return embeddings