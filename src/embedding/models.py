import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec
from enum import Enum
import logging 


class EmbeddingModelName(Enum):
    anthropic_claude_3_sonnet_20240229_v1 = "anthropic.claude-3-sonnet-20240229-v1"
    all_MiniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    intfloat__e5_large_v2 = "intfloat/e5-large-v2"
    Alibaba_NLP__gte_large_en_v1_5 = "Alibaba-NLP/gte-large-en-v1.5"
    McGill_NLP__LLM2Vec_Mistral_7B_Instruct_v2_mntp = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
    McGill_NLP__LLM2Vec_Meta_Llama_3_8B_Instruct_mntp = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"


class TextEmbeddingModel:
    def embed(self, texts):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__


class UniversalTextEmbeddingModel(TextEmbeddingModel):
    def __init__(self, model_name: EmbeddingModelName) -> None:
        super().__init__()
        self.model_name = model_name
        logger = logging.getLogger('default')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Loading UniversalTextEmbeddingModel {model_name.value}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name.value, force_download=False)
        self.model = AutoModel.from_pretrained(model_name.value, trust_remote_code=True, force_download=False).to(self.device)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name.value})"
        
    def embed(self, texts):
        """
        Encode a list of texts
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Average pooling of token embeddings
        return embeddings
    

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

    def embed(self, texts):
        embeddings = self.l2v.encode(texts)
        return embeddings
