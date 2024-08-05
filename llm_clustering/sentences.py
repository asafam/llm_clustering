import torch
import numpy as np



class Encoder():
    def __init__(self, tokenizer, model, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model
        
    def encode(self, texts):
        """
        Encode a list of texts
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Average pooling of token embeddings
        return embeddings


def embed(model, tokenizer, dataloader, encode_fn=None):
    """
    Function to encode a batch of texts
    """
    # Iterate over the DataLoader and encode each batch
    all_embeddings = []
    all_labels = []

    encode_fn = encode_fn or Encoder(model, tokenizer).encode
    for batch_texts, batch_labels in dataloader:
        batch_embeddings = encode_fn(batch_texts)
        all_embeddings.append(batch_embeddings)
        all_labels.extend(batch_labels)

    # Convert the list of embeddings to a single numpy array
    all_embeddings = np.vstack(all_embeddings)

    # Combine embeddings with intents
    embedding_label_df = pd.DataFrame(all_embeddings)
    embedding_label_df['label'] = [tensor.item() for tensor in all_labels]
    
    return embedding_label_df