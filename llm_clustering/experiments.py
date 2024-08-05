from clustering_util import Encoder, embed, cluster, assess_clustering
from llm2vec import LLM2Vec

class BaseExperiment:
    def run(self, model, tokenizer, dataloader, k_known=False, encode_fn=None):
        encode_fn = encode_fn or Encoder(tokenizer, model).encode

        embedding_label_df = embed(model, tokenizer, dataloader, encode_fn=encode_fn)

        X = embedding_label_df.drop(columns=['label']).to_numpy()
        gold_labels = embedding_label_df['label'].to_list()
        gold_k = len(set(gold_labels))
        pred_labels = cluster(X, gold_labels, k=(gold_k if k_known else 0))
        scores = assess_clustering(gold_labels, pred_labels, X)

        return dict(
            gold_labels = gold_labels,
            pred_labels = pred_labels,
            scores=scores
        )
            

class LLM2VecExperiment(BaseExperiment):
    def run(self, model, tokenizer, dataloader, k_known=False):
        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        encode_fn=l2v.encode

        return super.run(model, tokenizer, dataloader, k_known, encode_fn)
