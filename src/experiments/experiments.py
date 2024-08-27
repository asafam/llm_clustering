import pandas as pd
from embedding.models import TextEmbeddingModel
from clustering.models import ClusteringModel
from clustering.utils import assess_clustering
from experiments.utils import embed


class BaseExperiment:
    def run(self, df: pd.DataFrame, clustering_model: ClusteringModel, text_embedding_model:TextEmbeddingModel, label_column='label', k_known=False):
        embedding_label_df = embed(df=df, text_embedding_model=text_embedding_model, label_column=label_column)
        X = embedding_label_df.drop(columns=['label']).to_numpy()
        gold_labels = embedding_label_df['label'].to_list()
        gold_k = len(set(gold_labels))
        pred_labels = clustering_model.cluster(X, gold_labels, k=(gold_k if k_known else 0))
        scores = assess_clustering(gold_labels, pred_labels, X)

        return dict(
            texts=df['text'].tolist(),
            gold_labels = gold_labels,
            pred_labels = pred_labels,
            scores=scores
        )
    

class LLMConstraintExperiment(BaseExperiment):
    def run(self, df: pd.DataFrame, clustering_model: ClusteringModel, text_embedding_model: TextEmbeddingModel, label_column='label', k_known=False):
        # predict with LLM

        # form constraint
        constraint = 

        # cluster with constraints
        embedding_label_df = embed(df=df, text_embedding_model=text_embedding_model, label_column=label_column)
        X = embedding_label_df.drop(columns=['label']).to_numpy()
        pred_labels = clustering_model.cluster(X, k=(gold_k if k_known else 0))

        # asses results
        gold_labels = embedding_label_df['label'].to_list()
        gold_k = len(set(gold_labels))
        scores = assess_clustering(gold_labels, pred_labels, X)

        return dict(
            texts=df['text'].tolist(),
            gold_labels = gold_labels,
            pred_labels = pred_labels,
            scores=scores
        )