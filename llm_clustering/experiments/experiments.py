import pandas as pd
from models.embedding import TextEmbeddingModel
from llm_clustering.clustering.constraints import ClusteringConstraints
from llm_clustering.clustering.utils import cluster, assess_clustering
from experiments.utils import embed


class BaseExperiment:
    def run(self, df: pd.DataFrame, text_embedding_model:TextEmbeddingModel, label_column='label', k_known=False):
        embedding_label_df = embed(df=df, text_embedding_model=text_embedding_model, label_column=label_column)
        X = embedding_label_df.drop(columns=['label']).to_numpy()
        gold_labels = embedding_label_df['label'].to_list()
        gold_k = len(set(gold_labels))
        pred_labels = cluster(X, gold_labels, k=(gold_k if k_known else 0))
        scores = assess_clustering(gold_labels, pred_labels, X)

        return dict(
            texts=df['text'].tolist(),
            gold_labels = gold_labels,
            pred_labels = pred_labels,
            scores=scores
        )
    

class ConstraintExperiment(BaseExperiment):
    def run(self, df: pd.DataFrame, constraint: ClusteringConstraints, text_embedding_model: TextEmbeddingModel, label_column='label', k_known=False):
