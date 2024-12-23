from InstructorEmbedding import INSTRUCTOR
from embedding.models import TextEmbeddingModel

class InstructorTextEmbeddingModel(TextEmbeddingModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = INSTRUCTOR('hkunlp/instructor-large')

    def embed(self, texts, instruction, **kwargs):
        embeddings = self.model.encode([[instruction,sentence] for sentence in texts])
        return embeddings