from tabert_cl_training.train_model import TableEmbeddingModule


def load_pylon_tabert_model(ckpt_path: str):
    return TableEmbeddingModule.load_from_checkpoint(ckpt_path).model
    