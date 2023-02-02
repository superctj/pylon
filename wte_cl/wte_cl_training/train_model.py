import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pytorch_lightning as pl
import torch

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader

from input_prep import GitTablesDataset, PretrainingDatasetWrapper, collate
from model import ContrastiveLoss, SimTableCLR


class TableEmbeddingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.model = SimTableCLR(embedding_dim=self.hparams.embedding_dim, projection_size=self.hparams.projection_size)
        self.loss = ContrastiveLoss(temperature=self.hparams.temperature)
    
    def forward(self, X):
        return self.model(X)
    
    def step(self, batch, step_name = "train"):
        t1_embeddings, t2_embeddings = batch
        t1_projection = self.forward(t1_embeddings)
        t2_projection = self.forward(t2_embeddings)
        loss = self.loss(t1_projection, t2_projection)

        self.log(f"{step_name}/loss", loss.detach(), on_step=False, on_epoch=True)
        loss_key = f"{step_name}_loss"
        
        return { ("loss" if step_name == "train" else loss_key): loss, "progress_bar": {loss_key: loss.detach()}}
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def predict_step(self, batch, batch_idx):
        return self.model(batch)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad: params_to_update.append(param)

        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.lr / 50
        )

        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    hparams = dict(
        lr=5e-4,
        weight_decay=1e-2,
        temperature=0.1,
        epochs=20,
        batch_size=32,
        embedding_dim=150, # dimension of pre-trained word embeddings
        projection_size=64,
    )

    disk_dir = "/mnt/disks/data/congtj/" # "/ssd/congtj/" # "/mnt/disks/data/congtj/"
    table_dir = os.path.join(disk_dir, "pooled_tables/") # "data/GitTables_preprocessed_csv/pooled_tables"
    model_path = os.path.join(disk_dir, "checkpoints/web_table_embeddings_combo150.bin")
    
    train_table_roster = "../data/pretraining/v1/pretrain_train_100k.txt"
    valid_table_roster = "../data/pretraining/v1/pretrain_valid_5k.txt"

    train_dataset = GitTablesDataset(table_dir, train_table_roster, read_csv=True)
    valid_dataset = GitTablesDataset(table_dir, valid_table_roster, read_csv=True)

    train_ds_wrapper = PretrainingDatasetWrapper(train_dataset, model_path)
    valid_ds_wrapper = PretrainingDatasetWrapper(valid_dataset, model_path)

    train_loader = DataLoader(train_ds_wrapper,
                              batch_size=hparams["batch_size"],
                              collate_fn=collate,
                              shuffle=True,
                              num_workers=16,
                            #   persistent_workers=True,
                            #   pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_ds_wrapper,
                              batch_size=hparams["batch_size"],
                              collate_fn=collate,
                              shuffle=True,
                              num_workers=4,
                              drop_last=False)

    module = TableEmbeddingModule(hparams)
    logger = WandbLogger(save_dir=disk_dir, project="pylon-wte-training")
    logger.watch(module, log="all", log_freq=500, log_graph=False)

    checkpoint_callback = ModelCheckpoint(monitor="val/loss", save_top_k=5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    trainer = pl.Trainer(max_epochs=hparams["epochs"],
                         gpus=1,
                         auto_select_gpus=True,
                         strategy="ddp_find_unused_parameters_false",
                         logger=logger,
                         callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # ckpt_dir = os.path.join(disk_dir, "checkpoints/")
    # ckpt_file = os.path.join(ckpt_dir, "404_pretrain_100k_wem.ckpt")
    # trainer.save_checkpoint(ckpt_file)
    # trainer.logger.experiment.log_artifact(checkpoint_file, type='model')
