import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pytorch_lightning as pl
import torch

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader

from model import ContrastiveLoss, SimTableCLR
from dataset import GitTablesDataset, PretrainingDatasetWrapper, collate
# from dataset_only_text import GitTablesDataset, PretrainingDatasetWrapper, collate


class TableEmbeddingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.model = SimTableCLR(projection_size=self.hparams.projection_size)
        self.loss = ContrastiveLoss(temperature=self.hparams.temperature)
    
    def forward(self, X):
        return self.model(X)
    
    def step(self, batch, step_name="train"):
        t1_projection, t2_projection = self.forward(batch)
        # loss, acc = self.loss(t1_projection, t2_projection)
        loss = self.loss(t1_projection, t2_projection)
        self.log(f"{step_name}/loss", loss.detach(), on_step=False, on_epoch=True)
        # self.log(f"{step_name}/accuracy", acc.detach(), on_step=False, on_epoch=True)
        
        loss_key = f"{step_name}_loss"
        # acc_key = f"{step_name}_acc"

        # return {("loss" if step_name == "train" else loss_key): loss, "progress_bar": {loss_key: loss.detach(), acc_key: acc.detach()}}
        return {("loss" if step_name == "train" else loss_key): loss, "progress_bar": {loss_key: loss.detach()}}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def predict_step(self, batch, batch_idx):
        # return self.model.get_tabert_encoding(batch)
        return self.model.inference(batch)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad: params_to_update.append(param)

        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[int(self.hparams.epochs * 0.8)],
        #     gamma=0.1
        # )

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
        epochs=100,
        batch_size=8,
        projection_size=128,
    )

    disk_dir = "/mnt/disks/data/congtj/" # "/ssd/congtj/" 
    table_dir = os.path.join(disk_dir, "pooled_tables/") # "data/GitTables_preprocessed_csv/pooled_tables/"
    
    train_table_roster = "./data/pretraining/v1/pretrain_train_100k.txt" # "./data/pretraining/pretrain_100k.txt"
    valid_table_roster = "./data/pretraining/v1/pretrain_valid_5k.txt"

    train_dataset = GitTablesDataset(table_dir, train_table_roster, read_csv=True)
    valid_dataset = GitTablesDataset(table_dir, valid_table_roster, read_csv=True)
    
    train_ds_wrapper = PretrainingDatasetWrapper(train_dataset, num_rows=5)
    valid_ds_wrapper = PretrainingDatasetWrapper(valid_dataset, num_rows=5)

    train_loader = DataLoader(train_ds_wrapper,
                              batch_size=hparams["batch_size"],
                              collate_fn=collate,
                              shuffle=True,
                              num_workers=24, #10
                            #   persistent_workers=True,
                            #   pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_ds_wrapper,
                              batch_size=hparams["batch_size"],
                              collate_fn=collate,
                              shuffle=True,
                              num_workers=4, #2
                              drop_last=False)

    module = TableEmbeddingModule(hparams)
    logger = WandbLogger(save_dir=disk_dir, project="pylon-full-training")
    logger.watch(module, log="all", log_freq=500, log_graph=False)

    checkpoint_callback = ModelCheckpoint(monitor="val/loss", save_top_k=3) # "train/loss"
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    trainer = pl.Trainer(max_epochs=hparams["epochs"],
                         gpus=4, #2
                         auto_select_gpus=True,
                         strategy="ddp_find_unused_parameters_false",
                         logger=logger,
                         callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # ckpt_dir = os.path.join(disk_dir, "checkpoints/")
    # ckpt_file = os.path.join(ckpt_dir, f"406_pretrain_100k_save_memory_col_clr.ckpt")
    # trainer.save_checkpoint(ckpt_file)
    # trainer.logger.experiment.log_artifact(checkpoint_file, type='model')
