import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import sys
import argparse
import math

from utils.datasets import RockfishDataModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=114):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.layers = nn.Sequential(
                     nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                     nn.GELU(),
                     nn.InstanceNorm1d(out_channels, affine=True))

    def forward(self, x):
        return self.layers(x)


class Rockfish(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)

        self.ke = nn.Embedding(4, 2, max_norm=1.)

        self.conv1 = ConvBlock(3, 256, 13, stride=3, padding=6)
        self.conv2 = ConvBlock(256, 256, 7, stride=1, padding=3)
        self.conv3 = ConvBlock(256, 256, 3, stride=2, padding=1)

        self.pos_encoder = PositionalEncoding(256, self.hparams.dropout)

        encoder_layer = nn.TransformerEncoderLayer(256, self.hparams.nhead, self.hparams.dim_ff, self.hparams.dropout,
                                                   activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, self.hparams.nlayers)

        self.fc1 = nn.Linear(256, 1)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)

    def forward(self, x, ref_k):
        ref_k = self.ke(ref_k).transpose(1, 2)

        x = torch.unsqueeze(x, 1)
        x = torch.cat((x, ref_k), 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x.permute(1, 2, 0)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)

        return self.fc1(x)

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dim_ff', type=int, default=1024)
        parser.add_argument('--nlayers', type=int, default=6)

        parser.add_argument('--wd', type=float, default=1e-4)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--step_size_up', type=int, default=None)

        return parser

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.wd, eps=1e-6,
                                betas=(0.9, 0.98))

        if self.hparams.step_size_up is not None:
            step_size_up = self.hparams.step_size_up
        else:
            steps_per_epoch = self.train_ds_len // self.effective_batch_size
            step_size_up = 4 * steps_per_epoch

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, self.hparams.lr / 10,
                                self.hparams.lr, step_size_up=step_size_up,
                                mode='triangular2', cycle_momentum=False)

        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, ref_k, y = batch

        out = self(x, ref_k).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(out, y.float())

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_batch_acc', self.train_acc(torch.sigmoid(out), y))

        return loss

    def training_epoch_end(self, outputs):
        self.log('train_epoch_acc', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, ref_k, y = batch

        out = self(x, ref_k).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(out, y.float())

        self.val_acc(torch.sigmoid(out), y)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc.compute()
        self.log('val_acc', val_acc, prog_bar=True)


def main(args):
    gpu_devices = torch.cuda.device_count()
    if gpu_devices == 0 or gpu_devices == 1:
        accelerator = None
        effective_batch_size = args.train_batch_size
    else:
        accelerator = 'ddp'
        effective_batch_size = gpu_devices * args.train_batch_size
    print(f'GPU devices: {gpu_devices}, accelerator: {accelerator}', file=sys.stderr)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{epoch}-acc:{val_acc:.5f}',
        save_top_k=-1)

    model = Rockfish(args)
    model.effective_batch_size = effective_batch_size

    data = RockfishDataModule.from_argparse_args(args)
    data.prepare_data()
    data.setup('train')
    model.train_ds_len = data.train_len

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=gpu_devices,
        max_epochs=args.epochs,
        default_root_dir='rockfish_train',
        accelerator=accelerator,
        precision=16 if gpu_devices > 0 else 32,
        gradient_clip_val=2.0,
        benchmark=True)
    trainer.fit(model, datamodule=data)


def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=25)

    parser = Rockfish.add_module_specific_args(parser)
    parser = RockfishDataModule.add_argparse_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = create_args()

    main(arguments)
