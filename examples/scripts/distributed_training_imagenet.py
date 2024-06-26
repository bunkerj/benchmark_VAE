import argparse
import logging
import os
import time

import hostlist
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.pythae.data.datasets import DatasetOutput
from src.pythae.models import VQVAE, VQVAEConfig
from src.pythae.models.base.base_utils import ModelOutput
from src.pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from src.pythae.models.nn.benchmarks.utils import ResBlock
from src.pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="imagenet-distributed",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="pythae",
)

args = ap.parse_args()


class Encoder_ResNet_VQVAE_ImageNet(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 4, 2, padding=1),
            nn.Conv2d(32, 64, 4, 2, padding=1),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.Conv2d(128, 128, 4, 2, padding=1),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
        )

        self.pre_qantized = nn.Conv2d(128, self.latent_dim, 1, 1)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()
        out = x
        out = self.layers(out)
        output["embedding"] = self.pre_qantized(out)

        return output


class Decoder_ResNet_VQVAE_ImageNet(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)

        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.dequantize = nn.ConvTranspose2d(self.latent_dim, 128, 1, 1)

        self.layers = nn.Sequential(
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            nn.ConvTranspose2d(128, 128, 4, 2, padding=1),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.ConvTranspose2d(32, self.n_channels, 4, 2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        out = self.dequantize(z)
        output["reconstruction"] = self.layers(out)

        return output


class ImageNet(Dataset):
    def __init__(self, data_dir=None, transforms=None):
        self.imgs_path = [os.path.join(data_dir, n) for n in os.listdir(data_dir)]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return DatasetOutput(data=img)


def main(args):

    img_transforms = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    train_dataset = ImageNet(
        data_dir="/gpfsscratch/rech/wlr/uhw48em/data/imagenet/train",
        transforms=img_transforms,
    )
    eval_dataset = ImageNet(
        data_dir="/gpfsscratch/rech/wlr/uhw48em/data/imagenet/val",
        transforms=img_transforms,
    )

    model_config = VQVAEConfig(
        input_dim=(3, 128, 128), latent_dim=128, use_ema=True, num_embeddings=1024
    )

    encoder = Encoder_ResNet_VQVAE_ImageNet(model_config)
    decoder = Decoder_ResNet_VQVAE_ImageNet(model_config)

    model = VQVAE(model_config=model_config, encoder=encoder, decoder=decoder)

    gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")

    training_config = BaseTrainerConfig(
        num_epochs=20,
        train_dataloader_num_workers=8,
        eval_dataloader_num_workers=8,
        output_dir="my_models_on_imagenet",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-4,
        steps_saving=None,
        steps_predict=None,
        no_cuda=False,
        world_size=int(os.environ["SLURM_NTASKS"]),
        dist_backend="nccl",
        rank=int(os.environ["SLURM_PROCID"]),
        local_rank=int(os.environ["SLURM_LOCALID"]),
        master_addr=hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0],
        master_port=str(12345 + int(min(gpu_ids))),
    )

    if int(os.environ["SLURM_PROCID"]) == 0:
        logger.info(model)
        logger.info(f"Training config: {training_config}\n")

    callbacks = []

    # Only log to wandb if main process
    if args.use_wandb and (training_config.rank == 0 or training_config == -1):
        from src.pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        callbacks.append(wandb_cb)

    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        callbacks=callbacks,
    )

    start_time = time.time()

    trainer.train()

    end_time = time.time()

    logger.info(f"Total execution time: {(end_time - start_time)} seconds")


if __name__ == "__main__":

    main(args)
