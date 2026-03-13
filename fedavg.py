import argparse
import os
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from medclip import constants, MedCLIPModel, MedCLIPVisionModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.client import Client
from medclip.multi_fusion import MLPFusion_Mdoel
from medclip.sever import Server
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, ZeroShotImageDataset, \
    ZeroShotImageCollator


def get_train_dataloader(client_id, bs):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])],
    )
    train_data = ImageTextContrastiveDataset(datalist_path=datalist_path, dataset_path=dataset_path,
                                             imgtransform=None, client_id=client_id)
    train_collate_fn = ImageTextContrastiveCollator()
    train_dataloader = DataLoader(train_data,
                                  batch_size=bs,
                                  collate_fn=train_collate_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=2,
                                  )
    return train_dataloader


def get_valid_dataloader(client, data_type, type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    client=client,
                                    data_type=data_type)
    val_collate_fn = ZeroShotImageCollator(mode='multiclass', type=type)
    val_dataloader = DataLoader(val_data,
                                batch_size=50,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=2,
                                )
    return val_dataloader

class Runner:
    def __init__(self):
        # set the initial environment
        self.client_ids = None
        self.rounds = None
        self.clients = None
        self.server = None
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONASHSEED'] = str(seed)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        mp.set_start_method('spawn', force=True)


    def config(self, batch_size):
        self.client_ids = constants.CLIENT_IDS
        self.rounds = constants.ROUNDS
        client_nums = constants.SELECT_NUM
        global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        # global_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        log_file = constants.LOGFILE
        self.server = Server(global_model=global_model,
                             client_ids=self.client_ids,)
        self.batch_size = batch_size
    def train(self, fed, gpu):
        global client
        server = self.server
        device = f'cuda:{gpu}'
        writer = SummaryWriter(f'outputs/log/fl-train-{fed}')
        for r in range(self.rounds):
            print(f"round {r} / {self.rounds} is beginning! fed is {fed}")
            for client_id in self.client_ids:
                print(f"{client_id} is starting training!")
                train_dataloader = get_train_dataloader(client_id, self.batch_size)

                val_global_BA = get_valid_dataloader("no_client", 'BA', None)
                val_global_ASRb1_bd = get_valid_dataloader("no_client", 'ASRb_1', 'bd1')
                val_global_ASR1 = get_valid_dataloader("no_client", 'ASR_1', None)
                val_global_ASR1_bd = get_valid_dataloader("no_client", 'ASR_1', 'bd1')
                val_global_ASRb2_bd = get_valid_dataloader("no_client", 'ASRb_2', 'bd2')
                val_global_ASR2 = get_valid_dataloader("no_client", 'ASR_2', None)
                val_global_ASR2_bd = get_valid_dataloader("no_client", 'ASR_2', 'bd2')
                val_global_ASRb3_bd = get_valid_dataloader("no_client", 'ASRb_3', 'bd3')
                val_global_ASR3 = get_valid_dataloader("no_client", 'ASR_3', None)
                val_global_ASR3_bd = get_valid_dataloader("no_client", 'ASR_3', 'bd3')
                val_global_ASRb4_bd = get_valid_dataloader("no_client", 'ASRb_4', 'bd4')
                val_global_ASR4 = get_valid_dataloader("no_client", 'ASR_4', None)
                val_global_ASR4_bd = get_valid_dataloader("no_client", 'ASR_4', 'bd4')
                val_global_ASRb5_bd = get_valid_dataloader("no_client", 'ASRb_5', 'bd5')
                val_global_ASR5 = get_valid_dataloader("no_client", 'ASR_5', None)
                val_global_ASR5_bd = get_valid_dataloader("no_client", 'ASR_5', 'bd5')

                client = Client(client_id=client_id,
                                train_dataloader=train_dataloader,
                                val_global_BA=val_global_BA,
                                val_global_ASRb1_bd=val_global_ASRb1_bd,
                                val_global_ASR1=val_global_ASR1,
                                val_global_ASR1_bd=val_global_ASR1_bd,
                                val_global_ASRb2_bd=val_global_ASRb2_bd,
                                val_global_ASR2=val_global_ASR2,
                                val_global_ASR2_bd=val_global_ASR2_bd,
                                val_global_ASRb3_bd=val_global_ASRb3_bd,
                                val_global_ASR3=val_global_ASR3,
                                val_global_ASR3_bd=val_global_ASR3_bd,
                                val_global_ASRb4_bd=val_global_ASRb4_bd,
                                val_global_ASR4=val_global_ASR4,
                                val_global_ASR4_bd=val_global_ASR4_bd,
                                val_global_ASRb5_bd=val_global_ASRb5_bd,
                                val_global_ASR5=val_global_ASR5,
                                val_global_ASR5_bd=val_global_ASR5_bd,
                                round=r,
                                local_dict=server.global_model.state_dict(),
                                device=device,
                                fed=fed
                                )
                if client_id == 'client_1_0ToAll':
                    client.validate_global(writer)
                client.local_train()
                # diff_local = client.compute_diff(server.global_model, "global")
                local_dict = client.local_dict()
                server.receive(client_id=client_id,
                               model=local_dict,
                               model_type="global_model")
            server.aggregate()
            if r == self.rounds - 1:
                client.validate_global(writer)

def main():
    parser = argparse.ArgumentParser(description="FedPSM")
    parser.add_argument('--fed', type=str, default='fed_avg_0ToAll_6_P+G_1wbd_i', help='fed strategy')
    # parser.add_argument('--fed', type=str, default='fed_avg_6_5e-5_no_backdoor', help='fed strategy')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=48, help='gpu')
    args = parser.parse_args()
    runner = Runner()
    runner.config(args.batch_size)
    runner.train(args.fed, args.gpu)


if __name__ == "__main__":
    main()
