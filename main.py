import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import ComplaintsDataset, WeightedCrossEntropyLoss, plot_confusion_matrix
from train import Trainer
from predict import Predictor

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--train', help="To train the model", action='store_true')
parser.add_argument('--predict', help="Enter a text to predict about which product it is about ", default=None)
parser.add_argument('--PathToModels', help="Path to save the model", default="../artifacts/my_models/")
parser.add_argument('--name', help="Name of the experiment", default="seq256_batch16_epoch30_gpu_lr2e_5_train")
parser.add_argument('--data_from_server', help="Fetch data from server. If False, load a local copy", default=False)
parser.add_argument('--max_seq_len', help="Maximum sequence length to be used", type=int, default=256)
parser.add_argument('--num_epochs', help="No. of epochs of training", type=int, default=30)
parser.add_argument('--batch_size', help="Batch size", type=int, default=16)
parser.add_argument('--lr', help="Learning rate", type=float, default=2e-5)
parser.add_argument('--lr_scheduler', help="Learning rate scheduler", type=bool, default=True)
parser.add_argument('--val_every', help="Validation every _ epochs ?", default=1)
parser.add_argument('--num_workers', help="No. of workers in data loader", default=1)
parser.add_argument('--restore_checkpoint', help="Flag whether checkpoint has to be restored", default=True)


args = parser.parse_args()

if __name__=='__main__':

    # Parse the input arguments
    train = args.train
    text = args.predict     # "I have a problem with my bank account"
    name = args.name
    data_from_server = args.data_from_server
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    val_every = args.val_every
    num_workers = args.num_workers
    restore_checkpoint = args.restore_checkpoint


    if train == True and text is None:
        logging.info("*****  Training process  *****")
        training = Trainer(restore_checkpoint, data_from_server,num_workers, name, max_seq_len, batch_size, num_epochs, lr_scheduler, lr, val_every)
        training.train()
        # Evaluate and save results of the model for the training set in result folder
        training.evaluate(training.train_data_loader, "training")
        # Evaluate and save results of the model for the validation set in result folder
        training.evaluate(training.train_data_loader, "val")
        # Evaluate and save results of the model for the test set in result folder
        training.evaluate(training.train_data_loader, "test")

    elif train == False and text is not None:
        logging.info("*****  Prediction process  *****")
        predictor = Predictor(name)
        predictor.predict(text)

    else:
        raise ValueError("You have to make a choice between train or predict arguments")
