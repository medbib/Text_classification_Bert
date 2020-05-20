import numpy as np
import os
import json
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import BertComplaint2Product
from utils import ComplaintsDataset, WeightedCrossEntropyLoss, plot_confusion_matrix, plot_f1, get_perf

from transformers import get_linear_schedule_with_warmup, AdamW

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)

PATH_TO_MODELS = "../artifacts/my_models/"


class Trainer():
    def __init__(self, restore_checkpoint=True, data_from_server=False, num_workers=1, name="seq256_batch16_epoch30_gpu_lr2e_5_train",
                        max_seq_len=256, batch_size=16, num_epochs=30, lr_scheduler=True, lr=2e-5, val_every=1):
        self.restore_checkpoint = restore_checkpoint
        self.num_workers = num_workers
        self.name = name
        # Creating directories if they don't exist already.
        self.PathToCheckpoint = PATH_TO_MODELS + "/checkpoint/" + name + "/"
        if not os.path.exists(self.PathToCheckpoint):
            os.makedirs(self.PathToCheckpoint)

        self.PathToTB = PATH_TO_MODELS + "/log/" + name + "/"
        if not os.path.exists(self.PathToTB):
            os.makedirs(self.PathToTB)

        self.PathToResults = PATH_TO_MODELS + "/results/" + name + "/"
        if not os.path.exists(self.PathToResults):
            os.makedirs(self.PathToResults)

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.val_every = val_every
        # CUDA device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.num_workers > 0) else "cpu")

        self.train_data = ComplaintsDataset(data_mode="train", max_seq_len=self.max_seq_len, data_from_server=data_from_server)
        self.val_data = ComplaintsDataset(data_mode="val", max_seq_len=self.max_seq_len, data_from_server=data_from_server)
        self.test_data = ComplaintsDataset(data_mode="test", max_seq_len=self.max_seq_len, data_from_server=data_from_server)

        self.train_data_loader = DataLoader(self.train_data, batch_size=self.batch_size, 
                                                shuffle=True, drop_last=True, 
                                                num_workers=self.num_workers)
        self.validation_data_loader = DataLoader(self.val_data, batch_size=self.batch_size, 
                                                shuffle=True, drop_last=True, 
                                                num_workers=self.num_workers)
        self.test_data_loader = DataLoader(self.test_data, batch_size=self.batch_size, 
                                                shuffle=True, drop_last=True, 
                                                num_workers=self.num_workers)

        self.train_writer = SummaryWriter(log_dir=self.PathToTB + "train/")
        self.val_writer = SummaryWriter(log_dir=self.PathToTB + "val/")

        self.model = BertComplaint2Product(num_labels=len(self.train_data.classes))
        self.model.to(self.device)
        # self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = WeightedCrossEntropyLoss(num_classes=len(self.train_data.classes))
        if lr_scheduler:
            # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
            # I believe the 'W' stands for 'Weight Decay fix"
            self.optimizer = AdamW(self.model.parameters(),
                                lr = lr,
                                eps = 1e-8 #-default is 1e-8
                                )
            total_steps = len(self.train_data_loader) * num_epochs
            # Create the learning rate scheduler.
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                      num_warmup_steps = 0, # Default value in run_glue.py
                                                      num_training_steps = total_steps)
        else:
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

    def train(self):
        # Training
        scores = {}
        val_scores = {}
        prev_epoch = 0
        best_val_loss = np.infty

        # Restore checkpoint if it exists
        if self.restore_checkpoint and os.path.exists(self.PathToCheckpoint + "model.pt"):
            print("Previous model found, restoring from checkpoint!")
            checkpoint = torch.load(self.PathToCheckpoint + "model.pt")
            prev_epoch = checkpoint["epoch"] + 1
            scores = checkpoint["scores"]
            val_scores = checkpoint["val_scores"]
            best_val_loss = checkpoint["best_val_loss"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for i in range(prev_epoch, prev_epoch + self.num_epochs):
            self.model.train()

            epoch_loss = 0.0
            epoch_accuracy = 0.0

            epoch_targets = []
            epoch_preds = []

            logging.info("------------------- epoch {} -------------------".format(i))

            for step, batch in enumerate(self.train_data_loader):
                if step%1000 == 0:
                    logging.info("step {}/{} of epoch {}".format(step, len(self.train_data_loader), i))

                input_ids, token_type_ids, attention_mask, target = batch

                # Move variables to gpu
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                proba = self.model(input_ids, token_type_ids, attention_mask)

                pred_class_num = proba.argmax(1)

                loss = self.criterion(proba, target)

                accuracy = torch.sum(torch.eq(pred_class_num, target), dtype=torch.float32) / target.shape[0]

                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.lr_scheduler:
                    self.scheduler.step()

                epoch_loss += loss.item()

                epoch_accuracy += accuracy.item()

                epoch_targets.append(target.cpu().numpy())

                epoch_preds.append(pred_class_num.cpu().numpy())

            epoch_targets = np.concatenate(epoch_targets, axis=0)
            epoch_preds = np.concatenate(epoch_preds, axis=0)

            _, _, _, scores = get_perf(epoch_targets, epoch_preds, self.train_data.classes, scores)
            fig_f = plot_f1(scores, 1)

            fig = plot_confusion_matrix(
                y_true=epoch_targets, y_pred=epoch_preds,
                classes=np.array(self.train_data.classes_string), normalize=True)

            # Monitoring loss.
            self.train_writer.add_scalar('Training Loss', epoch_loss / len(self.train_data_loader), i)
            self.train_writer.add_scalar('Training Accuracy', epoch_accuracy / len(self.train_data_loader), i)
            self.train_writer.add_figure("Training F1-score", fig_f, i)
            self.train_writer.add_figure("Training Confusion Matrix", fig, i)

            # Predict on Dev set
            if i % self.val_every == 0:
                logging.info("validation of the epoch {}".format(i))
                val_loss, val_scores = self.dev(i, val_scores)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("Validation loss improved, :D")
                    torch.save({
                        "epoch": i,
                        "scores": scores,
                        "val_scores": val_scores,
                        "best_val_loss": best_val_loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, self.PathToCheckpoint + "model.pt")


    def dev(self, epoch, val_scores):
        # Validation.
        self.model.eval()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_targets = []
        epoch_preds = []

        with torch.no_grad():
            for batch in self.validation_data_loader:
                input_ids, token_type_ids, attention_mask, target = batch

                # Move variables of interest in gpu
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                target = target.to(self.device)

                proba = self.model(input_ids, token_type_ids, attention_mask)
                pred_class_num = proba.argmax(1)

                loss = self.criterion(proba, target)
                pred_class_num = pred_class_num.cpu()
                target = target.cpu()
                accuracy = torch.sum(torch.eq(pred_class_num, target), dtype=torch.float32) / target.shape[0]
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                epoch_targets.append(target.numpy())
                epoch_preds.append(pred_class_num.numpy())

        # Monitoring loss
        val_loss = epoch_loss / len(self.validation_data_loader)
        val_accuracy = epoch_accuracy / len(self.validation_data_loader)
        epoch_targets = np.concatenate(epoch_targets, axis=0)

        epoch_preds = np.concatenate(epoch_preds, axis=0)

        _, _, val_f_score, val_scores = get_perf(epoch_targets, epoch_preds, self.val_data.classes, val_scores)
        # print("val_f_score", val_f_score)
        # print("val_scores", val_scores)
        val_fig_f = plot_f1(val_scores, 2)

        val_fig = plot_confusion_matrix(y_true=epoch_targets,
                                        y_pred=epoch_preds,
                                        classes=np.array(self.val_data.classes_string), 
                                        normalize=True)

        self.val_writer.add_scalar("Validation Loss", val_loss, epoch)
        self.val_writer.add_figure("Validation F1-score", val_fig_f, epoch)
        self.val_writer.add_figure("Validation Confusion Matrix", val_fig, epoch)
        self.val_writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

        return val_loss, val_scores


    def evaluate(self, data_loader, result_filename):
        checkpoint = torch.load(PATH_TO_MODELS + "checkpoint/" + self.name +"/model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        scores = {}
        targets = []
        preds = []
        accuracy_epoch = 0.0

        with torch.no_grad():
            for batch in data_loader:
                input_ids, token_type_ids, attention_mask, target = batch

                # Move variables of interest in gpu
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                target = target.to(self.device)

                proba = self.model(input_ids, token_type_ids, attention_mask)         # shape [16, 30]
                pred_class_num = proba.argmax(1)                                      # shape [16]

                accuracy = torch.sum(torch.eq(pred_class_num, target), dtype=torch.float32) / target.shape[0]                                                       # Shape: (B*T) x .

                accuracy_epoch += accuracy.item()
                preds.append(pred_class_num.cpu().numpy())
                targets.append(target.cpu().numpy())

        # Concatenate preds and targets over the entire dataset
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        accuracy = accuracy_epoch / len(data_loader)
        _, _, _, scores = get_perf(targets, preds, self.train_data.classes, scores)

        fig = plot_confusion_matrix(targets, preds, np.array(self.train_data.classes_string), True)
        plt.savefig(self.PathToResults + result_filename + "_confmat.png", dpi=300)
        plt.close(fig)

        f = open(self.PathToResults + result_filename + "_perf.json", "w")
        d = {
            "preds": list(preds),
            "targets": list(targets),
            "accuracy": accuracy,
            "scores": scores
        }
        f.write(str(d))
        f.close()
