import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt

# from etl_ahmed import ETL

from transformers import BertTokenizer
import logging

logging.basicConfig(level=logging.DEBUG)


DATA_PATH = "../data"
CONFIG_PATH = "../misc/db_config.yaml"
SCHEMA_PATH = "../misc/schema_training.yaml"


class ComplaintsDataset(Dataset):
    def __init__(self, data_mode="train", 
                 path_to_train_data="../data/train_complaints_main_sub_product.csv", 
                 path_to_val_data="../data/val_complaints_main_sub_product.csv", 
                 path_to_test_data="../data/test_complaints_main_sub_product.csv", 
                 data_from_server=False,
                 classes=list((('product1', 'subproduct11'),
                               ('product1', 'subproduct12')
                               ('product2', 'subproduct21'), 
                               ('product2', 'subproduct22'),
                               ('product2', 'subproduct23'), 
                               ('product2', 'subproduct24'),
                               ('product3', 'product31'), 
                               ('product3', 'product32'))),
                 max_seq_len=256,
                 tokenizer='bert-base-uncased'):

        self.data_mode = data_mode
        self.max_seq_len = max_seq_len
        self.classes = classes
        self.classes_string = ["(" + ', '.join(tup) + ")" for tup in classes]

        self.path_to_train_data = path_to_train_data
        self.path_to_val_data = path_to_val_data
        self.path_to_test_data = path_to_test_data

        if data_from_server == True:
            try:
                # Get input data from server
                etl = ETL(DATA_PATH, CONFIG_PATH, SCHEMA_PATH)
                if self.data_mode=="train":
                    self.input_texts, self.target_main_products, self.target_sub_products = etl.get_data(table_name="train_complaints_main_sub_product")
                elif self.data_mode=="val":
                    self.input_texts, self.target_main_products, self.target_sub_products = etl.get_data(table_name="val_complaints_main_sub_product")
                elif self.data_mode=="test":
                    self.input_texts, self.target_main_products, self.target_sub_products = etl.get_data(table_name="test_complaints_main_sub_product")
            except:
                logging.info("Fetching data from server failed")
                logging.info("Loading a local copy of the data..")

                # open file csv file from computer
                df_train = pd.read_csv(self.path_to_train_data)
                df_val = pd.read_csv(self.path_to_val_data)
                df_test = pd.read_csv(self.path_to_test_data)

                if self.data_mode=="train":
                    self.input_texts, self.target_main_products, self.target_sub_products =  df_train['COMPLAINT_TEXT'].values, df_train['MAIN_PRODUCT'].values, df_train['SUB_PRODUCT'].values
                elif self.data_mode=="val":
                    self.input_texts, self.target_main_products, self.target_sub_products =  df_val['COMPLAINT_TEXT'].values, df_val['MAIN_PRODUCT'].values, df_val['SUB_PRODUCT'].values
                elif self.data_mode=="test":
                    self.input_texts, self.target_main_products, self.target_sub_products =  df_test['COMPLAINT_TEXT'].values, df_test['MAIN_PRODUCT'].values, df_test['SUB_PRODUCT'].values

        else:
            logging.info("Loading a local copy of the data..")
            # open file csv file from computer
            df_train = pd.read_csv(self.path_to_train_data)
            df_val = pd.read_csv(self.path_to_val_data)
            df_test = pd.read_csv(self.path_to_test_data)

            if self.data_mode=="train":
                self.input_texts, self.target_main_products, self.target_sub_products =  df_train['COMPLAINT_TEXT'].values, df_train['MAIN_PRODUCT'].values, df_train['SUB_PRODUCT'].values
            elif self.data_mode=="val":
                self.input_texts, self.target_main_products, self.target_sub_products =  df_val['COMPLAINT_TEXT'].values, df_val['MAIN_PRODUCT'].values, df_val['SUB_PRODUCT'].values
            elif self.data_mode=="test":
                self.input_texts, self.target_main_products, self.target_sub_products =  df_test['COMPLAINT_TEXT'].values, df_test['MAIN_PRODUCT'].values, df_test['SUB_PRODUCT'].values

        # Process targets to tensors
        #Create dictionaries to map distinct classes to distinct index numbers
        self.class2id_product = {}                 # Product class to ID dictionary
        for i, cls in enumerate(self.classes):
            self.class2id_product[cls] = i

        self.id_product2class = {}
        for key in self.class2id_product:
            self.id_product2class[self.class2id_product[key]] = key

        # Bert Tokenizer
        # Load tokenizer (vocabulary_id, truncate/pad, attention mask, type_of_id(padded or not))
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.input_texts)


    def __getitem__(self, idx):
        input_text = self.input_texts[idx]                       # Text input
        target_product = (self.target_main_products[idx], self.target_sub_products[idx])        # product tuple

        product_id = self.class2id_product[target_product]       # id of the product class

        # Encode input for Bert
        encoded_input = self.tokenizer.encode_plus(
                        input_text,                      # Sentence to encode.
                        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
                        max_length = self.max_seq_len,   # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,    # Construct attn. masks.
                        return_tensors = 'pt',           # Return pytorch tensors.
        )

        input_ids = encoded_input['input_ids']                #shape: [1, 256]
        token_type_ids = encoded_input['token_type_ids']
        attention_mask = encoded_input['attention_mask']

        return input_ids, token_type_ids, attention_mask, torch.tensor(product_id)


class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduction='elementwise_mean', num_classes=None):
        super(WeightedCrossEntropyLoss, self).__init__(weight, size_average, reduction)
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, input, target):
        lst_class_num = []
        for i in range(self.num_classes):
            lst_class_num.append(torch.sum(target == i).item())

        weight = self.num_classes * [1]
        for i in range(self.num_classes):
            if lst_class_num[i] != 0:
                weight[i] = 1/lst_class_num[i]
            else:
                weight[i] = 0
        weight = np.asarray(weight, dtype=np.float32)
        if torch.cuda.is_available():
            self.weight = torch.from_numpy(weight).cuda()
        else:
            self.weight = torch.from_numpy(weight)
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,
                              cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    # Calculate chart area size
    leftmargin = 0.5 # inches
    rightmargin = 0.5 # inches
    categorysize = 0.5 # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)           

    f = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = f.add_subplot(111)
    ax.set_aspect(1)
    f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar(res)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    f.tight_layout()
    return f


def plot_f1(scores, fig_num):
    """
        This function take predictions and ground-truths as input and returns Precision,
        Recall, F1 scores ana also plots.
    """
    classes = scores.keys()

    # F1 score
    fig_f1 = plt.figure(fig_num)
    for label in classes:
        plt.plot(range(len(scores[label]["f1-score"])), scores[label]["f1-score"])
    plt.title("F1 score")
    plt.legend(classes)

    return fig_f1


def get_perf(gts, preds, classes, scores_cache=None):
    """
        This function calculates the precision, recall and F1-score for the
        given ground-truths and predictions
    """

    classes = list(classes)
    labels = [c for c in range(len(classes))]
    report = classification_report(gts, preds,labels=labels, target_names=classes, output_dict=True)

    precision = ["Precision:"]
    recall = ["Recall:"]
    f_score = ["F-score:"]

    for cls in ["macro avg"] + classes:
        precision.append("({}, {:.2f})".format(cls, report[cls]['precision']))
        recall.append("({}, {:.2f})".format(cls, report[cls]['recall']))
        f_score.append("({}, {:.2f})".format(cls, report[cls]['f1-score']))

        if scores_cache is not None:

            if cls in scores_cache:
                scores_cache[cls]["precision"].append(report[cls]["precision"])
                scores_cache[cls]["recall"].append(report[cls]["recall"])
                scores_cache[cls]["f1-score"].append(report[cls]["f1-score"])

            else:
                scores_cache[cls] = {
                    "precision": [report[cls]["precision"]],
                    "recall": [report[cls]["recall"]],
                    "f1-score": [report[cls]["f1-score"]]
                }

    f_score = " ".join(f_score)
    precision = " ".join(precision)
    recall = " ".join(recall)

    return precision, recall, f_score, scores_cache
