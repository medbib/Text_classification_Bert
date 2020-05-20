from train import Trainer
import torch

import logging
logging.basicConfig(level=logging.INFO)


PATH_TO_MODELS = "../artifacts/my_models/"


class Predictor:
    def __init__(self, name = "seq256_batch16_epoch30_gpu_lr2e_5_train"):
        logging.info("Loading the weights of the model..")
        trainer = Trainer()
        self.device = trainer.device
        self.tokenizer = trainer.train_data.tokenizer
        self.id_product2class = trainer.train_data.id_product2class
        self.max_seq_len = trainer.max_seq_len
        self.model = trainer.model
        self.name = name

        # self.PathToCheckpoint = trainer.PathToCheckpoint

    def predict(self, text):
        logging.info("Prediction in progress..")
        # Tokenize text
        # Encode input for Bert
        encoded_input = self.tokenizer.encode_plus(
                        text,                            # Sentence to encode.
                        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
                        max_length = self.max_seq_len,   # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,    # Construct attn. masks.
                        return_tensors = 'pt',           # Return pytorch tensors.
        )
        input_ids = encoded_input['input_ids']                #shape: [1, 256]
        token_type_ids = encoded_input['token_type_ids']
        attention_mask = encoded_input['attention_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)

        # Load weights to the model and predict
        checkpoint = torch.load(PATH_TO_MODELS + "checkpoint/" + self.name +"/model.pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        with torch.no_grad():
            proba = self.model(input_ids, token_type_ids, attention_mask)
            pred_class_num = proba.argmax(1)
            product = self.id_product2class[pred_class_num.cpu().numpy()[0]]
            print(product)
        logging.info("***** The complaint is about {} *****".format(product))
