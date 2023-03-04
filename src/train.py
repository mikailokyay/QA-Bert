"""
This is Question Answering trainer function
"""
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering

from src.load_data import LoadDataset


class SquadDataset(torch.utils.data.Dataset):
    """
    This class is using for getting torch dataset
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class QATrainer:
    """
    This is training class
    """
    def __init__(self, model_name):

        self.train_df, self.val_df, self.test_df = LoadDataset("squad", "../data")

        if self.test_df is None:
            self.test_df = self.val_df

        self.set_seed(seed=82)

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            print('Training on GPU.')
            self.device = 'cuda'
        else:
            print('No GPU available, training on CPU.')
            self.device = 'cpu'

        self.model = BertForQuestionAnswering.from_pretrained(model_name).to(device=self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    @staticmethod
    def set_seed(seed=1234):
        """
        Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.
        :return: None
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

    @staticmethod
    def add_token_positions(encodings, data_df, tokenizer):
        """
        This function is using for convert character start/end positions to token start/end positions.
        :param encodings: data encodings
        :param data_df: data
        :param tokenizer: tokenizer
        :return: encodings
        """

        start_positions = []
        end_positions = []
        for i in range(len(data_df)):
            start_positions.append(encodings.char_to_token(i, data_df.loc[i, 'answer_start']))
            if data_df.loc[i, 'answer_end'] == 0:
                end_positions.append(encodings.char_to_token(i, data_df.loc[i, 'answer_end']))
            else:
                end_positions.append(encodings.char_to_token(i, data_df.loc[i, 'answer_end'] - 1))

            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length

            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    def train(self, params):
        """
        This is QA training function
        :param params: training parameters
        :return: model
        """

        train_encodings = self.tokenizer(
            list(self.train_df['context']),
            list(self.train_df['question']),
            truncation=True,
            padding=True,
            max_length=512)

        val_encodings = self.tokenizer(
            list(self.val_df['context']),
            list(self.val_df['question']),
            truncation=True,
            padding=True,
            max_length=params["max_length"])

        train_encodings = self.add_token_positions(train_encodings, self.train_df, self.tokenizer)
        val_encodings = self.add_token_positions(val_encodings, self.val_df, self.tokenizer)

        # create the corresponding datasets
        train_set = SquadDataset(train_encodings)
        val_set = SquadDataset(val_encodings)

        train_dataloader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)
        model = self.fit_model(train_dataloader, val_dataloader, params)
        return model

    def fit_model(self, train_dataloader, val_dataloader, params):
        """
        This is model fit function for training
        :param train_dataloader: training dataloader
        :param val_dataloader: validation dataloader
        :param params: training parameters
        :return: fine-tuned model
        """
        model = self.model
        tokenizer = self.tokenizer
        train_losses = []
        val_losses = []

        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])

        for i in range(params["epoch"]):
            batch_losses = []
            model.train()  # training mode

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()  # Delete previously stored gradients
                batch.to(self.device)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                start_positions = batch['start_positions']
                end_positions = batch['end_positions']

                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions
                                )

                batch_losses.append(outputs[0].item())

                outputs[0].backward()

                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            train_current_loss = sum(batch_losses) / len(train_dataloader)
            train_losses.append(train_current_loss)

            with torch.no_grad():
                model.eval()
                batch_losses = []
                for batch in tqdm(val_dataloader):
                    batch.to(self.device)
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    start_positions = batch['start_positions']
                    end_positions = batch['end_positions']

                    output = model(input_ids,
                                   attention_mask=attention_mask,
                                   start_positions=start_positions,
                                   end_positions=end_positions
                                   )
                    batch_losses.append(output[0].item())

            val_current_loss = sum(batch_losses) / len(val_dataloader)
            val_losses.append(val_current_loss)

            path = f"{params['output_dir']}/" \
                   f"bert-epoch-{i}-train-loss-{str(train_current_loss)}" \
                   f"-val-loss-{str(val_current_loss)} "
            if params["save_only_last_epoch"]:
                if i == params["epoch"] - 1:
                    tokenizer.save_pretrained(path)
                    model.save_pretrained(path)
            else:
                tokenizer.save_pretrained(path)
                model.save_pretrained(path)

            print("-Epoch: {}/{}...".format(i + 1, params["epoch"]),
                  "Train Loss: {:.6f}".format(train_current_loss),
                  "Val Loss: {:.6f}".format(val_current_loss)
                  )

        return model
