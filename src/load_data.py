"""
This is data load file
"""
import os
import pandas as pd
from datasets import load_dataset, table


class LoadDataset:
    """
    This class is using for data load
    """
    def __init__(self, hf_data_name, data_path="../data"):
        self.data_name = hf_data_name
        self.data_path = data_path

    def get_data_from_hf(self):
        """
        This function is using for getting data from huggingface
        :return:train, validation and test data
        """
        raw_data = load_dataset(self.data_name)
        dataset_train = raw_data.data["train"]
        dataset_val = raw_data.data["validation"]
        dataset_test = None

        if raw_data.data["test"]:
            dataset_test = raw_data.data["test"]

        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def get_dataframes(dataset_table: table.MemoryMappedTable) -> pd.DataFrame:
        """
        This function is using for getting data as dataframe
        :param dataset_table: this is datasets table
        :return: data as formatted DataFrame
        """
        df_dataset = pd.DataFrame({"id": dataset_table["id"],
                                   "title": dataset_table["title"],
                                   "context": dataset_table["context"],
                                   "question": dataset_table["question"],
                                   "answer_text": [str(item["text"][0]) for item in dataset_table["answers"]],
                                   "answer_start": [int(str(item["answer_start"][0])) for item in
                                                    dataset_table["answers"]]
                                   }
                                  )
        return df_dataset

    @staticmethod
    def add_end_idx(data_df):
        """
        This function is using for get the character position at which every answer ends and store it
        :param data_df: data to add end index
        :return: end index added data
        """

        list_end_idx = []
        for _, row in data_df.iterrows():
            context = row['context'].lower().strip()
            answer = row['answer_text'].lower().strip()
            start_idx = row['answer_start']
            end_idx = start_idx + len(answer)

            if context[start_idx:end_idx] == answer:
                list_end_idx.append(end_idx)
            elif context[start_idx - 1:end_idx - 1] == answer:
                row['answer_start'] = start_idx - 1
                list_end_idx.append(end_idx - 1)
            elif context[start_idx - 2:end_idx - 2] == answer:
                row['answer_start'] = start_idx - 2
                list_end_idx.append(end_idx - 2)
            else:
                list_end_idx.append(end_idx)

        data_df['answer_end'] = list_end_idx
        return data_df

    def get_train_val_test(self):
        """
        This function is using for getting train, validation and test data as DataFrame
        :return: train, validation and test dataframes
        """
        dataset_train, dataset_val, dataset_test = self.get_data_from_hf()
        train_df = self.get_dataframes(dataset_train)
        train_df = self.add_end_idx(train_df)
        self.export_data(train_df, "train")

        val_df = self.get_dataframes(dataset_val)
        val_df = self.add_end_idx(val_df)
        self.export_data(val_df, "val")

        test_df = None
        if dataset_test:
            test_df = self.get_dataframes(dataset_test)
            test_df = self.add_end_idx(test_df)
            self.export_data(test_df, "test")
        return train_df, val_df, test_df

    def export_data(self, export_data, file_name):
        """
        This file is using for export data to data file
        :param export_data: dataframe to export
        :param file_name: data name
        :return: None
        """
        export_data.to_csv(f"{self.data_path}/{file_name}.csv", index=False)

    def load_exported_data(self):
        """
        This function is using for load datasets
        :return: train, validation and test dataframes
        """
        train_df = pd.read_csv(f"{self.data_path}/train.csv")
        val_df = pd.read_csv(f"{self.data_path}/val.csv")

        test_file = f"{self.data_path}/test.csv"
        if os.path.isfile(test_file):
            test_df = pd.read_csv(f"{self.data_path}/test.csv")
            return train_df, val_df, test_df

        return train_df, val_df, None
