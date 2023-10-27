import config
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from typing import Any



class Preprocess:
    def __init__(self, train_data_path: str, test_data_path: str) -> None:
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        if os.path.exists(train_data_path) and os.path.exists(test_data_path):
                # train
                self.train_data = pd.read_parquet(train_data_path)
                features = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
                self.labels_train = self.train_data.drop(columns = features)
                self.features_train = self.train_data.iloc[:, :2]

                # test
                self.id_map = pd.read_csv(test_data_path)
        else:
            raise FileNotFoundError(f"File not exists!")
        

    def tokenize(self, vocab_size: None, hidden_size: int = 128):
        '''
        :param vocab_size: the number of the cell/compound pairs
        :param hidden_size: the hidden_size of model
        :return tokenized embedding of each cell/compound pair
        '''
        # map the cell_type and compound name to inputs_index
        # step 1: LabelEncoder--get the encode label of cell_type and sm_name
        if self.features_train is None:
            raise ValueError("please process the train_data first to get the tokenizer")

        le = LabelEncoder()
        ct = le.fit_transform(self.features_train['cell_type'])
        sm = le.fit_transform(self.features_train['sm_name'])

        # step 1: # train
        features_train_encoded = pd.DataFrame({'cell_type_encoded':ct,'sm_name_encoded':sm})

        # step 1: # test
        df = pd.concat([features_train_encoded, self.features_train],axis=1)
        cell_type_mapping = df.groupby(['cell_type_encoded', 'cell_type']).size().reset_index(name='count')
        sm_name_mapping = df.groupby(['sm_name_encoded', 'sm_name']).size().reset_index(name='count')
        cell_type_mapping_dict = dict(zip(cell_type_mapping['cell_type'], cell_type_mapping['cell_type_encoded']))
        sm_name_mapping_dict = dict(zip(sm_name_mapping['sm_name'], sm_name_mapping['sm_name_encoded']))
        self.id_map['cell_type'] = self.id_map['cell_type'].map(cell_type_mapping_dict)
        self.id_map['sm_name'] = self.id_map['sm_name'].map(sm_name_mapping_dict)
        features_test_encoded = self.id_map.iloc[:, 1:]


        # step 2: combine two columns to get the union-id "2" + "116"-> "2116"
        features_train_encoded["union_id"] = features_train_encoded["cell_type_encoded"].astype(str) + features_train_encoded["sm_name_encoded"].astype(str)
        features_train_encoded["union_id"] = features_train_encoded["union_id"].astype(int)

        features_test_encoded["union_id"] = features_test_encoded["cell_type"].astype(str) + features_test_encoded["sm_name"].astype(str)
        features_test_encoded["union_id"] = features_test_encoded["union_id"].astype(int)

        # step 3: generate the embedding based on the union_id
        if vocab_size is None:
            vocab_size = max(len(features_train_encoded["union_id"])) + 2
        else:
            assert vocab_size > max(len(features_train_encoded["union_id"])) + 2

        emb_func = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        train_embedding = emb_func(torch.LongTensor(features_train_encoded["union_id"]).reshape(-1, 1))
        test_embedding = emb_func(torch.LongTensor(features_test_encoded["union_id"]).reshape(-1, 1))

        return train_embedding.squeeze(1).detach(), test_embedding.squeeze(1).detach()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass