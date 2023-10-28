import config
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from typing import Any
from util import split



class Preprocess:
    def __init__(self, train_data_path: str, test_data_path: str) -> None:
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        if os.path.exists(train_data_path) and os.path.exists(test_data_path):
                # train
                self.train_data = pd.read_parquet(train_data_path)
                self.features = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
                self.labels_train = self.train_data.drop(columns = self.features)
                self.features_train = self.train_data.iloc[:, :2]
                self.positive_index = self.train_data[self.train_data['control'] == True].index

                # test
                self.id_map = pd.read_csv(test_data_path)
        else:
            raise FileNotFoundError(f"File not exists!")
        

    def tokenize(self, hidden_size: int = 128, vocab_size: int = 0):
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

        # train
        features_train_encoded = pd.DataFrame({'cell_type_encoded':ct,'sm_name_encoded':sm})

        # test
        df = pd.concat([features_train_encoded, self.features_train],axis=1)
        cell_type_mapping = df.groupby(['cell_type_encoded', 'cell_type']).size().reset_index(name='count')
        sm_name_mapping = df.groupby(['sm_name_encoded', 'sm_name']).size().reset_index(name='count')
        cell_type_mapping_dict = dict(zip(cell_type_mapping['cell_type'], cell_type_mapping['cell_type_encoded']))
        sm_name_mapping_dict = dict(zip(sm_name_mapping['sm_name'], sm_name_mapping['sm_name_encoded']))
        self.id_map['cell_type'] = self.id_map['cell_type'].map(cell_type_mapping_dict)
        self.id_map['sm_name'] = self.id_map['sm_name'].map(sm_name_mapping_dict)
        features_test_encoded = self.id_map.iloc[:, 1:]
        self.test_data = features_test_encoded.copy()


        # step 2: combine two columns to get the union-id "2" + "116"-> "2116" 
        features_train_encoded["union_id"] = features_train_encoded["cell_type_encoded"].astype(str) + features_train_encoded["sm_name_encoded"].astype(str)
        features_train_encoded["union_id"] = features_train_encoded["union_id"].astype(int)

        features_test_encoded["union_id"] = features_test_encoded["cell_type"].astype(str) + features_test_encoded["sm_name"].astype(str)
        features_test_encoded["union_id"] = features_test_encoded["union_id"].astype(int)

        # step 3: generate the embedding based on the union_id
        if vocab_size == 0:
            vocab_size = max(features_train_encoded["union_id"]) + 2
        else:
            assert vocab_size > max(features_train_encoded["union_id"]) + 2

        hidden_size = self.labels_train.shape[1]

        emb_func = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        train_embedding = emb_func(torch.LongTensor(features_train_encoded["union_id"]).reshape(-1, 1))
        test_embedding = emb_func(torch.LongTensor(features_test_encoded["union_id"]).reshape(-1, 1))

        return train_embedding.squeeze(1).detach().numpy(), test_embedding.squeeze(1).detach().numpy()
    
    def expand(self):
        '''
        use postive different expressional data to expand
        '''
        de_b, de_myeloid, de_nk, de_cd4, de_cd8, de_regt = 0, 0, 0, 0, 0, 0
        de = {}

        for id in self.positive_index:
            if self.train_data["cell_type"][id] == "B cells":
                de_b += self.labels_train.iloc[id, :].values / 2
                de["B cells"] = de_b
            elif self.train_data["cell_type"][id] == "Myeloid cells":
                de_myeloid += self.labels_train.iloc[id, :].values / 2
                de["Myeloid cells"] = de_myeloid
            elif self.train_data["cell_type"][id] == "NK cells":
                de_nk += self.labels_train.iloc[id, :].values / 2
                de["NK cells"] = de_nk
            elif self.train_data["cell_type"][id] == "T cells CD4+":
                de_cd4 += self.labels_train.iloc[id, :].values / 2
                de["T cells CD4+"] = de_cd4
            elif self.train_data["cell_type"][id] == "T cells CD8+":
                de_cd8 += self.labels_train.iloc[id, :].values / 2
                de["T cells CD8+"] = de_cd8
            elif self.train_data["cell_type"][id] == "T regulatory cells":
                de_regt += self.labels_train.iloc[id, :].values / 2
                de["T regulatory cells"] = de_regt
            else:
                raise ValueError("no such cell type")

        self.expand_train_data = self.train_data.copy()
        self.expand_test_data = self.test_data.copy()
        self.expand_test_data = pd.concat((self.expand_test_data, pd.DataFrame(np.random.randn(self.expand_test_data.shape[0], self.labels_train.shape[1]))), axis=1)
        cell_types = [types for types in self.expand_train_data["cell_type"].astype("category").cat.categories]
        for cell_type in cell_types:
            indices_train = self.expand_train_data[self.expand_train_data["cell_type"] == cell_type].index
            self.expand_train_data.iloc[indices_train, 5:] = de[cell_type]
            if cell_type in ["B cells", "Myeloid cells"]:
                indices_test = self.expand_test_data[self.expand_test_data["cell_type"] == cell_type].index
                self.expand_test_data.iloc[indices_test, 2:] = de[cell_type]
            

        self.expand_de = self.expand_train_data.drop(columns=self.features)
        self.text_de = self.expand_test_data.drop(columns=["cell_type", "sm_name"])
        return self.expand_de.to_numpy(), self.text_de.to_numpy()

    
    def split(self, split_size, loop):
        sub_input, column_name = split(self.expand_de, config.SPLIT_SIZE, loop)
        return sub_input, column_name

    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
