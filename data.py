import os
import config
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dataset:
    labelencoder = LabelEncoder()

    def __init__(self, *, train_dataset_name = './data/de_train.parquet', test_dataset_name = "./data/id_map.csv") -> None:
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name

        if os.path.exists(train_dataset_name):
            self.train_data = pd.read_parquet(self.train_dataset_name)
            drop_features = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
            self.labels = self.train_data.drop(columns = drop_features).iloc[:, 1:]
            self.X = self.train_data.iloc[:, :2]
        else:
            raise FileNotFoundError("File Not exists")
        
        if os.path.exists(test_dataset_name):
            self.test_data = pd.read_csv(test_dataset_name).iloc[:, 1:]
        else:
            raise FileNotFoundError("File Not exists")
    
    @property
    def train_test_split(self):
        # train
        ct = Dataset.labelencoder.fit_transform(self.X['cell_type'])
        sm = Dataset.labelencoder.fit_transform(self.X['sm_name'])
        self.X_encoded = pd.DataFrame({'cell_type_encoded':ct,'sm_name_encoded':sm})

        # test
        df = pd.concat([self.X_encoded, self.X], axis=1)
        cell_type_mapping = df.groupby(['cell_type_encoded', 'cell_type']).size().reset_index(name='count')
        sm_name_mapping = df.groupby(['sm_name_encoded', 'sm_name']).size().reset_index(name='count')
        cell_type_mapping_dict = dict(zip(cell_type_mapping['cell_type'], cell_type_mapping['cell_type_encoded']))
        sm_name_mapping_dict = dict(zip(sm_name_mapping['sm_name'], sm_name_mapping['sm_name_encoded']))
        test_encoded = {}
        test_encoded['cell_type'] = self.test_data['cell_type'].map(cell_type_mapping_dict)
        test_encoded["sm_name"] = self.test_data['sm_name'].map(sm_name_mapping_dict)
        self.test_encoded = pd.DataFrame(test_encoded)

        return self.X_encoded, self.test_encoded

    @property
    def label(self):
        return self.labels
    
    @property
    def dummy_train_test_split(self):
        data_encoded = pd.get_dummies(pd.concat((self.X, self.test_data), axis=1))
        train_data_dummy = data_encoded.iloc[:614,:]
        test_data_dummy = data_encoded.iloc[614:,:]
        return train_data_dummy, test_data_dummy
    
    def split_label(labels, split_size, i):
        """split the output"""
        output_size = labels.shape[1] // split_size
        sub_output = labels[labels.columns[i * output_size: (i + 1) * output_size]]
        column_name = sub_output.columns.tolist()
        return sub_output, column_name
