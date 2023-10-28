import torch

CHECKPOINT = "./tmp" #load模型的路径
EPOCHS = 200 # 训练每个模型要用的epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model1: nnRegression
SPLIT_SIZE = 50

filepath = {
    "csv": {
        "adata_obs_meta": './data/adata_obs_meta.csv',
        "id_map": './data/id_map.csv',
        "multiome_obs_meta": './data/multiome_obs_meta.csv',
        "multiome_var_meta": './data/multiome_var_meta.csv',
        "sample_submission": './data/sample_submission.csv'
    },
    "parquet": {
        "adata_train": './data/adata_train.parquet',
        "de_trian": "./data/de_train.parquet",
        "multiome_train": "./data/multiome_train.parquet"  
    }
}