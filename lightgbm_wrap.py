import lightgbm as lgb
from utils import split, mean_rmse
from tqdm import tqdm
import pandas as pd
import numpy as np

class lightgbm_wrapper:
    def __init__(self, custom_loss) -> None:
        self.params = {
            'objective': "custom_loss",
        }
        self.model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, objective=custom_loss) 


    def fit(self, x, labels):
        self.model.fit(x, labels)

    def predict(self, test_data):
        predict = self.model.predict(test_data)
        return predict


def lightgbm_pipeline(x, labels, test_data, split_size):
    model = lightgbm_wrapper(custom_loss=mean_rmse)

    output = pd.DataFrame(np.empty(shape=(len(test_data), 0)))
    for loops in tqdm(range(split_size)):
        sub_labels, sub_column_name = split(labels, split_size, loops)
        model.fit(x=x, labels=sub_labels)
        sub_predictions = model.predict(test_data=test_data)
        sub_predictions = pd.DataFrame(sub_predictions, columns= sub_column_name)
        output = pd.concat((output, sub_predictions), axis=1)

    lgb.plot_metric(model)
    return output

