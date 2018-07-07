import config
import pandas as pd

headers = ["issemi", "normalized", "filter", "loss", "acc", "close"]

data = pd.read_csv(config.results_path, header=None, names=headers)
print(data)
data = data.groupby(headers[:3] + headers[-1:])

print(data.acc.agg({'u': lambda x: x.mean(),
                    'std': lambda x: x.std(), 'count': lambda x: x.count()}))
