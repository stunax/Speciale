import config
import pandas as pd
pd.set_option('display.width', 1200)

headers = ["issemi", "normalized", "filter",
           "loss", "acc", "close", "tp", "fp", "fn", "tn"]
groupby = ["issemi", "normalized", "filter", "close"]

data = pd.read_csv(config.results_path, header=None, names=headers)
# print(data)


types = ["tp", "fp", "fn", "tn"]
# print(data[types] / data[types].sum(axis=1) * 100)
data[types] = data.loc[:, types].div(data[types].sum(axis=1), axis=0) * 100
# print(data)


data = data.groupby(groupby)

print(data.acc.agg(
    {'u': lambda x: x.mean(),
     'std': lambda x: x.std(),
     'count': lambda x: x.count()}))


print(data[types].agg({'mean': lambda x: x.mean()}))
print(data[types].agg({'std': lambda x: x.std()}))
