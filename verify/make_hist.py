import pandas as pd
import pickle
from ggplot import *

with open('verify/sum_plot.pkl', 'rb') as f:
    data = pickle.load(f)
data = list(zip(*data))
# df = pd.DataFrame(data={'sums': data[0], 'fname': data[1], 'dataset': data[2]})
# df = pd.DataFrame.from_records(data, columns=['sums', 'fname', 'dataset'])
df = pd.DataFrame()
df['sums'] = data[0]
df['fname'] = data[1]
df['dataset'] = data[2]
df['z'] = [int(x.split('_')[0]) for x in df.fname]
print(df.shape)
print(df.columns.values)
path = "verify/"
p = ggplot(aes(x='z', y='sums', colour='dataset'), data=df)

fn = 'sum_density.png'
ggsave(p + geom_point(), filename=fn, path=path)

fn = 'sum_hist.png'
ggsave(p + geom_histogram(), filename=fn, path=path)
