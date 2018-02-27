import pandas as pd
import pickle
from ggplot import *

with open('verify/sum_plot.pkl', 'rb') as f:
    data = pickle.load(f)
data = zip(*data)
df = pd.DataFrame(data={'sums': data[0], 'fname': data[1]})
df['z'] = [int(x.split('_')[0]) for x in data[1]]

path = "verify/"
p = ggplot(aes(x='z', y='sums'), data=df)

fn = 'sum_density.png'
ggsave(p + geom_point(), filename=fn, path=path)

fn = 'sum_density.png'
ggsave(p, filename=fn, path=path)
