import pandas as pd
import pickle
from ggplot import *

with open('verify/sum_plot.pkl', 'rb') as f:
    data = pickle.load(f)
data = zip(*data)
df = pd.DataFrame(data={'sums': data[0], 'fname': data[1]})

path = "verify/"
fn = 'sum_density.png'
p = ggplot(aes(x='sums'), data=df)

ggsave(p, filename=fn, path=path)
