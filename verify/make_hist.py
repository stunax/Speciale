import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import pickle
from ggplot import *

path = "verify/"
only_annot = True
fname = path + ('sum_plot_annot.pkl' if only_annot else 'sum_plot.pkl')
out_extra = 'annot' if only_annot else ''

with open(fname, 'rb') as f:
    data = pickle.load(f)
data = list(zip(*data))
df = pd.DataFrame()
df['sums'] = data[0]
df['fname'] = data[1]
df['dataset'] = data[2]
df['dataset'] = [x.split('/')[-1] for x in df['dataset']]
df['z'] = [int(x.split('_')[1][-3:]) for x in df.fname]
df['t'] = [int(x.split('_')[0][-3:]) for x in df.fname]
print(df.shape)
print(df.columns.values)

fn = out_extra + 'sum_density_z.png'
p = ggplot(aes(x='z', y='sums', colour='dataset'), data=df)
p_dens = p + geom_point() + facet_wrap('dataset', ncol=3) + xlab("z")
p_dens.save(path + fn)

fn = out_extra + 'sum_density_t.png'
p = ggplot(aes(x='t', y='sums', colour='dataset'), data=df)
p_dens2 = p + geom_point() + facet_wrap('dataset', ncol=3) + xlab("time")
p_dens2.save(path + fn)

fn = out_extra + 'sum_hist.png'
p = ggplot(aes(x='sums'), data=df)
p_hist = p + geom_histogram() + facet_wrap('dataset', ncol=3)
p_hist.save(path + fn)
