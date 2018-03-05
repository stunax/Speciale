import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import pickle
import re
from ggplot import *

path = "verify/"
only_annot = True
rem_sec = True
fname = path + 'sum_plot_annot_front.pkl'
out_extra = 'front_annot_'

with open(fname, 'rb') as f:
    data = pickle.load(f)
data = list(zip(*data))
df = pd.DataFrame()
df['sums'] = data[0]
df['fname'] = data[1]
df['dataset'] = data[2]
if rem_sec:
    df.filter(axis=0, like='sec')
df['fname'] = [re.sub('^[a-z_]+', '', x) for x in df.fname]
df['dataset'] = [x.split('/')[-1] for x in df['dataset']]
df['z'] = [int(x.split('_')[1]) for x in df.fname]
df['t'] = [int(x.split('_')[0]) for x in df.fname]

fn = out_extra + 'sum_density_z.png'
p = ggplot(aes(x='z', y='sums', colour='dataset'), data=df)
p_dens = p + geom_point() + xlab("z")
p_dens.save(path + fn)

fn = out_extra + 'sum_density_t.png'
p = ggplot(aes(x='t', y='sums', colour='dataset'), data=df)
p_dens2 = p + geom_point() + xlab("time")
p_dens2.save(path + fn)

fn = out_extra + 'sum_hist.png'
p = ggplot(aes(x='sums', colour='dataset'), data=df)
p_hist = p + geom_histogram() #+ facet_wrap('dataset', ncol=3)
p_hist.save(path + fn)
