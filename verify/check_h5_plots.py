import matplotlib as mpl
mpl.use('Agg')
import pickle
import pandas as pd
from ggplot import *

out_extra = ''
path = "verify/"
fname = path + 'check_h5.pkl'
with open(fname, "rb") as f:
    df = pickle.load(f)


def geom_plots(y, ylabel, labs, df):

    p = ggplot(aes(y=y), df) + ylab(ylabel)

    p_z_intensity = p + geom_point(aes(x='z')) + facet_wrap('fname', ncol=3) \
        + xlab("Z axis")
    # p_z_intensity += geom_smooth()
    p_z_intensity.save(path + "z_%s.png" % labs, width=3, height=3)

    p = ggplot(aes(y=y), df) + ylab(ylabel)

    p_t_intensity = p + geom_point(aes(x='t')) + \
        facet_wrap('fname', ncol=3) + xlab("Frames")
    p_t_intensity.save(path + "t_%s.png" % labs, width=3, height=3)


geom_plots('intensity_sum', 'Slice intensity sum', 'intensity', df)
geom_plots('front_sum', 'front label sum', 'front', df)
geom_plots('back_sum', 'back label sum', 'back', df)
