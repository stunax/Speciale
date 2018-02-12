import h5py

data_path = "/mnt/orico/PANCREAS/"
save_path = "models/"
tiff_path = data_path + "tiffs/"
pred_path = "images/"
rerun_name = "second"
df_info_path = data_path + "df_info.h5"
tiff_pivot = 80
annotations_path = data_path + "Annotations from Alex 2017/"
annotation_groupname = "anno"
second_annotaion_groupname = "sec_anno"
groupname_format = "%s_%s"

random_state = 1337
bag_sizes = [5]

kernel_size = [(5, 5, 1), (5, 5, 3)]
# clust_kernel_size = [(9, 9, 1), (9, 9, 3), (9, 9, 5)]
clust_kernel_size = [(5, 5, 1), (5, 5, 3)]
# bag_sizes = [1, 5, 7]

# kernel_size = [(1, 1, 1), (5, 5, 1), (3, 3, 1), (7, 7, 1), (5, 5, 3)]
median_filter = range(0, 3)
image_size = (1024, 1024)  # Image size and channel amount.
nchannels = 1
true_percentage = .20650622591164353

lsm_images = [
    ("LI 2015-07-12 MIPGFP_Muc1_40x_2015_07_14__pos1.ims", "1", 40, 159),
    ("LI 2015-07-12 MIPGFP_Muc1_pos2.ims", "2", 40, 159),
    ("LI20160304_ last timepoint_Subset pos1.lsm", "4", 27, 111),
    ("LI-2016-03-04_ last timepoint_Subset pos2.lsm", "5", 27, 111),
    ("LI2016-03-04_ last timepoint_Subset pos3.lsm", "3", 27, 111)
]

im_paths = [(data_path + p[0], p[1], p[2], p[3]) for p in lsm_images]

h5s = [data_path + str(i) + ".h5" for i in range(1, 6)]

# First run yielded log2 nad 8192, which was max. Rerun with log2, and more
# estimators
parameters = {
    "n_estimators": [2**i for i in range(9, 10)],
    # "n_estimators": [2**i for i in range(1, 15)],
    "max_features": ["log2"]
    # "leaf_size": [1, 50]
}
parametersfull = {
    "n_estimators": [2**i for i in range(10, 11)],
    "max_features": ["log2"]
}


def get_h5(annotation_name=annotation_groupname):
    X = []

    for h5f in h5s:
        h5f = h5py.File(h5f, 'r+')
        X.extend([(h5f, df)
                  for df in h5f.keys() if annotation_name + df in h5f])
    return X
