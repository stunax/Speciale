import h5py
from time import gmtime, strftime

data_path = "/home/dpj482/data/"
save_path = data_path + "models/"
tiff_path = data_path + "tiffs/"
imag_path = data_path + "images/"
test_imag_path = data_path + "test_images/"
weights_path = data_path + 'model_weights/%s_%i_%i_%i_weights.h5'
results_path = data_path + "result.csv"
rerun_name = "second"
df_info_path = data_path + "df_info.h5"
tiff_pivot = 80
annotations_path = data_path + "Annotations from Alex 2017/"
annotations_path_c = data_path + "new_labels/"
annotations_path_prep = data_path + "prep_images/"
kmeans_path = data_path + "kmeans_labels/"
annotation_groupname = "anno"
second_annotaion_groupname = "sec_anno"
c_anno_groupname = "canno"

groupname_format = "%s_%s"

# normalize weightshare config
find_close_group = 5

# Experimental setup
learning_rate = 0.0001
num_steps = 2000
max_epochs = 100
batch_size = 128
random_state = 1337
bag_size = 2
patch_size = (16, 16, 5)
num_classes = 2
dropout = 0.3  # Dropout, probability to drop a unit

# keras generator options
use_saved_weights = False
len_settings = int(30000 / batch_size) + 1
max_queue_size = len_settings * 3

# tensorboard options
histogram_freq = 0

# Log parameters
logs_path = '/tmp/tensorflow_logs/'
run_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

# Preprocessing parameters
normalize_input = True
median_time = 0

kernel_size = [(5, 5, 1)]
# clust_kernel_size = [(9, 9, 1), (9, 9, 3), (9, 9, 5)]
clust_kernel_size = [(5, 5, 1), (5, 5, 3)]
# bag_sizes = [1, 5, 7]

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

labels = [
    [], [],
    ['040_15', '028_15', '095_15'],
    ['039_15', '028_15', '093_15'],
    ['010_15', '037_15', '091_15'],
]

im_paths = [(data_path + p[0], p[1], p[2], p[3]) for p in lsm_images]

h5s = [data_path + str(i) + ".h5" for i in range(1, 6)]


def get_h5(annotation_name=annotation_groupname, ignore_1_2=False):
    X = []
    h5files = h5s
    if ignore_1_2:
        h5files = h5files[2:]

    for h5f in h5files:
        h5f = h5py.File(h5f, 'r+')
        X.extend(
            [(h5f, df)
             for df in h5f.keys() if annotation_name + df in h5f and
             len(df) == 6])
    return X


def print_to_result(Semisupervised, normalized, median_filter, loss, accuracy):
    string = "%s,%s,%i,%f,%f\n" % (
        Semisupervised, normalized, median_filter, loss, accuracy)
    with open(results_path, "a") as f:
        f.write(string)
