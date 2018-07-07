function make_stats {
python3 src/keras_simple.py --normalize=$1 --median_time=$2 --run_name=test_$1_$2_1 --close_size=$3
python3 src/keras_simple.py --normalize=$1 --median_time=$2 --run_name=test_$1_$2_2 --close_size=$3 
python3 src/keras_simple.py --normalize=$1 --median_time=$2 --run_name=test_$1_$2_3 --close_size=$3 
python3 src/keras_simple.py --normalize=$1 --median_time=$2 --run_name=test_$1_$2_4 --close_size=$3 
python3 src/keras_simple.py --normalize=$1 --median_time=$2 --run_name=test_$1_$2_5 --close_size=$3 

}


# python3 src/simple_keras.py --normalize=1 --median_time=0 --close_size=0
# python3 src/simple_keras.py --median_time=0 
# python3 src/simple_keras.py --normalize=1 --median_time=0 
# python3 src/simple_keras.py --median_time=2 
# python3 src/simple_keras.py --normalize=1 --median_time=2 
# Rerun several times, to avoid memory stopping the system
# make_stats 0 0 0
# make_stats 0 0 20
# make_stats 0 2 20
# make_stats 1 0 20
# make_stats 1 2 20
# python3 src/keras_autoencoder.py --median_time=2 --learning_rate=0.0005
# python3 src/keras_autoencoder.py --normalize=1 --median_time=2 --learning_rate=0.000001
python3 src/keras_autoencoder2.py --normalize=1 --median_time=2 --learning_rate=0.000001
