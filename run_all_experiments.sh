function make_stats {
    for (( i=1; i<=10; i++))
        do
            python3 src/keras_simple.py --normalize=$1 --median_time=$2 --run_name=test_$1_$2_$i --close_size=$3 --random_state=$i
        done

}
function make_stats2 {
    for (( i=1; i<=5; i++))
        do
            python3 src/keras_autoencoder2.py --normalize=$1 --median_time=$2 --learning_rate=0.000001 --run_name=ae_test_$1_$2_$i --random_state=$i
        done

}


# python3 src/simple_keras.py --median_time=0 --close_size=0
# python3 src/simple_keras.py --normalize=1 --median_time=0 --close_size=0
# python3 src/simple_keras.py --median_time=0 
# python3 src/simple_keras.py --normalize=1 --median_time=0 
# python3 src/simple_keras.py --median_time=2 
# python3 src/simple_keras.py --normalize=1 --median_time=2 
make_stats 0 0 0
make_stats 0 0 20
make_stats 0 2 20
make_stats 1 0 20
make_stats 1 2 20
# python3 src/keras_autoencoder.py --median_time=2 --learning_rate=0.0005
# python3 src/keras_autoencoder.py --normalize=1 --median_time=2 --learning_rate=0.000001
# make_stats2 1 2
# make_stats2 0 2
# make_stats2 1 0
# make_stats2 0 0
