
function test_func {
    python3 src/keras_autoencoder.py --median_time=2 --learning_rate=$1 --epochs=6 --run_name=lr_test_$1 --normalize=0 --skip_pred=1
    python3 src/keras_autoencoder.py --median_time=2 --learning_rate=$1 --epochs=6 --run_name=lr_test_$1_norm --normalize=1 --skip_pred=1
}


test_func 0.0001
test_func 0.0005
test_func 0.00001
test_func 0.00005
test_func 0.000001
test_func 0.000005
test_func 0.0000001
test_func 0.0000005
test_func 0.00000001
test_func 0.00000005