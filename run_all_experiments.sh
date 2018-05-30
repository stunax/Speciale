
FILE="results.csv"
python3 src/simple_keras.py --normalize=0 --median_time=0 >> $FILE
python3 src/simple_keras.py --normalize=1 --median_time=0 >> $FILE
python3 src/simple_keras.py --normalize=0 --median_time=2 >> $FILE
python3 src/simple_keras.py --normalize=1 --median_time=2 >> $FILE
python3 src/keras_autoencoder.py --normalize=0 --median_time=0 >> $FILE
python3 src/keras_autoencoder.py --normalize=1 --median_time=0 >> $FILE
python3 src/keras_autoencoder.py --normalize=0 --median_time=2 >> $FILE
python3 src/keras_autoencoder.py --normalize=1 --median_time=2 >> $FILE
python3 src/keras_autoencoder.py --normalize=0 --median_time=4 >> $FILE
python3 src/keras_autoencoder.py --normalize=1 --median_time=4 >> $FILE