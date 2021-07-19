curl -L -o dsb2018_topcoders.zip https://www.dropbox.com/s/qvtgbz0bnskn9wu/dsb2018_topcoders.zip

unzip dsb2018_topcoders.zip albu/weights/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip selim/nn_models/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip victor/nn_models/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip victor/lgbm_models/* -d ./dsb2018_topcoders/
unzip dsb2018_topcoders.zip data/folds.csv -d ./dsb2018_topcoders/

mkdir dsb2018_topcoders/data_test
mkdir dsb2018_topcoders/predictions

rm dsb2018_topcoders.zip