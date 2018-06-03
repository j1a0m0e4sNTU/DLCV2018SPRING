test -e 'rnn_seq.pkl' && echo 'RNN_seq weights exists!' || wget 'https://www.dropbox.com/s/tkf07skgczijbh7/rnn_seq.pkl' | tr -d '\r'
python3 P3.py $1 $2