test -e 'rnn.pkl' && echo 'RNN weights exists!' || wget 'https://www.dropbox.com/s/2ps31sdsgc51rcm/rnn.pkl' | tr -d '\r'
python3 P2.py $1 $2 $3 