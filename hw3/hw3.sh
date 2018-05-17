test -e 'model.h5' &&  echo 'Model exists' || wget 'https://www.dropbox.com/s/lbo0n3t1kfknhux/model.h5'| tr -d '\r'
python3 hw3.py $1 $2
