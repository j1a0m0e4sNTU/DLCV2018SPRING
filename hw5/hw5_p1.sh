test -e 'cnn.pkl' && echo 'CNN weight exists!' || wget 'https://www.dropbox.com/s/vk92xoqh5cmileg/cnn.pkl' | tr -d '\r'
python3 P1.py $1 $2 $3