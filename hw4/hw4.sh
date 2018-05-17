test -e 'VAE.h5' && echo 'VAE.h5 exist' || wget 'https://www.dropbox.com/s/8xzfau0ikywd0vs/VAE.h5'| tr -d '\r'
test -e 'GAN_generator.h5' && echo 'GAN_generator.h5 exist' || wget 'https://www.dropbox.com/s/fw0mzog7wc0tc8g/GAN_generator.h5'| tr -d '\r'
test -e 'ACGAN_generator.h5' && echo 'ACGAN_generator.h5 exist' || wget 'https://www.dropbox.com/s/d9tyodiuke4kmus/ACGAN_generator.h5'| tr -d '\r'
python3 hw4.py $1 $2
