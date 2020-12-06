cd data
cd DanceRevolution

wget --load-cookies /tmp/cookies.txt "https://doc.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://doc.google.com/uc?export=download&id=1x2PwxjsnQFlKltmleYY_DiWi63H0tQRl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1x2PwxjsnQFlKltmleYY_DiWi63H0tQRl" -O hh_env.tar.xz && rm -rf /tmp/cookies.txt
tar -xf hh_env.tar.xz
rm hh_env.tar.xz
