# !/bin/bash

sudo rm -rf store_img/*
sudo rm -rf index.txt
python3 manage.py runserver 8000
