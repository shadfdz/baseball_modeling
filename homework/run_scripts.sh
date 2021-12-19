#!/bin/bash


USER=''
PASS=''
HOST=db
PORT=3306
DB_SCRIPT=./hw5.sql

# alter db table
mysql -u$USER -p$PASS -h $HOST -P $PORT <$DB_SCRIPT

python ./homework5.py

