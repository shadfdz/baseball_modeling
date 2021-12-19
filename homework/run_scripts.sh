#!/bin/bash


USER='root'
PASS='root'
HOST=db
PORT=3306
DB_SCRIPT=./baseball_stats.sql

# alter db table
mysql -u$USER -p$PASS -h $HOST -P $PORT <$DB_SCRIPT

python ./homework5.py

