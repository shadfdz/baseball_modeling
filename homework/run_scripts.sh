#!/bin/bash

USER='root'
PASS='root'
HOST=db
PORT=3306
DB_SCRIPT=./baseball_stats.sql

sleep 10m

# alter db table
mysql -u$USER -p$PASS -h $HOST -P $PORT <$DB_SCRIPT

python ./final.py

