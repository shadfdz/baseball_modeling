# HW1
# Author: Shad Fernandez
# Date: 02-OCT-2021


import pymysql

con = pymysql.connect(host='localhost',user='root',
                      password='sd.xd.mmc',
                      database='baseball')
try:
    with con.cursor() as cur:
        cur.execute('Show tables from baseball')
        rows = cur.fetchall()
        for row in rows:
            print(f'{row[0]}')
finally:
    con.close()
