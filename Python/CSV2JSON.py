#-*- coding:utf-8 -*-

import os
import os.path
import csv
rootdir = "/Users/ying/Documents"     # folder stores csv files

for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        abs_path = os.path.join(parent,filename)
        if ".csv" in abs_path:
            print abs_path
            #对每个文件进行处理
            with open(abs_path, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
                header = rows[0]
                for rowOne in rows[1:]:
                    json_row = {}
                    for i in range(0,len(rowOne)):
                        json_row[header[i]] = rowOne[i]
                    print json_row
