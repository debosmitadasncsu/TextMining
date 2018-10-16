# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
f = open('C:\Text Mining\Dec2017 - Copy.txt','r+')
count = 0
arr = []

for line in f:
    
    start = line.find("title\":\"")
    end = line[start:].find("\",")+start
    title = line[start+8:end].lower()
    created_utc = line[line.find("created_utc")+13:line.find("distinguished")-2]
    if ("bitcoin" in title) or ("btc" in title):
        content = title + " " +created_utc
        arr.append(content)
        print(content)

file = open("C:/Text Mining/testfile-2017-12.txt","w") 
for a in arr:
    file.write(a)
    file.write('\n')