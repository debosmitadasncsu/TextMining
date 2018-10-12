# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:42:58 2018

@author: trevo
"""

# Attempting to get this JSON data into workable form
# https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
# 
import json

# This has taken a ton of tinkering just to get the file to finally read in
with open('RS_2011-01.json', 'r', encoding = 'utf-8-sig') as f:
    distros_dict = json.load(f)

# We cant even use this yet
for distro in distros_dict:
    print(distro['Name'])



# Open text file for reading: this is interesting but unfortunately we cant do much with a txt file
# also dont print it cause it will crash
f = open('RS_2011-01.txt', 'r', encoding = 'latin-1')
fr = f.readlines()

# This correctly opens the json file but there are still some issues with it encoding = 'utf-8-sig' may be worth trying
j = open('RS_2011-01.json', 'r', encoding = 'latin-1')


# This is supposed to interpret JSON code i believe
j.json()
json.loads(j.text)


json = json.dumps(fr, sort_keys=True, indent=4)
