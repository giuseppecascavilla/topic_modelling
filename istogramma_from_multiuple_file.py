# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:05:40 2019

@author: giuseppec
"""


import pandas as pd
import os
directory='C:/Users/giuseppec/Desktop/PYTHONTopicModelling/topics'

df=[]
i=0

for f in os.listdir(directory):
    with open(os.path.join(directory, f), mode='r') as file:
        #print(file.name)
        df.append(pd.read_csv(file.name, sep=',', encoding='latin-1', engine='python'))
        #print(df[i])
        i = i+1
