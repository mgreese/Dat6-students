# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:43:14 2015

@author: reesem
"""
import pandas as pd
import requests


files = {'f': ('../../dat6-students/reese/domestic_information/adoption_subsidy2013.pdf', open('../../dat6-students/reese/domestic_information/adoption_subsidy2013.pdf', 'rb'))}
response = requests.post("https://pdftables.com/api?key=k8zc21wp64e1", files=files)
response.raise_for_status() # ensure we notice bad responses
response.text

get_html('../../dat6-students/reese/domestic_information/adoption_subsidy2013.pdf','025hyv8yh01e')


data = requests.get('http://api.census.gov/data/2013/pep/stchar5?get=AGE,SEX,DATE,STNAME,RACE5,HISP,POP&for=state:*&key=aa64249a41c4528213a6e3dfa34ecbb2616a7a0a')