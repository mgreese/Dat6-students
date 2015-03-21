# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test =  pd.read_csv('./in/test.csv').dropna() 
test.isnull().sum()==0 #clean!
train = pd.read_csv('./in/train.csv').dropna() 
train.isnull().sum()==0 #clean!

#Have to rename the count variable to "Demand" since count is a method in python
train.rename(columns={'count':'demand'},inplace=True)

#Clean data, break out granular time variable (format 2011-01-01 05:00:00)

def FeatureEngineer(data):  
  
  data.rename(columns={'season':'Season','workingday':'Workday'}, inplace=True)
  data['datetime_unformatted']=data.datetime
  data['datetime'] = pd.to_datetime(data.datetime)
  data.set_index('datetime', inplace=True)
  data['Year'] = data.index.year
  data['Month'] = data.index.month
  data['Day'] = data.index.day
  data['Weekday'] = data.index.weekday
  data['Hour'] = data.index.hour
  data.reset_index(inplace=True)
  
  data['AM_PM']= data.Hour  
  data.AM_PM[data.Hour<12]=0 #AM
  data.AM_PM[data.Hour>=12]=1 #PM
  
  data['Time_of_Day']= data.Hour  
  data.Time_of_Day[(data.Hour>=2) & (data.Hour<=5)]=0 #"Very Early AM" 
  data.Time_of_Day[(data.Hour>=6) & (data.Hour<=9)]=1 #"Morning Commute"
  data.Time_of_Day[(data.Hour>=10) & (data.Hour<=13)]=2 #"Mid-morning and lunch"
  data.Time_of_Day[(data.Hour>=14) & (data.Hour<=17)]=3 #"Mid Afternoon"
  data.Time_of_Day[(data.Hour>=18) & (data.Hour<=21)]=4 #"Evening Commute and HH rush"
  data.Time_of_Day[(data.Hour==22) | (data.Hour==23)| (data.Hour==0)| (data.Hour==1)]=5 #"Late PM"
  
  
  data['atemp2']=data.atemp
  data['atemp']=data.atemp*9/5+32 

  data['Weather_Temp']= data.atemp    
  data.Weather_Temp[data.atemp<50]=0
  data.Weather_Temp[(data.atemp>=50) & (data.atemp<65)]=1
  data.Weather_Temp[(data.atemp>=65) & (data.atemp<80)]=2
  data.Weather_Temp[(data.atemp>=80) & (data.atemp<95)]=3
  data.Weather_Temp[(data.atemp>=95) & (data.atemp<110)]=4
  data.Weather_Temp[data.atemp>=110]=5

FeatureEngineer(train)
FeatureEngineer(test)

graphs = ['Season', 'Weekday', 'Workday','Time_of_Day', 'Year','Month','Day','Hour', 'Weather_Temp']

def kagglechart(frequency):
  
  #first collapse data  
  regm=train.groupby(['Year',frequency]).registered.sum()/train.groupby(['Year',frequency]).registered.count()  
  regd=train.groupby(['Year',frequency]).registered.count()

  reg=train.groupby(['Year',frequency]).registered.sum()/train.groupby(['Year',frequency]).registered.count()
  
  cas=train.groupby(['Year',frequency]).casual.sum()/train.groupby(['Year',frequency]).casual.count()
  #now make some labels and graph parameters
  Season=('Winter','Spring', 'Summer', 'Fall')
  Weekday=('Mon.','Tue.','Wen.','Thu.','Fri.','Sat.','Sun.')
  Workday=("Playday","Workday")
  Time_of_Day=("Early AM","AM Commute", "Mid AM+Lunch","Mid PM", "PM Commute","Late PM")  
  Year = tuple(set(train.Year))
  Month = tuple(set(train.Month))
  Day = tuple(set(train.Day))
  Hour = tuple(set(train.Hour))
  Weather_Temp=("<50","50-65","65-80","80-95","95-110",">110")
  labels=[Season,Weekday,Year,Month,Day,Workday,Hour,Time_of_Day,Weather_Temp]
  dict={'Season':0,'Weekday':1,'Year':2,'Month':3,'Day':4,'Workday':5,'Hour':6,'Time_of_Day':7,'Weather_Temp':8}   
  
  width = 0.35       
  
  p1 = plt.bar(np.arange(1,len(reg)+1),reg, width, color='r')
  p2 = plt.bar(np.arange(1,len(cas)+1),cas, width, color='b', bottom=reg)
  
  plt.ylabel('Demand')
  plt.title('Average Hourly Demand by '+frequency)
  #plt.yticks(np.arange(0,max(reg+casual),15000))
  plt.xticks(np.arange(1,len(reg)+1)+width/2.,labels[dict[frequency]]+labels[dict[frequency]],rotation=90)

  if frequency!='Year':  
    plt.xlabel('2011                                       2012')
  plt.legend( (p1[0], p2[0]), ('Registered Renters', 'Casual Renters'),loc=0 )
  plt.savefig('Average Hourly Demand by '+frequency+'.png',bbox_inches='tight', transparent=True,dpi=1080) 

  plt.show()

def dographs():
  for g in graphs:
    kagglechart(g)
    

