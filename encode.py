import pandas as pd
import geoip2.database
from ipwhois import IPWhois
from pprint import pprint
from math import sin, cos, sqrt, atan2, radians
import os
import re
import numpy as np
from pandas import DataFrame as df
import pickle
from sqlalchemy import create_engine
from sklearn import preprocessing
import datetime
from time import strptime

import sys
name=sys.argv[1]
#"dovecot-logs-deeproot"
def log_to_dict(name):
    a=[]
    line_regex = re.compile(".*imap-login:.*$")
    # Output file, where the matched loglines will be copied to
    with open(name, "r") as in_file:
        # Loop over each log line
        l=[]
        for line in in_file:
            
            # If log line matches our regex
            if (line_regex.search(line)):
                d=dict()
                for i in line:
                    tokens=line.split()
                    if tokens[0]=="mail.info:":
                        tokens.remove(tokens[0])
                    d['date']=str(tokens[0])+" "+str(tokens[1])
                    d['time']=tokens[2]
                    d['info']=tokens[5]
                    d['task']=""
                    for i in range(5,len(tokens)):
                        if str(tokens[i]).isalpha():
                            d['task']+=str(tokens[i])+" "
                        elif "down." in str(tokens[i]):
                            d['task']+=str(tokens[i])
                        elif "=" in str(tokens[i]):
                            s=tokens[i].split("=")
                            d[s[0]]=s[1].replace(',','')
                        elif "Login:" == str(tokens[i]):
                            d['task']+=str(tokens[i])
                           
                if d not in l:
                    l.append(d)
    dff=df(l)                     
    return dff
dff=log_to_dict(name)

reader = geoip2.database.Reader('geolite2/GeoLite2-City_20200107/GeoLite2-City.mmdb')
#distance calculation

lat1=radians(13.002789)
lon1=radians(77.596464)
R = 6373.0

dff1=dff
#dff1 = pd.read_pickle(name)
ispd=dict()
distd=dict()
isp=list()
dist=list()

#pd.set_option('display.max_rows',None)
#print(dff1)
def get_isp_dist(rip):
    response = reader.city(rip)
    lat2=radians(response.location.latitude)
    lon2=radians(response.location.longitude)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    obj = IPWhois(rip)
    results = obj.lookup_whois()
    ISP=results['asn_description'].split('-')[0]
    return ISP,distance 


for i in dff.rip.unique():
    try:
        a=get_isp_dist(i)
    except:
        a=('Local','0')
    ispd[i]=a[0]
    distd[i]=a[1]
x=[ispd[i] for i in dff.rip]
y=[distd[i] for i in dff.rip]
dff['isp']=x
dff['distance']=y
dff.to_pickle('LogFileDfs/original')
dfall=pd.read_pickle('LogFileDfs/LogDumpdf')
dfall=dfall.append(dff,ignore_index=True)
dfall.to_pickle('LogFileDfs/LogDumpdf')
#preprocessing 


le= preprocessing.LabelEncoder()
users=dfall.user.unique()    
rips=dfall.rip.unique()
lips=dfall.lip.unique()
isps=dfall.isp.unique()
lerip= preprocessing.LabelEncoder()
lelip= preprocessing.LabelEncoder()
leisp= preprocessing.LabelEncoder()
leuser= preprocessing.LabelEncoder()
lerip.fit(rips)
lelip.fit(lips)
leisp.fit(isps)
leuser.fit(users)
dff1=dff.copy()
dff1['rip']=lerip.transform(dff1['rip'])
dff1['lip']=lelip.transform(dff1['lip'])
dff1['user']=leuser.transform(dff1['user'])
dff1['isp']=leisp.transform(dff1['isp'])


dt=[]
now = datetime.datetime.now()
for i in range(0,len(dff1)):
    d=dff1.loc[i,'date']
    t=dff1.loc[i,'time']
    x=strptime(str(d[:3]),'%b').tm_mon
    hms=t.split(':')
    log_time=datetime.datetime(2020,x,int(d[4:]),int(hms[0]),int(hms[1]),int(hms[2]))
    diff = now - log_time
    dt.append(diff.days)
dff1['date_time']=dt
dff2 = pd.DataFrame(dff1, columns = ['user','rip', 'lip','date_time','isp','distance']) 
#engine = create_engine("mysql+pymysql://{user}:{pw}@172.17.0.3/{db}".format(user="richul",pw="richul123",db="emss"))
#dff2.to_sql('encoded', con = engine, if_exists = 'replace', chunksize = 1000)
#dff1.to_sql('original', con = engine, if_exists = 'replace', chunksize = 1000)
df_en_all=pd.read_pickle('LogFileDfs/LogDumpEncoded')
df_en_all=df_en_all.append(dff2,ignore_index=True)
df_en_all.to_pickle('LogFileDfs/LogDumpEncoded')
with pd.ExcelWriter('Output/Encoding.xlsx') as writer:
    dff1.to_excel(writer, sheet_name='original')
    dff2.to_excel(writer, sheet_name='encoded')
    df_en_all.to_excel(writer, sheet_name='all')
def call_me():
    print(dff2)
    dff2.to_pickle('LogFileDfs/encoded')
call_me()
        


