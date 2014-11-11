
import numpy
import Quandl
import cPickle
import csv

t = {}
with open('data/DJtech.csv', 'rb') as handle:
    r = csv.reader(handle)
    headers = r.next()
    for row in r:
        t[row[0]] = row[1]

auth = 'cVRzVMeaffZas5bzUyD8'

data = {}
for key, value in t.items():
    print key, value
    df = Quandl.get(value,
                    authtoken=auth,
                    collapse="monthly",
                    transformation="diff")
    df = df.dropna(subset=['Close'])
    data[key] = df

with open('data/dumpMonthly.pkl', 'wb') as handle:
    cPickle.dump(data, handle)

data = {}
for key, value in t.items():
    print key, value
    df = Quandl.get(value,
                    authtoken=auth,
                    collapse="daily",
                    transformation="diff")
    df = df.dropna(subset=['Close'])
    data[key] = df

with open('data/dumpDaily.pkl', 'wb') as handle:
    cPickle.dump(data, handle)
