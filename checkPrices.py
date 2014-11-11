
import numpy
import Quandl
import cPickle
import csv
import datetime as dt
import calendar as cl

t = {}
with open('data/DJtech.csv', 'rb') as handle:
    r = csv.reader(handle)
    headers = r.next()
    for row in r:
        t[row[0]] = row[1]

auth = 'cVRzVMeaffZas5bzUyD8'

df = Quandl.get(t['DOX'],
                authtoken=auth,
                collapse="daily",
                transformation="diff")

print df

# month = 8
# year = 1997
# endday = cl.monthrange(year, month)[1]
# start = df.index.searchsorted(dt.datetime(year, month, 1))
# end = df.index.searchsorted(dt.datetime(year, month, endday))
#
# print df.ix[start:end]
