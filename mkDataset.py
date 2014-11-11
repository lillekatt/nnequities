
import numpy
import csv
import cPickle
import datetime as dt
import calendar as cl

dataMonth = cPickle.load(open('data/dumpMonthly.pkl','rb'))
dataDay = cPickle.load(open('data/dumpDaily.pkl','rb'))

# get median for labelling returns
getAll = numpy.asarray([])
for key, value in dataMonth.items():
    getAll = numpy.concatenate([getAll,value.values[:,3]])
median = numpy.median(getAll)

# make dataset
handle_tr = open('data/trainData.dat', 'w')
handle_tr.write("M12,M11,M10,M9,M8,M7,M6,M5,M4,M3,M2,M1,"
                "D15,D14,D13,D12,D11,D10,D9,D8,D7,D6,D5,D4,D3,D2,D1,"
                "Jan,Label,Ticker,Month,Year\n")
handle_ts = open('data/testData.dat', 'w')
handle_ts.write("M12,M11,M10,M9,M8,M7,M6,M5,M4,M3,M2,M1,"
                "D15,D14,D13,D12,D11,D10,D9,D8,D7,D6,D5,D4,D3,D2,D1,"
                "Jan,Label,Ticker,Month,Year\n")
for key, value in dataMonth.items():
    data = []
    for k in xrange(len(dataMonth[key].values[:,3])-12):

        month =  dataMonth[key].index[k+12].month
        year = dataMonth[key].index[k+12].year

        # skip current month because insufficient data
        if month!=11 or year!=2014:
            rm = numpy.zeros(12)
            for i in xrange(k,k+12):
                rm[i-k] =  dataMonth[key].values[i,3]
            label = dataMonth[key].values[k+12,3] > median

            endday = cl.monthrange(year, month)[1]
            start = dataDay[key].index.searchsorted(dt.datetime(year, month, 1))
            end = dataDay[key].index.searchsorted(dt.datetime(year, month, endday))
            rd = dataDay[key].ix[start:end].values[:,3][-15:]

            if month==1:
                idct = 1
            else:
                idct = 0

            if len(rd)==15:
                datapoint = numpy.concatenate([rm, rd]).tolist()
                datapoint.append(idct)
                datapoint.append(int(label)+1)
                datapoint.append(key)
                datapoint.append(month)
                datapoint.append(year)
                data.append(datapoint)
            else:
                print "skipped",key,month,year

    train = data[:int(len(data)*0.75)]
    test = data[int(len(data)*0.75):]
    for tr in train:
        handle_tr.write(','.join([str(s) for s in tr])+'\n')
    for ts in test:
        handle_ts.write(','.join([str(s) for s in ts])+'\n')

handle_tr.close()
handle_ts.close()
