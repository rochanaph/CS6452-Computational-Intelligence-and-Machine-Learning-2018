import pandas as pd
import numpy as np

data = pd.read_csv("USD IDR Historical Data.csv", sep="\t")
dataInflasi = pd.read_csv("Rate inflasi.csv", sep=",")

# kebutuhan matching rate inflasi
yearInflasi = dataInflasi["Year"].tolist()
monthInflasi = dataInflasi["Month"].tolist()
rateInflasi    = dataInflasi["Rate"].tolist()
dictInflasi = {}
for i in range(len(dataInflasi)):
    dictInflasi[str(yearInflasi[i])+"-"+monthInflasi[i]] = rateInflasi[i]

tanggalExchange = data["Date"].tolist()
inflasi = []
for item in tanggalExchange:
    inflasi.append(dictInflasi[item.split("-")[2]+"-"+item.split("-")[1]])

# kebutuhan train test
data["Date"] = pd.to_datetime(data['Date'])
data.index = data["Date"]
del (data["Date"])
train = data['2017/03/6':]["Price"].str.replace(",","").values
train = train.astype(np.float32)
test = data[:'2017/03/6']["Price"].str.replace(",","").values
test = test.astype(np.float32)

print(len(data))
print (len(train))
print (len(test))

print (len(inflasi))

print (len(inflasi[:len(train)]))
print (len(inflasi[len(train)-1:]))