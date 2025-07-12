import numpy as np
from datetime import datetime

# Monday 0
# Tuesday 1
# Wednesday 2
# Thursday 3
# Friday 4
# Saturday 5
# Sunday 6
def datestr2num(s):
   if isinstance(s, bytes):
      s = s.decode('utf-8')
   return datetime.strptime(s.strip(), "%d-%m-%Y").weekday()

dates, close=np.loadtxt('data.csv', delimiter=',', usecols=(1,6), converters={1: datestr2num}, unpack=True)
print("Dates =", dates)

averages = np.zeros(5)

for i in range(5):
   indices = np.where(dates == i) 
   prices = np.take(close, indices)
   avg = np.mean(prices)
   print("Day", i, "prices", prices, "Average", avg)
   averages[i] = avg


top = np.max(averages)
print("Highest average", top)
print("Top day of the week", np.argmax(averages))

bottom = np.min(averages)
print("Lowest average", bottom)
print("Bottom day of the week", np.argmin(averages))
