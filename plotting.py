import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from matplotlib import dates
import datetime

# This is just a basic plotting script to show model returns overall and specifically
# on INTC, which is a commonly benchmarked stock for backtesting.


# Grabbing data
store = h5py.File('plotting.h5', 'r')
dji_ret = np.array(store.get('dji_ret'))
model_ret_b = np.array(store.get('model_ret_b'))
model_ret_e = np.array(store.get('model_ret_e'))

intc_ret = np.array(store.get('INTC_ret'))
intc_ret_b = np.array(store.get('INTC_ret_b'))
intc_ret_e = np.array(store.get('INTC_ret_e'))

store.close()

store = pd.HDFStore('plotting.h5')
test_dates = np.array(store.select('test_dates'))
intc_dates = np.array(store.select('INTC_dates'))

# Pyplot manipulations for dates as x-axis
test_dates = [t[0] for t in test_dates]
converted_dates = list(map(datetime.datetime.strptime, test_dates, len(test_dates)*['%Y-%m-%d']))
x_axis = converted_dates
formatter = dates.DateFormatter('%Y-%m-%d')

# Plotting model vs DJIA
plt.plot(x_axis, dji_ret - 1, label='DJIA')
plt.plot(x_axis, model_ret_b - 1, label='Baseline Model')
plt.plot(x_axis, model_ret_e - 1, label='Experimental Model')
plt.legend(loc=2)
plt.title('Dow Jones Industrial Average')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
ax = plt.gcf().axes[0] 
ax.xaxis.set_major_formatter(formatter)
plt.gcf().autofmt_xdate(rotation=25)
plt.grid()
plt.show()

intc_dates = [t[0] for t in intc_dates]
converted_dates = list(map(datetime.datetime.strptime, intc_dates, len(intc_dates)*['%Y-%m-%d']))
x_axis = converted_dates

# Plotting model vs INTC
plt.plot(x_axis, intc_ret - 1, label='INTC')
plt.plot(x_axis, intc_ret_b - 1, label='Baseline Model')
plt.plot(x_axis, intc_ret_e - 1, label='Experimental Model')
plt.legend(loc=3)
plt.title('Intel Corporation (INTC)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
ax = plt.gcf().axes[0]
ax.xaxis.set_major_formatter(formatter)
plt.gcf().autofmt_xdate(rotation=25)
plt.grid()
plt.show()