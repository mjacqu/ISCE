import sys
import os
sys.path.append('/scratch-second/mylenej/ISCE')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import re
import rasterio
import interferogram

#set path
path = '/scratch-third/mylenej/radar/SpitzerStein/results/asc_88/'

# analyze temporal baselines
dirs = os.listdir(path)
# Define the regex pattern to extract dates
pattern = re.compile(r'(\d{8})_(\d{8})')
# Create a list of datetime objects from the directory names
datetime_list = []
for d in dirs:
    match = re.match(pattern, d)
    if match:
        start_date = datetime.strptime(match.group(1), '%Y%m%d')
        end_date = datetime.strptime(match.group(2), '%Y%m%d')
        datetime_list.append((start_date, end_date))

# Calculate the datetime delta for each item in the list
datetime_deltas = [end - start for start, end in datetime_list]


def delta_files(dirs, datetime_deltas, delta):
    boolean = [d.days == delta for d in datetime_deltas]
    dt_dirs = [item for item, boolean in zip(dirs, boolean) if boolean]
    return dt_dirs

dt_dirs = delta_files(dirs, datetime_deltas, 6)

cohs = []
ds = []
for d in dt_dirs:
    coh_file = os.path.join(os.path.join(path,d),'merged/phsig.cor.geo')
    ds.append(datetime.strptime(d[0:8], '%Y%m%d'))
    dataset = rasterio.open(coh_file)
    coh = dataset.read(1)
    pt_coh= coh[400,1500]
    cohs.append(pt_coh)

fp ='/scratch-third/mylenej/radar/SpitzerStein/results/asc_88'
fls = os.listdir(fp)

coh_list = []
for f in fls:
    ifg =  interferogram.Interferogram(
        path=os.path.join(fp,f)
    )
    coh = ifg.coherence
    coh_list.append(coh)

coh_ts = np.stack(coh_list, axis=2)

coh_med = np.median(coh_ts, axis=2)
plt.imshow(coh_med)
plt.show()

plt.imshow(coh)

# Synthetic example
# Define the exponential decay function
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate sample data
x_data = np.linspace(0, 4, 50)
y = exponential_decay(x_data, 2.5, 1.3, 0.5)  # Example data
y_noise = 0.2 * np.random.normal(size=x_data.size)
y_data = y + y_noise  # Add some noise to the data

# Fit the curve to the data
popt, pcov = curve_fit(exponential_decay, x_data, y_data, p0=(1, 1e-6, 1))

# Generate fitted curve data
y_fit = exponential_decay(x_data, *popt)

# Plot the original data and the fitted curve
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data with Noise')
plt.plot(x_data, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.legend()
plt.show()
