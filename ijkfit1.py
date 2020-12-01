import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models

spectrum_cs = pd.read_csv('Cs137_spectrum.csv')
spectrum_cs['y_err'] = np.sqrt(spectrum_cs['counts'])

sel1 = spectrum_cs.query('pulseheight >= 1000 and pulseheight <= 1500')

model = models.GaussianModel() + models.LinearModel()
fit = model.fit(sel1['counts'], x=sel1['pulseheight'], weights=1/sel1['y_err'], center=1200, slope=0)
# fit.plot(numpoints=100)
center_cs = fit.params['center']
# print(center_cs.value, center_cs.stderr)

spectrum_na = pd.read_csv('Na22_spectrum.csv')
spectrum_na['y_err'] = np.sqrt(spectrum_na['counts'])

sel2 = spectrum_na.query('pulseheight >= 800 and pulseheight <= 1200')

fit2 = model.fit(sel2['counts'], x=sel2['pulseheight'], weights=1/sel2['y_err'], center=980, sigma=40, amplitude=1000, slope=0)
# fit2.plot(numpoints=100)
center_na_1 = fit2.params['center']
# print(center_na_1.value, center_na_1.stderr)

sel3 = spectrum_na.query('pulseheight >= 2200 and pulseheight <= 2700')

fit3 = model.fit(sel3['counts'], x=sel3['pulseheight'], weights=1/sel3['y_err'], center=2400, sigma=40, amplitude=110, slope=0)
# fit3.plot(numpoints=100)
center_na_2 = fit3.params['center']
# print(center_na_2.value, center_na_2.stderr)

df = pd.DataFrame([
    [0.5110, center_na_1.value, center_na_1.stderr],
    [0.6617, center_cs.value, center_cs.stderr],
    [1.2746, center_na_2.value, center_na_2.stderr],
], columns=['Energy', 'Pulseheight', 'PH_err'])

model2 = models.LinearModel()
fit4 = model2.fit(df['Pulseheight'], x=df['Energy'], weights=1/df['PH_err'], slope=2000)
# fit4.plot()
# plt.xlim(0, 1.5)
# plt.ylim(0, 4000)

f = lambda x, a: a*x
model3 = models.Model(f, name="linear_origin")

fit5 = model3.fit(df['Pulseheight'], x=df['Energy'], weights=1/df['PH_err'], a=2000)
# fit5.plot()
# plt.xlim(0, 1.5)
# plt.ylim(0, 4000)
# plt.show()

slope = fit4.params['slope']
intercept = fit4.params['intercept']

spectrum_cs['energy'] = (spectrum_cs['pulseheight'] - intercept)/slope
spectrum_na['energy'] = (spectrum_na['pulseheight'] - intercept)/slope

# spectrum_cs.plot.scatter('energy', 'counts', yerr='y_err')
# spectrum_na.plot.scatter('energy', 'counts', yerr='y_err')
# plt.show()

## opdracht 1.3

amplitude = fit.params['center']
FWHM = fit.params['fwhm']

resolution = (FWHM/amplitude.value)*100

print(resolution)

# gebruik numpy.argmax om plek van maximum te berekenen in dataset

