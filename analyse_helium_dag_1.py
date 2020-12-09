## LET OP: foutenanalyse onvolledig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
import lmfit

spectrum_vacuum = pd.read_csv('ijkmetingen/ijkmeting_3_12.csv')

pulseheights_ijk = spectrum_vacuum['pulseheight'].tolist()
counts_ijk = spectrum_vacuum['counts'].tolist()

# binwidths = []

# for i in range(0, len(pulseheights)):
#     if i is not len(pulseheights) - 1:
#         binwidth = pulseheights[i + 1] - pulseheights[i]
#         binwidths.append(round(binwidth, 5))

# print(binwidths)

max_index_ijk = np.argmax(counts_ijk)
max_pulseheight_ijk = pulseheights_ijk[max_index_ijk]

max_pulseheight_ijk_err = max_pulseheight_ijk - pulseheights_ijk[max_index_ijk - 1]

print(f'The maximum pulseheight is {max_pulseheight_ijk:.3f} +/- {max_pulseheight_ijk_err:.3f}')

f = lambda x, a: a*x
model = models.Model(f, name="linear_origin")

df = pd.DataFrame([
    [5.4857, max_pulseheight_ijk, max_pulseheight_ijk_err],
], columns=['Energy', 'Pulseheight', 'PH_err'])

fit = model.fit(df['Pulseheight'], x=df['Energy'], weights=1/df['PH_err'], a=40)
# fit.plot()
# plt.show()

# print(lmfit.report_fit(fit))

pulse_per_mev = fit.params['a'].value

max_energies = []
distances = []
average_pressures_15cm = []
average_pressures_20cm = []

atm = 1013.249977

pressures_15cm = [30, 50, 100, 120, 150, 200, 300, 450, 600, 800, 820, 840, 860]
pressures_15cm_2 = [37.9, 61.1, 112.4, 133.5, 170.1, 230.4, 363.4, 509.6, 674.4, 875.9, 883.3, 909.1, 930.1]

for index in range(len(pressures_15cm)):
    spectrum = pd.read_csv(f'data_helium/{pressures_15cm[index]}mbar_meting_helium.csv')

    pulseheights = spectrum['pulseheight'].tolist()
    counts = spectrum['counts'].tolist()

    max_index = np.argmax(counts)
    max_pulseheight = pulseheights[max_index]
    max_energy = max_pulseheight / pulse_per_mev
    max_energies.append(max_energy)

    average_pressure = (pressures_15cm[index] + pressures_15cm_2[index]) / 2
    average_pressures_15cm.append(average_pressure)

    factor = atm / average_pressure
    distance = 15 / factor
    distances.append(distance)

pressures_20cm = [750, 770]
pressures_20cm_2 = [794.0, 820.1]

for index in range(len(pressures_20cm)):
    spectrum = pd.read_csv(f'data_helium/{pressures_20cm[index]}mbar_meting_helium_20cm.csv')

    pulseheights = spectrum['pulseheight'].tolist()
    counts = spectrum['counts'].tolist()

    max_index = np.argmax(counts)
    max_pulseheight = pulseheights[max_index]
    max_energy = max_pulseheight / pulse_per_mev
    max_energies.append(max_energy)

    average_pressure = (pressures_20cm[index] + pressures_20cm_2[index]) / 2
    average_pressures_20cm.append(average_pressure)

    factor = atm / average_pressure
    distance = 20 / factor
    distances.append(distance)

# plt.scatter(distances, max_energies)
# plt.xlabel("distance (cm)")
# plt.ylabel("max remaining energy (MeV)")
# plt.show()

measurements = {
    'Max Energy': max_energies,
    'Distance': distances
}

df_2 = pd.DataFrame(measurements)

linear = lambda x, a, b: a*x + b
model2 = models.Model(linear, name="linear")

fit_2 = model2.fit(df_2['Max Energy'], x=df_2['Distance'], a=-1, b=5.5)
fit_2.plot()
plt.show()

# print(lmfit.report_fit(fit_2))

stopping_power_helium = fit_2.params['a'].value
stopping_power_helium_kev = stopping_power_helium * 1000

print(stopping_power_helium_kev)