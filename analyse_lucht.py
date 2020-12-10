## LET OP: foutenanalyse onvolledig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
import lmfit

spectrum_vacuum = pd.read_csv('ijkmetingen/ijkmeting_19_11_2.csv')

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
max_energies_err = []
distances = []
distances_err = []
average_pressures = []
average_pressures_err = []

atm = 1013.249977

pressures = [50, 100, 120, 150, 200, 300, 450, 470, 490, 500]
pressures_err = [2, 3, 5, 5, 5, 10, 15, 15, 15, 15]

pressures_2 = [60.3, 111.2, 131.4, 166.9, 224.8, 352.3, 499.1, 519.0, 536.6, 554.2]
pressures_2_err = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

for index in range(len(pressures)):
    spectrum = pd.read_csv(f'data_lucht/{pressures[index]}mbar_meting_lucht.csv')

    pulseheights = spectrum['pulseheight'].tolist()
    counts = spectrum['counts'].tolist()

    max_index = np.argmax(counts)
    max_pulseheight = pulseheights[max_index]
    max_pulseheight_err = max_pulseheight - pulseheights[max_index - 1]

    max_energy = max_pulseheight / pulse_per_mev
    max_energy_err = max_pulseheight_err / pulse_per_mev
    max_energies.append(max_energy)
    max_energies_err.append(max_energy_err)

    average_pressure = (pressures[index] + pressures_2[index]) / 2
    average_pressure_err = 0.5 * np.sqrt((pressures_err[index])**2 + (pressures_2_err[index])**2)
    average_pressures.append(average_pressure)
    average_pressures_err.append(average_pressure_err)

    factor = atm / average_pressure
    factor_err = (atm / (average_pressure)**2) * average_pressure_err

    distance = 5 / factor
    distance_err = np.sqrt((0.05 / factor)**2 + ((5 / factor**2) * factor_err)**2)
    distances.append(distance)
    distances_err.append(distance_err)

# plt.scatter(distances, max_energies)
# plt.xlabel("distance (cm)")
# plt.ylabel("max remaining energy (MeV)")
# plt.show()

measurements = {
    'Max Energy': max_energies,
    'Max Energy Error': max_energies_err,
    'Distance': distances
}

df_2 = pd.DataFrame(measurements)

linear = lambda x, a, b: a*x + b
model2 = models.Model(linear, name="linear")

fit_2 = model2.fit(df_2['Max Energy'], x=df_2['Distance'], weights=1/df_2['Max Energy Error'], a=-1, b=5.5)
fit_2.plot()
plt.show()

stopping_power_air = fit_2.params['a'].value
stopping_power_air_kev = stopping_power_air * -1000

stopping_power_air_err = fit_2.params['a'].stderr
stopping_power_air_kev_err = stopping_power_air_err * 1000

print(f'The stopping power of helium is {stopping_power_air_kev:.3f} KeV +/- {stopping_power_air_kev_err:.3f} KeV')

