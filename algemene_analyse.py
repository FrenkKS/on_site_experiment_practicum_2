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

measurements_air = {
    'Max Energy': max_energies,
    'Max Energy Error': max_energies_err,
    'Distance': distances
}

df_air = pd.DataFrame(measurements_air)

linear = lambda x, a, b: a*x + b
model2 = models.Model(linear, name="linear")

fit_air = model2.fit(df_air['Max Energy'], x=df_air['Distance'], weights=1/df_air['Max Energy Error'], a=-1, b=5.5)
fit_air.plot()
plt.show()

stopping_power_air = fit_air.params['a'].value
stopping_power_air_kev = stopping_power_air * -1000

stopping_power_air_err = fit_air.params['a'].stderr
stopping_power_air_kev_err = stopping_power_air_err * 1000

print(f'The stopping power of air is {stopping_power_air_kev:.3f} KeV +/- {stopping_power_air_kev_err:.3f} KeV')


max_energies_helium = []
max_energies_helium_err = []
distances_helium = []
distances_helium_err = []
average_pressures_15cm = []
average_pressures_15cm_err = []
average_pressures_20cm = []
average_pressures_20cm_err = []

pressures_15cm = [30, 50, 100, 120, 150, 200, 300, 450, 600, 800, 820, 840, 860]
pressures_15cm_err = [2, 3, 5, 5, 10, 10, 15, 15, 20, 20, 20, 20, 20]

pressures_15cm_2 = [37.9, 61.1, 112.4, 133.5, 170.1, 230.4, 363.4, 509.6, 674.4, 875.9, 883.3, 909.1, 930.1]
pressures_15cm_2_err = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

for index in range(len(pressures_15cm)):
    spectrum = pd.read_csv(f'data_helium/{pressures_15cm[index]}mbar_meting_helium.csv')

    pulseheights = spectrum['pulseheight'].tolist()
    counts = spectrum['counts'].tolist()

    max_index = np.argmax(counts)
    max_pulseheight = pulseheights[max_index]
    max_pulseheight_err = max_pulseheight - pulseheights[max_index - 1]

    max_energy = max_pulseheight / pulse_per_mev
    max_energy_err = max_pulseheight_err / pulse_per_mev
    max_energies_helium.append(max_energy)
    max_energies_helium_err.append(max_energy_err)

    average_pressure = (pressures_15cm[index] + pressures_15cm_2[index]) / 2
    average_pressure_err = 0.5 * np.sqrt((pressures_15cm_err[index])**2 + (pressures_15cm_2_err[index])**2)
    average_pressures_15cm.append(average_pressure)
    average_pressures_15cm_err.append(average_pressure_err)

    factor = atm / average_pressure
    factor_err = (atm / (average_pressure)**2) * average_pressure_err

    distance = 15 / factor
    distance_err = np.sqrt((0.05 / factor)**2 + ((15 / factor**2) * factor_err)**2)
    distances_helium.append(distance)
    distances_helium_err.append(distance_err)

pressures_20cm = [750, 770]
pressures_20cm_err = [30, 30]

pressures_20cm_2 = [794.0, 820.1]
pressures_20cm_2_err = [3, 3]

for index in range(len(pressures_20cm)):
    spectrum = pd.read_csv(f'data_helium/{pressures_20cm[index]}mbar_meting_helium_20cm.csv')

    pulseheights = spectrum['pulseheight'].tolist()
    counts = spectrum['counts'].tolist()

    max_index = np.argmax(counts)
    max_pulseheight = pulseheights[max_index]
    max_pulseheight_err = max_pulseheight - pulseheights[max_index - 1]

    max_energy = max_pulseheight / pulse_per_mev
    max_energy_err = max_pulseheight_err / pulse_per_mev
    max_energies_helium.append(max_energy)
    max_energies_helium_err.append(max_energy_err)

    average_pressure = (pressures_20cm[index] + pressures_20cm_2[index]) / 2
    average_pressure_err = 0.5 * np.sqrt((pressures_20cm_err[index])**2 + (pressures_20cm_2_err[index])**2)
    average_pressures_20cm.append(average_pressure)
    average_pressures_20cm_err.append(average_pressure_err)

    factor = atm / average_pressure
    factor_err = (atm / (average_pressure)**2) * average_pressure_err

    distance = 20 / factor
    distance_err = np.sqrt((0.05 / factor)**2 + ((20 / factor**2) * factor_err)**2)
    distances_helium.append(distance)
    distances_helium_err.append(distance_err)

# plt.scatter(distances, max_energies)
# plt.xlabel("distance (cm)")
# plt.ylabel("max remaining energy (MeV)")
# plt.show()

measurements_helium = {
    'Max Energy': max_energies_helium,
    'Max Energy Error': max_energies_helium_err,
    'Distance': distances_helium
}

df_helium = pd.DataFrame(measurements_helium)

fit_helium = model2.fit(df_helium['Max Energy'], x=df_helium['Distance'], weights=1/df_helium['Max Energy Error'], a=-1, b=5.5)
fit_helium.plot()
plt.show()

# print(lmfit.report_fit(fit_2))

stopping_power_helium = fit_helium.params['a'].value
stopping_power_helium_kev = stopping_power_helium * -1000

stopping_power_helium_err = fit_helium.params['a'].stderr
stopping_power_helium_kev_err = stopping_power_helium_err * 1000

print(f'The stopping power of helium is {stopping_power_helium_kev:.3f} KeV +/- {stopping_power_helium_kev_err:.3f} KeV')

stopping_powers_factor = stopping_power_air / stopping_power_helium

theoretical_stopping_powers_factor = 1142.5 / 163.85

print(f'The calculated stopping power ratio air:helium is {stopping_powers_factor:.2f}')
print(f'The theoretical stopping power ratio air:helium is {theoretical_stopping_powers_factor:.2f}')


c = 1 - (0.32)**(1/10)
print(c)