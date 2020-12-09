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
distances = []

atm = 1013.249977

pressures = [50, 100, 120, 150, 200, 300, 450, 470, 490, 500]
pressures_2 = [60.3, 111.2, 131.4, 166.9, 224.8, 352.3, 499.1, 519.0, 536.6, 554.2]

average_pressures = []

for index in range(len(pressures)):
    spectrum = pd.read_csv(f'data_lucht/{pressures[index]}mbar_meting_lucht.csv')

    pulseheights = spectrum['pulseheight'].tolist()
    counts = spectrum['counts'].tolist()

    max_index = np.argmax(counts)
    max_pulseheight = pulseheights[max_index]
    max_energy = max_pulseheight / pulse_per_mev
    max_energies.append(max_energy)

    average_pressure = (pressures[index] + pressures_2[index]) / 2
    average_pressures.append(average_pressure)

    factor = atm / average_pressure

    distance = 5 / factor
    distances.append(distance)

plt.scatter(distances, max_energies)
plt.xlabel("distance (cm)")
plt.ylabel("max remaining energy (MeV)")
plt.show()

print(average_pressures)

