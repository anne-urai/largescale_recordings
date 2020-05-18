"""
Anne Urai, CSHL, 2020-05-17
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from patsy import dmatrices
from datetime import datetime
import statsmodels.api as sm

# layout
sns.set(style="ticks", context="paper")
sns.despine(trim=True)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ====================================== #
# read data from Ian Stevenson
# ====================================== #

df = pd.read_csv('https://docs.google.com/spreadsheet/pub?hl=en_US&hl=en_US&key=0Ai7vcDJIlD6AdF9vQWlNRDh2S1dub09jMWRvTFRpemc&single=true&gid=0&output=csv')
print(df.describe())

# add some things - like the date for the x-axis
df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + '01')
df['years'] = (df['date'] - datetime(1950, 1, 1)).dt.days / 365
df['date_num'] = df['Year'] + (df['Month']-1)/12
df['neurons_log'] = np.log(df['Neurons']) # log

# ====================================== #
# refit the curve from Stevenson et al. 2011
# ====================================== #

y, X = dmatrices('neurons_log ~ date_num', data=df[(df['date_num'] > 1960) & (df['date_num'] < 2010)],
                 return_type='dataframe')
mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model

# what's the doubling time from this model? log(2) /
doubling_time = np.log(2) / res.params['date_num']
print('Doubling time: %f years'%doubling_time)

# ====================================== #
# SHOW SOME TARGET NUMBERS FOR NEURONS IN DIFFERENT SPECIES
# ====================================== #

# Herculano-Houzel et al. 2015, 10.1159/000437413
nneurons = [{'species':'Caenorhabditis elegans', 'name':'Worm',
             'nneurons_low':302, 'nneurons_high':302},
            {'species': 'Danio rerio', 'name': 'Zebrafish',  # https://elifesciences.org/articles/28158
             'nneurons_low': 100000, 'nneurons_high': 100000},
            {'species':'Drosophila melanogaster', 'name':'Fly', # https://doi.org/10.1016/j.cub.2010.11.056
             'nneurons_low':135000, 'nneurons_high':135000},
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3597383/
            {'species':'Mus musculus', 'name':'Mouse',
             'nneurons_low':67873741-10406194, 'nneurons_high':67873741+10406194},
            {'species':'Rattus norvegicus', 'name':'Rat',
             'nneurons_low':188867832-12622383, 'nneurons_high':188867832+12622383},
            {'species': 'Macaca mulatta', 'name': 'Monkey',
             'nneurons_low': 6376160000, 'nneurons_high': 6376160000},
            {'species': 'Homo sapiens', 'name': 'Human',
             'nneurons_low': 86060000000-8120000000, 'nneurons_high': 86060000000+8120000000},
            ]

# ====================================== #
# make the plot
# ====================================== #

fig, ax = plt.subplots(1, 1, figsize=[5, 3.5])
ax.scatter(df.date_num, df.neurons_log, marker='.', color='grey', zorder=-100)

# plot Stevenson curve on top
plt.plot(X['date_num'], res.predict(), color='k')

# then show extrapolation beyond 2011; to 2020
xvec = df[df['date_num'] > 1960]['date_num']
yvec = res.predict(sm.add_constant(xvec))
plt.plot(xvec, yvec, color='k', linestyle='--')

# and finally, all the way out into the future
xvec = np.linspace(2020, 2030, 100)
yvec = res.predict(sm.add_constant(xvec))
plt.plot(xvec, yvec, color='k', linestyle=':')

# show, for each species, the range
for a in nneurons:
    # when can we expect this species to have all its neurons recorded?
    year = 2040
    n_neurons = np.log((a['nneurons_low'] + a['nneurons_high'])/2)
    ax.axhline(y=n_neurons, color='grey', linestyle=':')
    ax.text(year, n_neurons, a['species'],
            verticalalignment='bottom', fontsize=8, fontstyle='italic')

# layout
yticks = np.logspace(0, 11, 12)
ax.set(ylabel='Simultaneously recorded neurons', xlabel='',
       yticks=np.log(yticks))
ax.set_yticklabels(['$\mathregular{10^{%i}}$' %np.log10(y) for y in yticks])
sns.despine(trim=True)
plt.show()
fig.savefig('scaling_figure.pdf')
fig.savefig('scaling_figure.png', dpi=600)

