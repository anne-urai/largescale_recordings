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

# original spreadsheet from Ian Stevenson:  https://stevenson.lab.uconn.edu/scaling/
# df = pd.read_csv('https://docs.google.com/spreadsheet/pub?hl=en_US&hl=en_US&key=0Ai7vcDJIlD6AdF9vQWlNRDh2S1dub09jMWRvTFRpemc&single=true&gid=0&output=csv')

# instead, use my own sheet with additional datapoints for imaging
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQdv2uGPz4zSZmfpiIUrHvpB90Cz6cs8rgObbAqNQmsaLb5moGg8sYlIvfSZvXhoh1R1id8lZFyASkC/pub?gid=1390826946&single=true&output=csv')
print(df.describe())

# add some things - like the date for the x-axis
df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + '01')
df['years'] = (df['date'] - datetime(1950, 1, 1)).dt.days / 365
df['date_num'] = df['Year'] + (df['Month']-1)/12
df['neurons_log'] = np.log(df['Neurons']) # take log

# ====================================== #
# refit the curve from Stevenson et al. 2011
# ====================================== #

# separate out data for fit to original papers
fit_data = df[(df['Source'] == 'S&K')].copy()

# from https://github.com/ihstevenson/scaling/blob/master/scaling.py:
# Only keep first M papers to record >=N neurons
tmp = fit_data.groupby(['Neurons'])['DOI'].nunique().reset_index()
assert(all(tmp['DOI'] <= 10))

# use patsy
y, X = dmatrices('neurons_log ~ date_num', data=fit_data, return_type='dataframe')
mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model

# what's the doubling time from this model? log(2) / a
doubling_time = np.log(2) / res.params['date_num']
print('Doubling time S&K: %f years'%doubling_time)

# extrapolate to whenever
xvec1 = np.linspace(2000, 2500, 100)
yvec1 = res.predict(sm.add_constant(xvec1))

# ====================================== #
# also fit on all data, including imaging
# ====================================== #

fit_data2 = df[(df['Method'] == 'Imaging')].copy()

# use patsy
y2, X2 = dmatrices('neurons_log ~ date_num', data=fit_data2, return_type='dataframe')
mod2 = sm.OLS(y2, X2)    # Describe model
res2 = mod2.fit()       # Fit model
print(res2.summary())   # Summarize model

# what's the doubling time from this model? log(2) / a
doubling_time2 = np.log(2) / res2.params['date_num']
print('Doubling time imaging: %f years'%doubling_time2)

# extrapolate to whenever
yvec2 = res2.predict(sm.add_constant(xvec1))

# ====================================== #
# SHOW SOME TARGET NUMBERS FOR NEURONS IN DIFFERENT SPECIES
# ====================================== #

# Herculano-Houzel et al. 2015, 10.1159/000437413
nneurons = [{'species':'Caenorhabditis elegans', 'name':'C. elegans',
             'nneurons_low':302, 'nneurons_high':302},
            {'species': 'Danio rerio (larvae)', 'name': 'Zebrafish (larva)',  # https://elifesciences.org/articles/28158
             'nneurons_low': 100000, 'nneurons_high': 100000},
            # {'species':'Drosophila melanogaster', 'name':'Drosophila', # https://doi.org/10.1016/j.cub.2010.11.056
            #  'nneurons_low':135000, 'nneurons_high':135000},
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3597383/
            {'species':'Mus musculus', 'name':'Mouse', # Vincent et al 2010: 7.5x10^7
             'nneurons_low':67873741-10406194, 'nneurons_high':67873741+10406194},
            # {'species':'Rattus norvegicus', 'name':'Rat',
            #  'nneurons_low':188867832-12622383, 'nneurons_high':188867832+12622383},
            {'species': 'Macaca mulatta', 'name': 'Macaque',
             'nneurons_low': 6376160000, 'nneurons_high': 6376160000},
            {'species': 'Homo sapiens', 'name': 'Human',
             'nneurons_low': 86060000000-8120000000, 'nneurons_high': 86060000000+8120000000},
            ]

# ====================================== #
# for each species, when will record
# from all their neurons?
# ====================================== #

for sp in nneurons:
    avg_log = np.log((sp['nneurons_low'] + sp['nneurons_high']) / 2)
    max_year = xvec1[np.abs(yvec1-avg_log).argmin()]
    min_year = xvec1[np.abs(yvec2-avg_log).argmin()]
    print('%s: expected %d - %d'%(sp['name'], min_year, max_year))

# ====================================== #
# make the plot
# ====================================== #

fig, ax = plt.subplots(1, 1, figsize=[5, 3.5])
sns.scatterplot(data=df, x='date_num', y='neurons_log', style='Source',
                hue='Method', zorder=0, s=10, linewidths=0.5, alpha=0.5,
                palette=sns.color_palette(["firebrick", "midnightblue"]),
                hue_order=['Imaging', 'Ephys'],
                markers={'S&K':'s', 'Stevenson':'o', 'Urai':'o',
                         'Rupprecht':'o', 'Charles':'o', 'Meijer':'o',
                         'Svoboda':'o'}, legend=False)
# write labels in plot, instead of legend
ax.text(2004, np.log(2), 'Electrophysiology',
        {'color':"midnightblue", 'fontsize':9, 'fontstyle':'italic'})
ax.text(1985, np.log(1000), 'Optical\nimaging',
        {'color':"firebrick", 'fontsize':9, 'fontstyle':'italic'})

# plot Stevenson curve on top
ax.plot(X['date_num'], res.predict(), color='k')

# then show extrapolation beyond 2011; to now
xvec = df[df['date_num'] > 1960]['date_num']
yvec = res.predict(sm.add_constant(xvec))
ax.plot(xvec, yvec, color='k', linestyle='--')

# and finally, all the way out into the future
xvec = np.linspace(2020, 2025, 100)
yvec = res.predict(sm.add_constant(xvec))
ax.plot(xvec, yvec, color='k', linestyle=':')

# show, for each species, the range
for a in nneurons:
    # when can we expect this species to have all its neurons recorded?
    year = 1958
    n_neurons = np.log((a['nneurons_low'] + a['nneurons_high'])/2)
    ax.axhline(y=n_neurons, color='grey', linestyle=':', zorder=-100, xmax=0.92)
    ax.text(year, n_neurons, a['name'],
            verticalalignment='bottom', fontsize=7,
            color='grey')

# layout
yticks = np.logspace(0, 11, 12)
ax.set(ylabel='Simultaneously recorded neurons', xlabel='',
       yticks=np.log(yticks))
yticklabs = ['1', '10', '100', '1k', '10k', '100k', '1m', '10m', '100m', '1b', '10b', '100b']
ax.set_yticklabels(['$\mathregular{10^{%i}}$' %np.log10(y) for y in yticks])
ax.set_yticklabels(yticklabs)

sns.despine(trim=True)
plt.show()
fig.savefig('scaling_figure.pdf')
fig.savefig('scaling_figure.png', dpi=600)

