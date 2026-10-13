#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: ARIMA tools for model selection
subtitle:
version: 1.0
type: module
keywords: [ARIMA, statsmodels]
description: |
remarks:
todo:
sources:
    - title: ARIMA Model – Complete Guide to Time Series Forecasting in Python
      link: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
      date: 2019-02-18
      authors:
          - nick: selva86
            fullname: Selva Prabhakaran
            email:
      usage: |
          ideas & inspiration
    - link: https://github.com/selva86/datasets
file:
    date: 2020-09-08
    authors:
"""

# %%
"""
pwd
cd ~/Projects/Python/RCanDo/...
ls
"""
# %%
# from dateutil.parser import parse
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd

import statsmodels.api as sm
# import statsmodels.stats as st
"""
better use  sm.stats  as e.g. there is everything from  st.diagnostic
"""
import statsmodels.tsa.api as ts
# import statsmodels.tsa.stattools as tst
# import statsmodels.graphics.tsaplots as tsp

from pmdarima.arima.utils import ndiffs

from scipy import stats

# %%
import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)

# %%


class ModelARIMA():
    def __init__(self, time_series, order):
        self.model = ts.ARIMA(time_series, order)
        self.fit = self.model.fit(disp=-1)
        self.params = pd.DataFrame({'param': self.fit.params,
                                    'p-val': self.fit.pvalues})
        self.gofs = pd.Series({'AIC': self.fit.aic,
                               'BIC': self.fit.bic,
                               'HQIC': self.fit.hqic})
        self._resids_tests()

    @property
    def data(self):
        data = self.model.data.endog
        if self.model.data.dates is not None:
            idx = self.model.data.dates
        else:
            idx = range(len(data))
        return pd.Series(data, index=idx)

    def _resids_tests(self):
        # Shapiro-Wilk - normality via order stats
        self.sw = stats.shapiro(self.fit.resid)
        # Jarque-Bera - normality via skew and kurtosis
        self.jb = sm.stats.jarque_bera(self.fit.resid)
        # Ljung-Box - ACF=0 up to given lag
        self.lb = sm.stats.acorr_ljungbox(self.fit.resid, lags=[10], return_df=True)
        # Breusch-Godfrey - as above but by Lagrange Multipliers (and designed rather for Regression Model)
        self.bg = sm.stats.acorr_breusch_godfrey(self.fit)
        # Durbin-Waatson - is AR(1) == r1<>0; dw ~ 2(1-r1) \in [0, 4], close to 2 means r1~=0
        self.dw = sm.stats.durbin_watson(self.fit.resid)

    @property
    def resids_pvals(self):
        return pd.Series({'sw': self.sw[1],
                          'jb': self.jb[1],
                          'bg': self.bg[1],
                          'lb-10': self.lb['lb_pvalue'].values[0],
                          'dw': self.dw
                          })

    @property
    def summary(self):
        return self.fit.summary()

    def forecast(self, steps=5, plot=False):

        fc, se, conf = self.fit.forecast(steps, alpha=.05)
        fc_df = pd.DataFrame(conf, columns=['lower', 'upper'])
        fc_df['fc'] = fc

        idx = self.data.index
        if isinstance(self.data.index, pd.DatetimeIndex):
            idx = idx[-steps:].shift(steps, freq=idx.freq)
        else:
            start = idx[-1] + 1
            idx = range(start, start + steps)
        fc_df.index = idx

        if plot:
            plt.figure(figsize=(7, 5), dpi=120)
            plt.title("Forecasts")
            plt.plot(self.data)
            plt.plot(fc_df.fc, label='forecast')
            plt.fill_between(fc_df.index, fc_df.lower, fc_df.upper, color='k', alpha=.15)
            plt.legend(loc="upper left", fontsize=8)
            plt.show()

        return fc_df

    def test(self, test, plot=False):

        fc, se, conf = self.fit.forecast(len(test), alpha=.05)
        fc_df = pd.DataFrame(conf, columns=['lower', 'upper'], index=test.index)
        fc_df['fc'] = fc
        fc_df['test'] = test

        mape = np.mean(np.abs(fc - test) / np.abs(test))  # MAPE
        me = np.mean(fc - test)             # ME
        mae = np.mean(np.abs(fc - test))    # MAE
        mpe = np.mean((fc - test) / test)     # MPE
        rmse = np.mean((fc - test)**2)**.5  # RMSE
        corr = np.corrcoef(fc, test)[0, 1]   # corr
        mins = np.amin(np.hstack([fc[:, None],
                                  test[:, None]]), axis=1)
        maxs = np.amax(np.hstack([fc[:, None],
                                  test[:, None]]), axis=1)
        minmax = 1 - np.mean(mins / maxs)             # minmax
        acf1 = ts.acf(fc - test)[1]                 # ACF1
        accuracy = pd.Series({'mape': mape, 'me': me, 'mae': mae,
                              'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
                              'corr': corr, 'minmax': minmax
                              })

        if plot:
            plt.figure(figsize=(7, 5), dpi=120)
            plt.title("Forecasts vs Actual")
            plt.plot(self.data, label='training')
            plt.plot(fc_df.fc, label='forecast')
            plt.fill_between(fc_df.index, fc_df.lower, fc_df.upper, color='k', alpha=.15)
            plt.plot(test, label='test')
            plt.legend(loc="upper left", fontsize=8)
            plt.show()

        return fc_df, accuracy


class SearchARIMA():
    def __init__(self, train, test=None, p=None, d=None, q=None):
        self.train = train
        self.test = test
        self._set_orders(p, d, q)

        self._search_models()

    def _set_orders(self, p, d, q):
        if p is None:
            p = list(range(2))
        if d is None:
            adf = ndiffs(self.train, test='adf')    # Augmented Dickey-Fuller  (unit root exists)
            kpss = ndiffs(self.train, test='kpss')  # KPSS                     (trend stationarity)
            pp = ndiffs(self.train, test='pp')      # Philips-Perron           (integrated 1)
            d = list(range(max(adf, kpss, pp)))
        if q is None:
            q = list(range(2))

        self.p = p if isinstance(p, list) else list(range(p + 1))
        self.d = d if isinstance(d, list) else list(range(d + 1))
        self.q = q if isinstance(q, list) else list(range(q + 1))

    @property
    def _orders_prod(self):
        return len(self.p) * len(self.d) * len(self.q)

    def _search_models(self):
        self.models = pd.DataFrame({'model': [None] * self._orders_prod},
                                   index=pd.MultiIndex.from_product([self.p, self.d, self.q], names=list('pdq'))) \
            .drop([(0, d, 0) for d in self.d], errors='ignore')
        for o in self.models.index:
            try:
                self.models.loc[o, 'model'] = ModelARIMA(self.train, order=o)
                print(o)
            except Exception as e:
                self.models.loc[o, 'model'] = None
                print("{} : {}".format(o, e))

    def get_model(self, idx):
        return self.models.loc[idx, 'model']

    def __getitem__(self, idx):
        return self.models.loc[idx, 'model']

    @property
    def pvalues(self):
        def remove(s):
            s = "" if s[0] == 'D' or s in ['y', 'value'] else s.lstrip('L')
            return s

        def pvalues(row):
            if row[0] is not None:
                pvals = pd.Series(row[0].fit.pvalues).round(3)
                # simplifying index names
                pvals.index = ["".join([remove(s) for s in i.split(".")]) for i in pvals.index]
            else:
                pvals = None
            return pvals

        return self.models.apply(pvalues, axis=1)

    @property
    def statistics(self):
        def gofs(row):
            return row[0].gofs.round(3) if row[0] is not None else None
        stats = self.models.apply(gofs, axis=1)

        def restests(row):
            return row[0].resids_pvals.round(3) if row[0] is not None else None
        stats = pd.concat([stats, self.models.apply(restests, axis=1)], axis=1)

        def accuracy(row):
            return row[0].test(self.test)[1].round(3) if row[0] is not None else None
        if self.test is not None:
            stats = pd.concat([stats, self.models.apply(accuracy, axis=1)], axis=1)

        return stats

    @property
    def summary(self):
        return pd.concat([self.statistics, self.pvalues], axis=1)

    def plot_data(self):
        plt.figure(figsize=(7, 5))
        plt.title("data")
        plt.plot(self.train, label='training')
        if self.test is not None:
            plt.plot(self.test, label='test')
        plt.show()
