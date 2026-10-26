"""
data_pipeline_who.py
--------------------
Builds the full-epidemic Sierra Leone Ebola series (29 Aug 2014 - 7 Nov 2015)
used by the Module 4 PINN real-data validation.

Source: World Health Organization, Ebola situation reports, as published on the
Humanitarian Data Exchange (HDX):
    Dataset  "Number of Ebola Cases and Deaths in Affected Countries"
             https://data.humdata.org/dataset/ebola-cases-2014
    File     ebola_data_db_format.csv
    URL      https://data.humdata.org/dataset/0d089fa0-3567-4b01-9c03-39d340ff34e3/resource/c59b5722-ca4b-41ca-a446-472d6d824d01/download/ebola_data_db_format.csv
    SHA-256  053c325038b419619162482b8ab1e98bed66d80cc1929ed7862367fc9bf4b2fa
             (file as downloaded on 26 Oct 2026; saved unchanged in data/)

Series used: laboratory-confirmed cumulative cases and deaths for Sierra Leone.
Confirmed counts are used instead of "confirmed, probable and suspected" because
suspected cases were later reclassified in bulk (WHO changed its data sources
after the 29 Oct 2014 situation report, which makes the combined total drop on
5 Nov 2014). The confirmed series has a case-fatality ratio of 41.2% at the end of
the epidemic, the value used throughout the project.

Cleaning: a cumulative count cannot fall, so a report that is larger than a
later report was revised downwards afterwards. Such reports are dropped (the
later, revised value is kept). Nothing is interpolated: only real report dates
are kept, and the PINN fits them at their actual times.
"""

import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RAW_PATH = os.path.join(DATA_DIR, 'who_hdx_ebola_data_db_format.csv')
PROCESSED_PATH = os.path.join(DATA_DIR, 'sierra_leone_who_2014_2015.csv')
WHO_HDX_URL = ('https://data.humdata.org/dataset/0d089fa0-3567-4b01-9c03-39d340ff34e3/'
               'resource/c59b5722-ca4b-41ca-a446-472d6d824d01/download/ebola_data_db_format.csv')
RAW_SHA256 = '053c325038b419619162482b8ab1e98bed66d80cc1929ed7862367fc9bf4b2fa'

POPULATION_SIERRA_LEONE = 7_092_113
START_DATE = '2014-08-29'   # first Sierra Leone entry in the WHO file
END_DATE = '2015-11-07'     # WHO declared Sierra Leone Ebola-free on 7 Nov 2015

CASES = 'Cumulative number of confirmed Ebola cases'
DEATHS = 'Cumulative number of confirmed Ebola deaths'
ALL_CASES = 'Cumulative number of confirmed, probable and suspected Ebola cases'
ALL_DEATHS = 'Cumulative number of confirmed, probable and suspected Ebola deaths'


def download_raw(path=RAW_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(WHO_HDX_URL, path)
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != RAW_SHA256:
        print(f"[data_pipeline_who] WARNING: SHA-256 {digest} differs from the recorded file")
    return path


def _keep_monotone(values):
    """True for reports that are not larger than any later report."""
    v = np.asarray(values, float)
    later_min = np.minimum.accumulate(np.where(np.isnan(v), np.inf, v)[::-1])[::-1]
    return ~np.isnan(v) & (v <= later_min)


def build_sierra_leone_series(raw_path=RAW_PATH, out_path=PROCESSED_PATH):
    download_raw(raw_path)
    df = pd.read_csv(raw_path)
    sl = df[df['Country'] == 'Sierra Leone'].copy()
    sl['Date'] = pd.to_datetime(sl['Date'])
    wide = sl.pivot_table(index='Date', columns='Indicator', values='value', aggfunc='last')
    out = pd.DataFrame({
        'date': wide.index,
        'cumulative_cases': wide[CASES].values,
        'cumulative_deaths': wide[DEATHS].values,
        'cumulative_cases_all': wide[ALL_CASES].values,
        'cumulative_deaths_all': wide[ALL_DEATHS].values,
    })
    out = out[(out['date'] >= START_DATE) & (out['date'] <= END_DATE)].reset_index(drop=True)
    out['day'] = (out['date'] - pd.Timestamp(START_DATE)).dt.days
    out['cases_used'] = _keep_monotone(out['cumulative_cases'])
    out['deaths_used'] = _keep_monotone(out['cumulative_deaths'])
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    cols = ['day', 'date', 'cumulative_cases', 'cumulative_deaths', 'cases_used', 'deaths_used',
            'cumulative_cases_all', 'cumulative_deaths_all']
    out = out[cols]
    out.to_csv(out_path, index=False)
    print(f"[data_pipeline_who] {len(out)} report dates, {out['date'].iloc[0]} to {out['date'].iloc[-1]}")
    print(f"[data_pipeline_who] dropped as later revised: {int((~out['cases_used']).sum())} case reports, "
          f"{int((~out['deaths_used']).sum())} death reports")
    return out


def load_sierra_leone_series(path=PROCESSED_PATH):
    if not os.path.exists(path):
        return build_sierra_leone_series(out_path=path)
    return pd.read_csv(path)


if __name__ == '__main__':
    build_sierra_leone_series()
