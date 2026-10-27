# Outbreak Surveillance Dataset: 2014 Ebola Epidemic — Sierra Leone

## Where to Download the Real-World Data

### Primary Source (Recommended)
**Caitlin Rivers' curated Ebola dataset on GitHub**
- **URL**: https://github.com/cmrivers/ebola
- **File**: `country_timeseries.csv` — contains daily cumulative case and death counts for Sierra Leone, Liberia, and Guinea.
- **Columns to extract**: `Date`, `Cases_SierraLeone`, `Deaths_SierraLeone`
- **Licence**: Public domain / open research use.
- **Why this source**: This is the most widely cited machine-readable Ebola dataset in published research (Rivers et al., 2014). It aggregates WHO situation reports into a clean time-series format.

### Alternative Source 1
**Humanitarian Data Exchange (HDX)**
- **URL**: https://data.humdata.org/dataset/ebola-cases-2014
- **Format**: CSV with country-level daily and cumulative counts.
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### Alternative Source 2
**WHO Ebola Situation Reports Archive**
- **URL**: https://www.who.int/ebola/situation-reports/archive
- **Format**: PDF situation reports (must be manually digitised or cross-referenced with the GitHub dataset above).
- **Note**: Situation reports are the original authoritative source; the GitHub dataset above is derived from these.

### Alternative Source 3
**Our World in Data — Ebola**
- **URL**: https://github.com/owid/etl/tree/master/etl/steps/data/garden/who/2024-12-03/ebola
- **Format**: Processed CSV with standardised columns.

---

## How to Prepare the Data

After downloading `country_timeseries.csv` from the primary source:

```python
import pandas as pd

# 1. Load and filter for Sierra Leone
df = pd.read_csv('country_timeseries.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Cases_SierraLeone', 'Deaths_SierraLeone']].dropna()
df = df.sort_values('Date').reset_index(drop=True)

# 2. Select a ≈120-day window covering the first wave
#    Recommended: 25 May 2014 — 21 Sep 2014 (120 days)
start = pd.Timestamp('2014-05-25')
end   = pd.Timestamp('2014-09-21')
df = df[(df['Date'] >= start) & (df['Date'] <= end)].reset_index(drop=True)

# 3. Compute daily increments and 7-day average
df['day'] = range(len(df))
df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
df['cumulative_cases'] = df['Cases_SierraLeone'].astype(int)
df['cumulative_deaths'] = df['Deaths_SierraLeone'].astype(int)
df['daily_cases'] = df['cumulative_cases'].diff().fillna(df['cumulative_cases'].iloc[0]).clip(lower=0).astype(int)
df['daily_deaths'] = df['cumulative_deaths'].diff().fillna(df['cumulative_deaths'].iloc[0]).clip(lower=0).astype(int)
df['daily_cases_7day_avg'] = df['daily_cases'].rolling(7, center=True, min_periods=1).mean().round(1)

# 4. 75/25 chronological split
split_day = int(len(df) * 0.75)
df['split'] = ['train' if i < split_day else 'test' for i in range(len(df))]

# 5. Save
df[['day','date','daily_cases','daily_cases_7day_avg','cumulative_cases',
    'daily_deaths','cumulative_deaths','split']].to_csv('data/outbreak_data.csv', index=False)
```

---

## Expected Output Schema

| Column | Description |
|---|---|
| `day` | Integer index from $t = 0$ to $t = 119$ |
| `date` | Calendar date in ISO 8601 (`YYYY-MM-DD`) |
| `daily_cases` | Confirmed + probable daily EVD cases |
| `daily_cases_7day_avg` | 7-day rolling centered mean |
| `cumulative_cases` | Cumulative confirmed + probable cases $C(t)$ |
| `daily_deaths` | Confirmed Ebola-attributed daily deaths |
| `cumulative_deaths` | Cumulative disease deaths $D(t)$ |
| `split` | `train` (days 0–89) or `test` (days 90–119) |

## Provenance Metadata
- **Geography**: Republic of Sierra Leone ($N = 7{,}092{,}113$, 2014 census estimate).
- **Timeframe**: May 25, 2014 to September 21, 2014 ($T = 120$ days).
- **Access Date**: _(fill in when downloaded)_
- **Key Epidemiological References**:
  - WHO Ebola Response Team, "Ebola Virus Disease in West Africa — The First 9 Months," *NEJM* 371(16), 2014.
  - Legrand et al., "Understanding the dynamics of Ebola epidemics," *Epidemiology & Infection* 135(4), 2007.
  - Rivers et al., "Modeling the Impact of Interventions on an Epidemic of Ebola in Sierra Leone and Liberia," *PLOS Currents Outbreaks*, 2014.
  - Althaus, "Estimating the Reproduction Number of Ebola Virus (EBOV) During the 2014 Outbreak in West Africa," *PLOS Currents Outbreaks*, 2014.

---

## Second dataset (Module 4, Experiment E7): WHO, Sierra Leone, whole epidemic 2014-15

| Item | Value |
|---|---|
| Publisher | World Health Organization, Ebola situation reports, on the Humanitarian Data Exchange (HDX) |
| Dataset page | https://data.humdata.org/dataset/ebola-cases-2014 |
| File | `ebola_data_db_format.csv`, saved unchanged as `data/who_hdx_ebola_data_db_format.csv` |
| Download URL | https://data.humdata.org/dataset/0d089fa0-3567-4b01-9c03-39d340ff34e3/resource/c59b5722-ca4b-41ca-a446-472d6d824d01/download/ebola_data_db_format.csv |
| SHA-256 of the file | `053c325038b419619162482b8ab1e98bed66d80cc1929ed7862367fc9bf4b2fa` |
| Accessed | 26 October 2026 |
| Series used | `Cumulative number of confirmed Ebola cases` and `Cumulative number of confirmed Ebola deaths`, Sierra Leone |
| Window | 29 Aug 2014 (first Sierra Leone entry) to 6 Nov 2015 (last report before WHO declared Sierra Leone Ebola-free on 7 Nov 2015): 238 report dates |
| Processed file | `data/sierra_leone_who_2014_2015.csv`, built by `python data_pipeline_who.py` |

**Why laboratory-confirmed counts.** The combined "confirmed, probable and suspected" totals drop on 5 Nov 2014 because WHO changed its data sources after the 29 Oct 2014 situation report (stated in the HDX dataset notes), and suspected cases were later reclassified in bulk. The confirmed series rises smoothly from 935 to 8,704 cases, and its final deaths/cases ratio (3,589 / 8,704 = 41.2%) equals the case-fatality ratio used throughout the project.

**Cleaning.** A cumulative count cannot fall, so a report larger than a later report was revised downwards afterwards; such reports are flagged (`cases_used`, `deaths_used` = False) and not fitted. This removes no case reports and 6 death reports (4 October 2014 values later revised down, and 2 dates with no death figure). **Nothing is interpolated**: the PINN fits the 238 reports at their actual dates.

**Checks against the Module 5 file.** On 5 Sep 2014 the WHO combined total (1,261 cases, 491 deaths) equals the cmrivers value exactly; later the cmrivers values run 100-260 cases higher, consistent with the two sources dating the same reports a few days apart. The two series are therefore used separately, not joined.

**Checks on the Module 5 file (cmrivers).** `data/country_timeseries_raw.csv` is byte-for-byte the content of https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv (commit `2c696c79c72c49ab7a178bf4f4e391613aa42f18`, 9 Jan 2015), apart from Windows line endings. Its Sierra Leone values match WHO Disease Outbreak News of 31 Jul 2014 (533 cases, 233 deaths as of 27 Jul) and 22 Aug 2014 (910 cases, 392 deaths as of 20 Aug). Only 37 of the 120 days in `outbreak_data.csv` are real report dates (36 after dropping one June report whose deaths WHO later revised down); the rest are linear interpolation. In E5, every method (the PINN and both baselines) is fitted to and scored on only the real report dates.
