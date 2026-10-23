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
