# Outbreak Surveillance Dataset: 2014-16 Ebola Epidemic, Sierra Leone

All real-data experiments (E5 and E7) use one dataset: the WHO Ebola situation reports for Sierra Leone, as published on the Humanitarian Data Exchange (HDX). Only real report dates are used; **nothing is interpolated**. The notebook (`Final_MM_notebook.ipynb`, "Code 2") downloads the file, checks its SHA-256 and builds the processed series.

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
| Population | $N = 7{,}092{,}113$ (2014 census estimate) |
| Processed file | `data/sierra_leone_who_2014_2015.csv` (columns: `day` since 29 Aug 2014, `date`, cumulative cases and deaths, `cases_used`, `deaths_used`, and the combined totals for reference) |

**Why laboratory-confirmed counts.** The combined "confirmed, probable and suspected" totals drop on 5 Nov 2014 because WHO changed its data sources after the 29 Oct 2014 situation report (stated in the HDX dataset notes), and suspected cases were later reclassified in bulk. The confirmed series rises smoothly from 935 to 8,704 cases, and its final deaths/cases ratio (3,589 / 8,704 = 41.2%) equals the case-fatality ratio used throughout the project.

**Cleaning.** A cumulative count cannot fall, so a report larger than a later report was revised downwards afterwards. Such reports are flagged (`cases_used`, `deaths_used` = False) and not fitted. This removes no case reports and 6 death reports (4 October 2014 values later revised down, and 2 dates with no death figure).

**How the experiments use it.**
- **E5 (held-out test):** 29 Aug 2014 - 25 Apr 2015 (days 0-239). Training: days 0-179 (to 24 Feb 2015, the first 75%). Test: days 180-239 (25 Feb - 25 Apr 2015, the last 25%).
- **E7 (rolling forecasts):** 8 origins every 28 days from 16 Jan to 31 Jul 2015; each fit uses the previous 8 weeks of reports and forecasts the next 4 weeks. Three earlier origins (24 Oct, 21 Nov, 19 Dec 2014) were used only to choose the PINN's settings.
