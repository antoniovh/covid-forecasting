Covid19 - TFM
===========
Implementation of Machine Learning models to forecast epidemic spread, ICU beds and deaths caused by COVID-19 in Cantabria. Using mobility and daily COVID-19 data.

This project has been based on:
1. https://github.com/IFCA/mitma-covid
2. https://github.com/IFCA/covid-risk-map
3. https://github.com/IFCA/covid-dl

[comment]: <> (4. https://github.com/saul-torres/covid_cantabria)

[comment]: <> (5. https://github.com/midudev/covid-vacuna)

[comment]: <> (   https://github.com/midudev/covid-vacuna/blob/main/scripts/download-covid-vaccine-today-status.js)

## Datasets
Data obtained from:
1. [Mobility study with Big Data](https://www.mitma.gob.es/ministerio/covid-19/evolucion-movilidad-big-data)
   developed by the Ministry of Transport (MITMA) for the study of measures against COVID-19:
   https://opendata-movilidad.mitma.es/.
2. Cantabria Health Service: https://www.scsalud.es/coronavirus.
3. Cantabria Institute of Statistics (ICANE):
   https://www.icane.es/covid19/dashboard/home/home.
4. Ministry of Health, Consumer Affairs and Social Welfare: https://www.mscbs.gob.es.

## Generate the data

To generate and process mobility and COVID-19 data from scratch use:
```bash
python src/data.py
python src/data.py --files all
```
To generate only mobility data:
```bash
python src/data.py --files mitma
```
This will download the data from the first date (February 21, 2020) to the last date of the study (May 09, 2021) and process it. If you already have the data downloaded and processed from a previous time and just want to update it with the recent data, you can use:
```bash
python src/data.py --files mitma --update
```
This will look at what is the last downloaded day. Then it will download and process all subsequent days.

To generate COVID-19 and vaccination data for Cantabria and its municipalities use:
```bash
python src/data.py --files covid
```
Downloaded files will be stored in the folder ``data/raw`` and processed files in ``data/processed``.
