# Copyright (c) 2020 Spanish National Research Council
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from multiprocessing.pool import ThreadPool
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import PATHS
from download import install

# Install xlrd >= 1.0.0 for Excel support
install('xlrd')


cant_popul = 581078


def fix_1207():
    """
    Fix error in mobility dataset of Spain from INE (Instituto Nacional de Estadística).

    """
    rawdir = PATHS.rawdir / 'maestra1' / 'municipios'
    src = rawdir / '20200705_maestra_1_mitma_municipio.txt.gz'
    dst = rawdir / '20200712_maestra_1_mitma_municipio.txt.gz'

    df = pd.read_csv(src,
                     sep='|',
                     thousands='.',
                     dtype={'origen': 'string', 'destino': 'string'},
                     compression='gzip')

    # Replace date
    df['fecha'] = '20200712'

    # Apply thousands separator
    def add_sep(x):
        x = str(x)
        if len(x) > 3:
            return f'{str(x)[:-3]}.{str(x)[-3:]}'
        else:
            return x

    df['viajes'] = df['viajes'].apply(add_sep)
    df['viajes_km'] = df['viajes_km'].apply(add_sep)

    df.to_csv(dst,
              sep='|',
              compression='gzip',
              index=False)


def process_day(tarfile):
    """
    Process daily mobility files from INE.
    Args:
        tarfile [str, DataFrame]: Absolute path of mobility file.

    Returns: Mobility dataframe.

    """
    try:
        df = pd.read_csv(tarfile,
                         sep='|',
                         thousands='.',
                         dtype={'origen': 'string', 'destino': 'string'},
                         compression='gzip')
    except Exception as e:
        print(f'Error processing {tarfile}')
        raise Exception(e)

    df['fecha'] = pd.to_datetime(df.fecha, format='%Y%m%d')
    df['fecha'] = df['fecha'].dt.date

    # Aggregate data inside same province
    df['origen'] = df['origen'].transform(lambda x: x[:2])
    df['destino'] = df['destino'].transform(lambda x: x[:2])

    # Aggregate across hours and distances
    df = df.groupby(['fecha', 'origen', 'destino']).sum().reset_index()
    df = df.drop(['periodo', 'viajes_km'], 'columns')

    return df


def process_mobility(day_files='all',
                     exp='maestra1',
                     res='municipios',
                     update=False,
                     force=False):
    """
    Process daily mobility data. If 'all' is passed, it will process every file.
    Args:
        day_files [list, str]:  List of absolute paths to day-tars to process.
        exp: Folder.
        res: Folder inside exp.
        update: Update the data as of today's date
        force: To overwrite already downloaded data

    Returns: DataFrame of mobility flow data between autonomous communities in Spain.

    """

    # Prepare files
    rawdir = PATHS.rawdir / f'{exp}' / f'{res}'
    if day_files == 'all':
        day_files = sorted(os.listdir(rawdir))
        day_files = [rawdir / d for d in day_files]

    if not day_files:
        day_files = sorted(day_files)

    # Load INE code_map to add names to the tables
    cod_path = PATHS.rawdir / f'{exp}' / '20_cod_prov.xls'
    cod_map = pd.read_excel(cod_path, skiprows=4, names=['codigo', 'literal'], dtype={'codigo': str})
    cod_map = dict(zip(cod_map.codigo, cod_map.literal))

    # Load existing files
    if update and (PATHS.processed / f'{exp}' / "province_flux.csv").exists():
        previous_df = pd.read_csv(PATHS.processed / f'{exp}' / "province_flux.csv")

        # Compare dates
        last = datetime.datetime.strptime(previous_df['date'].iloc[-1], '%Y-%m-%d')
        first = datetime.datetime.strptime(day_files[0].name[:8], '%Y%m%d')

        if last + datetime.timedelta(days=1) != first and not force:
            raise Exception(f'The last day ({last.date()}) saved and the first day ({first.date()}) '
                            'of the update are not consecutive. '
                            'You will probably be safer rerunning the download  '
                            'and running the whole processing from scratch (update=False). '
                            'You can use force=True if you want to append the data '
                            'nevertheless.')
    else:
        previous_df = pd.DataFrame([])

    # Parallelize the processing for speed
    print('Processing mobility data...')
    thread_num = 4
    results = []
    out = ThreadPool(thread_num).imap(process_day, day_files)
    for r in tqdm(out, total=len(day_files)):
        results.append(r)
    full_df = pd.concat(results).reset_index()

    # Clean and add id codes
    full_df = full_df.rename(columns={'fecha': 'date',
                                      'origen': 'province id origin',
                                      'destino': 'province id destination',
                                      'viajes': 'flux'})

    full_df['province origin'] = full_df['province id origin'].map(cod_map)
    full_df['province destination'] = full_df['province id destination'].map(cod_map)
    full_df = full_df[['date', 'province origin', 'province id origin',
                       'province destination', 'province id destination', 'flux']]
    dates_format = np.repeat("%Y/%m/%d", len(full_df.index), axis=0)
    full_df['date'] = list(map(datetime.datetime.strftime, full_df['date'], dates_format))

    # Append and save
    full_df = pd.concat([previous_df, full_df])
    save_path = PATHS.processed / f'{exp}'
    save_path.exists() or os.makedirs(save_path)
    full_df.to_csv(f'{save_path}/province_flux.csv', index=False)

    return full_df


def process_sc(*argv):
    """
    Process historical covid-19 file of Cantabria (Spain) from Cantabrian Health Service.
    Args:
        argv [DataFrame]: DataFrame to process. If nothing is passed, it will process from default folder.

    Returns: DataFrame processed.
    """

    fpath = PATHS.rawdir / 'covid' / 'region' / 'historical_cantb.csv'
    save_path = PATHS.processed / 'covid' / 'region' / 'historical_cantb.csv'
    try:
        try:
            df = argv[0]
        except Exception as e:
            df = pd.read_csv(fpath, sep=',')

        df = df.fillna(0)
        df.drop(['% CRECIMIENTO COVID*',
                 'CASOS RESIDENCIAS',
                 'C. RESID. ACTIVOS',
                 'PROF. SANITARIOS',
                 'P. SANIT. ACTIVOS',
                 'HOSP. VALDECILLA',
                 'HOSP. LIENCRES',
                 'HOSP. LAREDO',
                 'HOSP. SIERRALLANA',
                 'HOSP. TRES MARES',
                 'UCI HUMV',
                 'UCI SIERRALLANA',
                 'TEST*100.000 HAB.',
                 'TEST PCR',
                 'TEST*100.000 HAB..1',
                 'TEST ANTICUERPOS',
                 'TEST*100.000 HAB..2',
                 'TEST ANTICUERPOS +',
                 'TEST ANTIGENOS',
                 'TEST*100.000 HAB..3',
                 'TEST ANTIGENOS +',
                 'INCIDENCIA AC 14',
                 'CASOS 7 DIAS',
                 'CASOS 14 DIAS'],
                axis='columns', inplace=True)
        df = df.rename(columns={'FECHA': 'date',
                                'TOTAL CASOS': 'cumulative_cases',
                                'CASOS NUEVOS PCR*': 'daily_cases',
                                '% MEDIA MOVIL 7 DIÂAS*': 'moving_average_7',
                                'TOTAL HOSPITALIZADOS': 'hospital_occ',
                                'HOSPITALIZADOS UCI': 'icu_occ',
                                'FALLECIDOS': 'cumulative_deaths',
                                'TOTAL TEST': 'cumulative_total_tests'})
        # New variables
        # daily deaths
        df['daily_deaths'] = df['cumulative_deaths'].fillna(0).diff()
        # daily occupancy of hospital beds
        newhosp = df['hospital_occ'].fillna(0).diff()
        newhosp[newhosp < 0] = 0
        df['new_hospital_cases'] = newhosp
        # daily occupancy of ICU beds
        newicu = df['icu_occ'].fillna(0).diff()
        newicu[newicu < 0] = 0
        df['new_icu_cases'] = newicu
        # daily total tests
        df['daily_total_tests'] = df['cumulative_total_tests'].fillna(0).diff()
        # total cases in 7 and 14 days
        cases7 = df['cumulative_cases'].to_numpy()[7:] - df['cumulative_cases'].to_numpy()[:-7]
        df['cases7'] = np.append(np.repeat(0, 7), cases7)
        cases14 = df['cumulative_cases'].to_numpy()[14:] - df['cumulative_cases'].to_numpy()[:-14]
        df['cases14'] = np.append(np.repeat(0, 14), cases14)
        # cumulative incidence in 7 and 14 days
        df['incidence7'] = df['cases7'] * 100000 / cant_popul
        df['incidence7'] = df['incidence7'].apply(np.ceil)
        df['incidence14'] = df['cases14'] * 100000 / cant_popul
        df['incidence14'] = df['incidence14'].apply(np.ceil)

        # Fix some issues
        df = df.fillna(0)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(np.int64)

        # daily positivity
        df['daily_positivity'] = df['daily_cases'] / df['daily_total_tests']
        df['daily_positivity'] = df['daily_positivity'].fillna(0)
        df.loc[0, 'daily_positivity'] = 0

        # Save
        df.to_csv(f'{save_path}', index=False, header=True)

        return df

    except Exception as e:
        print(f'Error processing {fpath}')
        raise Exception(e)


def process_icane(*argv):
    """
    Process covid-19 file of Cantabria (Spain) from ICANE.
    Args:
        argv [DataFrame]: DataFrame to process. If nothing is passed, it will process from default folder.

    Returns: DataFrame processed.
    """

    fpath = PATHS.rawdir / 'covid' / 'region' / 'all_data_cantb.csv'
    save_path = PATHS.processed / 'covid' / 'region' / 'all_data_cantb.csv'
    try:
        try:
            df = argv[0]
        except Exception as e:
            df = pd.read_csv(fpath, sep=',').copy()

        # New variables and names
        df = df.fillna(0)
        df['hospital_occ'] = df[['Laredo', 'Liencres', 'Sierrallana', 'Tres Mares', 'Valdecilla']].sum(axis=1)
        df['daily_total_tests'] = df[['Test Anticuerpos diarios', 'Test Antigenos diarios',
                                      'Test PCR diarios']].sum(axis=1)

        df.drop(['Recuperados',
                 'Laredo',
                 'Liencres',
                 'Sierrallana',
                 'Tres Mares',
                 'Valdecilla',
                 'Cuantil 0,025 (R)',
                 'Cuantil 0,975 (R)',
                 'Media (R)',
                 'Dosis entregadas Pfizer (1)',
                 'Dosis entregadas Moderna (1)',
                 'Total Dosis entregadas (1)',
                 'Dosis administradas (2)',
                 '% sobre entregadas',
                 'Fecha de la última vacuna registrada (2)',
                 'Dosis entregadas AstraZeneca (1)',
                 'Nº Personas con al menos 1 dosis',
                 'Dosis entregadas Janssen (1)',
                 'UCI Sierrallana',
                 'UCI Valdecilla',
                 'Positividad',
                 'Incidencia 14 dias'],
                axis='columns', inplace=True)

        df = df.rename(columns={'Unnamed: 0': 'date',
                                'Casos': 'daily_cases',
                                'Casos.1': 'cumulative_cases',
                                'Fallecidos': 'daily_deaths',
                                'Fallecidos.1': 'cumulative_deaths',
                                'UCI': 'icu_occ',
                                # 'Positividad': 'positivity',
                                'Nº Personas vacunadas(pauta completada)': 'vaccinated_pp',
                                'Test Anticuerpos diarios': 'daily_antibody_tests',
                                'Test Antigenos diarios': 'daily_antigen_tests',
                                'Test PCR diarios': 'daily_pcr_tests'})

        # New variables
        # total cases in 7 and 14 days
        cases7 = df['cumulative_cases'].to_numpy()[7:] - df['cumulative_cases'].to_numpy()[:-7]
        df['cases7'] = np.append(np.repeat(0, 7), cases7)
        cases14 = df['cumulative_cases'].to_numpy()[14:] - df['cumulative_cases'].to_numpy()[:-14]
        df['cases14'] = np.append(np.repeat(0, 14), cases14)
        # cumulative incidence in 7 and 14 days
        df['incidence7'] = df['cases7'] * 100000 / cant_popul
        df['incidence7'] = df['incidence7'].apply(np.ceil)
        df['incidence14'] = df['cases14'] * 100000 / cant_popul
        df['incidence14'] = df['incidence14'].apply(np.ceil)
        # daily occupancy of hospital beds
        newhosp = df['hospital_occ'].fillna(0).diff()
        newhosp[newhosp < 0] = 0
        df['new_hospital_cases'] = newhosp
        # daily occupancy of ICU beds
        newicu = df['icu_occ'].fillna(0).diff()
        newicu[newicu < 0] = 0
        df['new_icu_cases'] = newicu

        # Fix some empty value errors
        df.iloc[0, 1] = 1.0
        df = df.fillna(0)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(np.int64)

        # daily positivity
        df['daily_positivity'] = df['daily_cases'] / df['daily_total_tests']
        df['daily_positivity'] = df['daily_positivity'].fillna(0)
        df.loc[0, 'daily_positivity'] = 0

        # Save
        df.to_csv(f'{save_path}', index=False, header=True)

        return df

    except Exception as e:
        print(f'Error processing {fpath}')
        print(e)
        raise Exception(e)

    # TODO: Process covid-19 data of the municipalities.


if __name__ == '__main__':
    # process_mobility(day_files='all',
    #                  exp='maestra1',
    #                  res='municipios',
    #                  update=False,
    #                  force=False)
    process_sc()
    process_icane()
