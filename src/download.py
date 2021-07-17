import os
import urllib.parse
import datetime
import subprocess
import sys
from tqdm import tqdm
import requests
from requests.packages import urllib3
import urllib
import json
import pandas as pd
import numpy as np

from utils import PATHS

# Disable SSL verification warnings warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Install odfpy to convert OpenDocuments (.ods) to a pandas DataFrame
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install('odfpy')


def download_ine(exp='maestra1',
                 res='municipios',
                 update=False,
                 force=False):
    """
    Download mobility data of Spain from INE (Instituto Nacional de Estadística).
    Args:
        exp: folder. See base_url.
        res: folder inside exp. See base_url.
        update: update the data as of today's date
        force: To overwrite already downloaded data

    Returns: Downloaded files in your repository and a list of path files.

    """

    # Prepare output dir
    files = []
    rawdir = PATHS.rawdir / f'{exp}' / f'{res}'
    rawdir.exists() or os.makedirs(rawdir)

    # Generate time range
    lsfiles = sorted(os.listdir(rawdir))
    if update:
        start = datetime.datetime.strptime(lsfiles[-1][:8], '%Y%m%d').date()
        start += datetime.timedelta(days=1)
    else:
        start = datetime.date(2020, 2, 21)
    end = datetime.datetime.today().date()
    dates = pd.date_range(start, end, freq='d')

    if dates.empty:
        print('Already up-to-date')
        return []

    # Download files
    print(f'Downloading mobility files for the period {start} - {end}')
    s = requests.Session()
    base_url = 'https://opendata-movilidad.mitma.es'

    for d in tqdm(dates):
        url = f'{base_url}/{exp}-mitma-{res}/ficheros-diarios/{d:%Y}-{d:%m}/{d:%Y%m%d}_maestra_{exp[-1]}_mitma_' \
              f'{res[:-1]}.txt.gz'

        aux = urllib.parse.urlparse(url)
        fpath = os.path.basename(aux.path)  # file name
        fpath = rawdir / fpath

        if fpath.exists() and not force:
            print(f"\t {os.path.basename(url)} already downloaded, not overwriting it."
                  "To overwrite it use (force=True)")
            continue

        try:
            resp = s.get(url, verify=False)
            if resp.status_code == 404:
                print(f'{d.date()} not available yet')
                continue
            with open(fpath, 'wb') as f:
                f.write(resp.content)
            files.append(fpath)

        except Exception as e:
            print(f'Error downloading {url}')
            print(e)

    return files


def download_sc(subject='historico'):
    """
    Download covid-19 data of Cantabria (Spain) from Cantabrian Health Service.
    Args:
        subject: Type of file you want to download. Options: historico, municipalizado, edadysexo.
                    See base_url because this variable is important for the url.

    Returns: Downloaded files in your repository and a list of path files.

    """

    # Prepare output dir
    files = []
    rawdir = PATHS.rawdir / 'covid' / 'region'
    rawdir.exists() or os.makedirs(rawdir)

    # Download files
    s = requests.Session()
    url = f'https://serviweb.scsalud.es:10443/ficheros/COVID19_{subject}.csv'
    print(f'Downloading historical covid-19 data from Cantabrian Health Service.')

    aux = urllib.parse.urlparse(url)
    fpath = os.path.basename(aux.path)  # file path
    fpath = rawdir / fpath

    try:
        resp = s.get(url, verify=False)
        if resp.status_code == 404:
            print('File not available')

        with open(fpath, 'wb') as f:
            f.write(resp.content)
        files.append(fpath)

    except Exception as e:
        print(f'Error downloading {url}')
        print(e)

    return files


def download_json(territory, name_var, file_path):
    """
    Auxiliary function.
    Download and save covid data of Cantabria (Spain) from json files of a project of Instituto Cántabro de Estadística
    (ICANE): https://www.icane.es/covid19/dashboard/home/home. Not for all files, all the information is
    in https://covid19icane.firebaseio.com/.json.

    Args:
        territory (str): Geographical accuracy. Options: region, mpio.
        name_var (str): Name of variable and the downloaded file.
        file_path (str): Path to save the file.

    Returns: DataFrame of downloaded file.

    """

    if territory == 'region':
        url = f'https://covid19can-data.firebaseio.com/saludcantabria/{name_var}.json'
    else:  # Municipios
        url = f'https://covid19can-data.firebaseio.com/saludcantabria/municipios/{territory}/{name_var}' \
              f'.json'
        url = urllib.parse.urlsplit(url)
        url = list(url)
        url[2] = urllib.parse.quote(url[2])
        url = urllib.parse.urlunsplit(url)

    try:
        response = urllib.request.urlopen(url)
        file_json = json.loads(response.read())
        # Dates
        dates = file_json['dimension']['Fecha']['category']['label'].keys()
        dates_format = np.repeat("%d-%m-%Y", len(dates), axis=0)
        dates = sorted(list(map(datetime.datetime.strptime, dates, dates_format)))
        dates_format = np.repeat("%Y/%m/%d", len(dates), axis=0)
        dates = list(map(datetime.datetime.strftime, dates, dates_format))  # Sorted and converted to date format
        # Variables
        if "age" in name_var:
            nvar = len(file_json['dimension']['Rango_edad']['category']['index'])
            names = list(file_json['dimension']['Rango_edad']['category']['index'].keys())  # Name of variables
        else:
            nvar = len(file_json['dimension']['Variables']['category']['index'])  # Number of variables
            names = list(file_json['dimension']['Variables']['category']['index'].keys())  # Name of variables
        values = np.array(file_json['value'], dtype='float')  # Values of all variables
        values = np.absolute(values)  # Process data a little
        values = values.reshape((len(dates), nvar))  # Correct format
        file = pd.DataFrame(values,
                            columns=names,
                            index=dates)
        if territory == 'region':
            file.to_csv(f'{file_path}/{name_var}.csv', index=True, header=True)
        else:
            nmun = territory.split(" - ")[1].replace(" ", "_")
            file.to_csv(f'{file_path}/{name_var}_{nmun}.csv', index=True, header=True)
        return file

    except Exception as e:
        print(f'Error downloading {url}')
        print(e)


def vaccine(date):
    """
    Auxiliary function.
    Download data about vaccination in Cantabria  from the Ministry of Health, Consumer Affairs and Social Welfare.
    https://www.mscbs.gob.es
    Args:
        date(str): Date in format %Y%m%d

    Returns: DataFrame with vaccination data from first day (2021/02/04) to the present day.

    """

    try:
        prefix_url = 'https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/documentos/' \
                     'Informe_Comunicacion_'
        suffix_url = '.ods'
        nfile = f'{prefix_url}{date}{suffix_url}'
        file_vaccine = pd.read_excel(nfile, engine='odf')
        file_vaccine.set_index('Unnamed: 0', inplace=True)
        vcant = file_vaccine.loc['Cantabria']
        vcant = pd.DataFrame(vcant).T
        vcant.index = [datetime.datetime.strptime(date, "%Y%m%d").strftime("%Y/%m/%d")]
        return vcant

    except Exception as e:
        date = datetime.datetime.strptime(date, "%Y%m%d").strftime("%Y/%m/%d")
        print(f"Error downloading vaccination data for {date}")
        # print(e)


def download_icane(exp='region',
                   variable='all'):
    """
    Download covid-19 temporal data of Cantabria (Spain) from json files of a project of Instituto Cántabro de
    Estadística (ICANE): https://www.icane.es/covid19/dashboard/home/home.
    All the information is in https://covid19icane.firebaseio.com/.json

    Args:
        exp: Geographical accuracy. Options: region, mpio.
        variable: Most important covid-19 variables for downloading.
                    Options with exp='region': all, vaccine, daily-cases, daily-deceases, daily-discharged, accumulated,
                    daily-test, hospitalizations, ucis, incidence. Also: elder, sanitarians, positivity, rho, dates,
                    sma, daily-types, cases-age, deceased-age, hospitalizations-age, uci-age.
                    Options with exp='mpio': casos-diarios, fallecidos, incidencia14, incidencia7.

    Returns: Downloaded files from covid-19.

    """

    # Prepare output dir
    files = []
    rawdir = PATHS.rawdir / 'covid' / f'{exp}'
    rawdir.exists() or os.makedirs(rawdir)
    processdir = PATHS.processed / 'covid' / f'{exp}'
    processdir.exists() or os.makedirs(processdir)

    if exp == 'region':
        print(f'Downloading covid-19 data of Cantabria from ICANE.')

        if variable == 'all':
            name_var = ['daily-cases', 'daily-deceases', 'daily-discharged', 'accumulated', 'daily-test',
                        'hospitalizations', 'ucis', 'incidence', 'rho']
            fpath = np.repeat(rawdir, len(name_var), axis=0)
            exp = np.repeat(exp, len(name_var), axis=0)
            covid_cant = list(map(download_json, exp, tqdm(name_var), fpath))
            covid_cant = pd.concat(covid_cant, axis=1)
            vac_cant = download_icane(exp='region', variable='vaccine')
            file_all = pd.concat([covid_cant, vac_cant], axis=1)
            # Save all variables of Cantabria
            file_all.to_csv(f'{rawdir}/all_data_cantb.csv', index=True, header=True)
            file_all.to_csv(f'{processdir}/all_data_cantb.csv', index=True, header=True)
            return file_all

        elif variable == 'vaccine':
            # Dates
            start = datetime.date(2021, 2, 4)
            end = datetime.datetime.today().date()
            dates_time = pd.bdate_range(start, end).strftime("%Y%m%d")
            vaccines = list(map(vaccine, tqdm(dates_time)))
            file_vaccines = pd.concat(vaccines)
            # Save vaccination data
            file_vaccines.to_csv(f'{rawdir}/vaccine_cantb.csv', index=True, header=True)
            file_vaccines.to_csv(f'{processdir}/vaccine_cantb.csv', index=True, header=True)
            return file_vaccines

        else:
            file_var = download_json(exp, variable, rawdir)
            return file_var

    elif exp == 'mpio':
        print(f'Downloading covid-19 data of the municipalities of Cantabria from ICANE.')

        name_mun = ["39001 - ALFOZ DE LLOREDO", "39002 - AMPUERO", "39003 - ANIEVAS", "39004 - ARENAS DE IGUÑA",
                    "39005 - ARGOÑOS", "39006 - ARNUERO", "39007 - ARREDONDO", "39008 - ASTILLERO (EL)",
                    "39009 - BARCENA DE CICERO", "39010 - BARCENA DE PIE DE CONCHA", "39011 - BAREYO",
                    "39012 - CABEZON DE LA SAL", "39013 - CABEZON DE LIEBANA", "39014 - CABUERNIGA",
                    "39015 - CAMALEÑO", "39016 - CAMARGO", "39017 - CAMPOO DE YUSO", "39018 - CARTES",
                    "39019 - CASTAÑEDA", "39020 - CASTRO-URDIALES", "39021 - CIEZA", "39022 - CILLORIGO DE LIEBANA",
                    "39023 - COLINDRES", "39024 - COMILLAS", "39025 - CORRALES DE BUELNA (LOS)",
                    "39026 - CORVERA DE TORANZO", "39027 - CAMPOO DE ENMEDIO", "39028 - ENTRAMBASAGUAS",
                    "39029 - ESCALANTE", "39030 - GURIEZO", "39031 - HAZAS DE CESTO",
                    "39032 - HERMANDAD DE CAMPOO DE SUSO", "39033 - HERRERIAS", "39034 - LAMASON",
                    "39035 - LAREDO", "39036 - LIENDO", "39037 - LIERGANES", "39038 - LIMPIAS", "39039 - LUENA",
                    "39040 - MARINA DE CUDEYO", "39041 - MAZCUERRAS", "39042 - MEDIO CUDEYO", "39043 - MERUELO",
                    "39044 - MIENGO", "39045 - MIERA", "39046 - MOLLEDO", "39047 - NOJA", "39048 - PENAGOS",
                    "39049 - PEÑARRUBIA", "39050 - PESAGUERO", "39051 - PESQUERA", "39052 - PIELAGOS",
                    "39053 - POLACIONES", "39054 - POLANCO", "39055 - POTES", "39056 - PUENTE VIESGO",
                    "39057 - RAMALES DE LA VICTORIA", "39058 - RASINES", "39059 - REINOSA", "39060 - REOCIN",
                    "39061 - RIBAMONTAN AL MAR", "39062 - RIBAMONTAN AL MONTE", "39063 - RIONANSA",
                    "39064 - RIOTUERTO", "39065 - ROZAS DE VALDEARROYO (LAS)", "39066 - RUENTE", "39067 - RUESGA",
                    "39068 - RUILOBA", "39069 - SAN FELICES DE BUELNA", "39070 - SAN MIGUEL DE AGUAYO",
                    "39071 - SAN PEDRO DEL ROMERAL", "39072 - SAN ROQUE DE RIOMIERA",
                    "39073 - SANTA CRUZ DE BEZANA", "39074 - SANTA MARIA DE CAYON", "39075 - SANTANDER",
                    "39076 - SANTILLANA DEL MAR", "39077 - SANTIURDE DE REINOSA", "39078 - SANTIURDE DE TORANZO",
                    "39079 - SANTOÑA", "39080 - SAN VICENTE DE LA BARQUERA", "39081 - SARO", "39082 - SELAYA",
                    "39083 - SOBA", "39084 - SOLORZANO", "39085 - SUANCES", "39086 - TOJOS (LOS)",
                    "39087 - TORRELAVEGA", "39088 - TRESVISO", "39089 - TUDANCA", "39090 - UDIAS",
                    "39091 - VALDALIGA", "39092 - VALDEOLEA", "39093 - VALDEPRADO DEL RIO",
                    "39094 - VALDERREDIBLE", "39095 - VAL DE SAN VICENTE", "39096 - VEGA DE LIEBANA",
                    "39097 - VEGA DE PAS", "39098 - VILLACARRIEDO", "39099 - VILLAESCUSA", "39100 - VILLAFUFRE",
                    "39101 - VILLAVERDE DE TRUCIOS", "39102 - VOTO"]

        if variable == 'all':  # All municipalities and all variables
            name_var = ["casos-diarios", "fallecidos", "incidencia14", "incidencia7"]

            fpath = np.repeat(rawdir, len(name_mun), axis=0)
            for mun in tqdm(name_mun):
                nmun = mun.split(" - ")[1].replace(" ", "_")
                muni = np.repeat(mun, len(name_var), axis=0)
                covid_mun = list(map(download_json, muni, name_var, fpath))
                covid_mun = pd.concat(covid_mun, axis=1)
                covid_mun.to_csv(f'{rawdir}/all_data_{nmun}.csv', index=True, header=True)
                covid_mun.to_csv(f'{processdir}/all_data_{nmun}.csv', index=True, header=True)
                files.append(covid_mun)
            return files.append
        else:  # All municipalities and 1 variable
            fpath = np.repeat(rawdir, len(name_mun), axis=0)
            var = np.repeat(variable, len(name_mun), axis=0)
            covid_mun = list(map(download_json, tqdm(name_mun), var, fpath))
            return covid_mun
    else:
        if variable == 'all':  # 1 mun and all variables
            name_var = ["casos-diarios", "fallecidos", "incidencia14", "incidencia7"]

            fpath = np.repeat(rawdir, len(name_var), axis=0)
            mun = np.repeat(exp, len(name_var), axis=0)
            covid_mun = list(map(download_json, tqdm(mun), name_var, fpath))
            covid_mun = pd.concat(covid_mun, axis=1)
            nmun = exp.split(" - ")[1].replace(" ", "_")
            covid_mun.to_csv(f'{rawdir}/all_data_{nmun}.csv', index=True, header=True)
            covid_mun.to_csv(f'{processdir}/all_data_{nmun}.csv', index=True, header=True)
            return covid_mun

        else:  # 1 mun and 1 variable
            var_mun = download_json(exp, variable, rawdir)
            return var_mun


if __name__ == '__main__':
    download_ine()
    download_sc()
    download_icane(exp="region",
                   variable='all')
    download_icane(exp="mpio",
                   variable='all')
