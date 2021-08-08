import click
import pandas as pd
from process import fix_1207, process_mobility

from download import download_ine, download_sc, download_icane


@click.command()
@click.option('--files', default='mitma', help="Download mobility and COVID-19 data. Options: all, mitma, covid.")
@click.option('--update', '-u', is_flag=True, help="Update current files without overwriting.")
def data(files, update):
    if update:
        print('Updating the existing files')
    else:
        print('Generating data from scratch')

    if files == 'mitma':
        files_ine = download_ine(exp='maestra1',
                                 res='municipios',
                                 update=update,
                                 force=False)
        fix_1207()
        process_mobility(day_files='all',
                         exp='maestra1',
                         res='municipios',
                         update=update,
                         force=False)

    elif files == 'covid':
        files_covid_hist = download_sc(subject='historico')
        files_covid_cant = download_icane(exp="region",
                                          variable='all')
        files_covid_mun = download_icane(exp="mpio",
                                         variable='all')

    else:
        files_ine = download_ine(exp='maestra1',
                                 res='municipios',
                                 update=update,
                                 force=False)
        fix_1207()
        process_mobility(day_files='all',
                         exp='maestra1',
                         res='municipios',
                         update=update,
                         force=False)
        files_covid_hist = download_sc(subject='historico')
        files_covid_cant = download_icane(exp="region",
                                          variable='all')
        files_covid_mun = download_icane(exp="mpio",
                                         variable='all')


if __name__ == '__main__':
    data()
