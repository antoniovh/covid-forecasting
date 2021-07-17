import click
from misc import fix_1207
# from process import process

from download import download_ine, download_sc, download_icane
# from flowmapblue import generate_flowmapblue


@click.command()
@click.option('--files', default='all', help="Download mobility and COVID-19 data. "
                                             "Options: all, mitma, covid.")
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
        # process(day_files=files_ine,
        #         exp='maestra1',
        #         res='municipios',
        #         update=update,
        #         force=False)
        # # generate_flowmapblue()

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
        files_covid_hist = download_sc(subject='historico')
        files_covid_cant = download_icane(exp="region",
                                          variable='all')
        files_covid_mun = download_icane(exp="mpio",
                                         variable='all')


if __name__ == '__main__':
    data()
