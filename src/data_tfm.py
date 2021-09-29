import click
from process import fix_1207, process_mobility, process_sc, process_icane
from download import download_ine, download_sc, download_icane


@click.command()
@click.option('--files', default='covid', help="Download mobility and COVID-19 data. Options: all, mitma, covid.")
@click.option('--update', '-u', is_flag=True, help="Update current files without overwriting.")
def data(files, update):
    if update:
        print('Updating the existing files')
    else:
        print('Generating data from scratch')

    if files == 'mitma':
        download_ine(exp='maestra1',
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

        download_sc(subject='historico')
        process_sc()
        download_icane(exp="region",
                       variable='all')
        process_icane()
        download_icane(exp="mpio", variable='all')

    else:  # All
        download_ine(exp='maestra1',
                     res='municipios',
                     update=update,
                     force=False)
        fix_1207()
        process_mobility(day_files='all',
                         exp='maestra1',
                         res='municipios',
                         update=update,
                         force=False)
        download_sc(subject='historico')
        process_sc()
        download_icane(exp="region",
                       variable='all')
        process_icane()
        download_icane(exp="mpio", variable='all')


if __name__ == '__main__':
    data()
