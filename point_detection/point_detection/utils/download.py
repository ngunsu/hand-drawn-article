import pathlib
import os
import gdown


def create_if_not_exists(folder):
    """ Creates folder if doesn't exists

    Parameters
    ----------
    folder(str):
        Folder to check | create

    Returns
    -------
    None
    """
    path = pathlib.Path(folder)
    if not path.exists():
        os.system(f'mkdir -p {path}')


def download_from_gdrive(target_folder: str, url: str, filename: str, decompress: bool = False) -> None:
    """
    Download file from gdrive

    Parameters
    ----------
    target_folder : str
        Target folder
    url: str
        Gdrive link
    filename: str
        Filename of the url link
    decompress: bool
        If True the file is compressed
    """
    create_if_not_exists(target_folder)
    # Download file

    path = os.path.join(target_folder, filename)
    gdown.download(url, output=path, quiet=False, fuzzy=True)

    if decompress:
        if '.tar' in filename:
            os.system(f'tar xvf {path} -C {target_folder}')
            os.system(f'rm {path}')
        else:
            exit('unknown format')
