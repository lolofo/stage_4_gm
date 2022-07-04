import wget
import zipfile
from zipfile import ZipFile
import os
from os import path
import pandas as pd

def download_snli_raw(cache_path):
    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    response = wget.download(url, "snli_data.zip")

    print("\t>> start unzip the files")

    # first clean the zip files

    original_zip = ZipFile('snli_data.zip', 'r')
    new_zip = ZipFile('clean_snli_data.zip', 'w')

    for item in original_zip.infolist():
        buffer = original_zip.read(item.filename)
        if not (str(item.filename).startswith("__MACOSX/")):
            if str(item.filename).endswith('.txt'):
                if not (str(item.filename).endswith('README.txt')):
                    new_zip.writestr(item, buffer)

    new_zip.close()
    original_zip.close()

    try:
        with zipfile.ZipFile('clean_snli_data.zip', 'r') as zip_ref:
            zip_ref.extractall(path.join(cache_path))
        os.remove("snli_data.zip")
        os.remove("clean_snli_data.zip")
        print("\t>> Finished for the snli part!")
    except FileNotFoundError:
        print("\t>> verify that all the files are present")

def download_e_snli_raw(cache_path):
    if not path.exists(cache_path):
        os.mkdir(cache_path)

    urls = [r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv",
            r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_2.csv",
            r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv",
            r"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_dev.csv"]

    for url in urls:
        nm = url.split("/")[-1]
        df = pd.read_csv(url)
        df.to_csv(path.join(cache_path, nm))

if __name__ == "__main__":
    download_snli_raw(os.path.join(os.getcwd(), ".cache", "raw_data", "e_snli"))
    download_e_snli_raw(os.path.join(os.getcwd(), ".cache", "raw_data", "e_snli"))
