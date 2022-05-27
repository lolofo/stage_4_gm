import wget
import zipfile
from zipfile import ZipFile
import os
from os import path

# downloading the files
print("start downloading")

url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
response = wget.download(url, "snli_data.zip")
print()

cache = path.join(os.getcwd(), '.cache')

print("start unzip the files")

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
        zip_ref.extractall(path.join(cache, 'raw_data', 'snli_data'))
    os.remove("snli_data.zip")
    os.remove("clean_snli_data.zip")
    print("Finished !")
except FileNotFoundError:
    print("verify that all the files are present")




