# What this is
The code repository of Deep Learning-Assisted De Novo Molecular Identification through Vibrational Spectra

# How to run
1. Prepare the data using QCForever(https://github.com/molecule-generator-collection/QCforever). The molecules that exctracted from GDB are listed in SMILES.txt.
2. Edit data_directory/create_data_scripts/constants.py and scripts/constants.py
3. run pipenv install to creat the environment
4. run chain.sh to crate the dataset
5. run pipenv python scripts/train.py

Youcan check the result in result_directory you specify in the constants.py
