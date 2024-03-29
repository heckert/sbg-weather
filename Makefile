# adapted from
# https://github.com/drivendata/cookiecutter-data-science/blob/master/%7B%7B%20cookiecutter.repo_name%20%7D%7D/Makefile
PROJECT_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# extract project name from full project directory
PROJECT_NAME=$(shell basename $(PROJECT_DIR))
# https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

environment:
# create conda env from environemt.yml file
# environment will be named same as the project folder
	@echo "Creating conda environment for $(PROJECT_NAME)..."
	@conda env create -n $(PROJECT_NAME) -f environment.yml

sbg_data:
# metadata: https://www.data.gv.at/katalog/dataset/meteorologische-daten-des-salzburger-luftgutemessnetzes-jahresdateien
	@echo "Loading raw sbg weather data..."
	@for year in {2011..2019} ; do \
	wget -O $(PROJECT_DIR)/data/raw/meteo-$$year.zip https://www.salzburg.gv.at/ogd/805314f6-5b72-4c2a-aa22-e1ece6d85db5/meteo-$$year.zip ; \
	unzip -o $(PROJECT_DIR)/data/raw/meteo-$$year.zip -d $(PROJECT_DIR)/data/raw/ ; \
	done

	@rm $(PROJECT_DIR)/data/raw/*.zip
	@echo "Successfully loaded and unzipped SBG weather data"

# data for 2011-2013 is split into two csvs, 
# e.g. meteo-2011-1.csv & meteo-2011-2.csv
	@echo "Concatenating files for 2011-13"
	@for year in {2011..2013} ; do \
	touch $(PROJECT_DIR)/data/raw/meteo-$$year.csv; \
	for part in 1 2 ; do \
	cat $(PROJECT_DIR)/data/raw/meteo-$$year-$$part.csv >> $(PROJECT_DIR)/data/raw/meteo-$$year.csv ; \
	rm $(PROJECT_DIR)/data/raw/meteo-$$year-$$part.csv ; \
	done; done

	@echo "Renaming meteorologie-2015.csv to meteo-2015.csv"
	@mv $(PROJECT_DIR)/data/raw/meteorologie-2015.csv $(PROJECT_DIR)/data/raw/meteo-2015.csv


pred_maint_data:
# metadata: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
	@echo "Loading raw predicitve maintenance data..."
	@wget -O $(PROJECT_DIR)/data/raw/predictive_maintenance_dataset.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv