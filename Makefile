# adapted from
# https://github.com/drivendata/cookiecutter-data-science/blob/master/%7B%7B%20cookiecutter.repo_name%20%7D%7D/Makefile
PROJECT_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# extract project name from full project directory
PROJECT_NAME=$(shell basename $(PROJECT_DIR))
# https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

create_env:
# create conda env from environemt.yml file
# environment will be named same as the project folder
	@echo "Creating conda environment for $(PROJECT_NAME)..."
	@conda env create -n $(PROJECT_NAME) -f environment.yml
# install src package in development mode
# so that `import from src.<module>` will work
	@echo "Installing src package..."
	@($(CONDA_ACTIVATE) $(PROJECT_NAME); pip install -e .)

get_sbg_data:
# metadata: https://www.data.gv.at/katalog/dataset/meteorologische-daten-des-salzburger-luftgutemessnetzes-jahresdateien
	@echo "Loading raw sbg weather data..."
	@for year in 2016 2017 2018 2019 ; do \
	wget -O $(PROJECT_DIR)/data/raw/meteo-$$year.zip https://www.salzburg.gv.at/ogd/805314f6-5b72-4c2a-aa22-e1ece6d85db5/meteo-$$year.zip ; \
	unzip -o $(PROJECT_DIR)/data/raw/meteo-$$year.zip -d $(PROJECT_DIR)/data/raw/ ; \
	done

	@rm $(PROJECT_DIR)/data/raw/*.zip
	@echo "Successfully loaded and unzipped SBG weather data"


get_pred_maint_data:
# metadata: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
	@echo "Loading raw predicitve maintenance data..."
	@wget -O $(PROJECT_DIR)/data/raw/predictive_maintenance_dataset.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv