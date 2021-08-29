# Salzburg Weather Data

> **NOTE:**  _Currently still work in progress_

Exploring various time-series deep learning methods on weather data for the province of Salzburg, Austria using TensorFlow and Keras.

Data is taken from Austria's [open government data platform](https://www.data.gv.at/).

Data preparation and feature engineering leans – quite heavily at times – on TensorFlow's 
[structured time series data tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series).

## How to start developing
### 1. Create conda environment  
This project assumes that a conda installation is available.  

    make environment

 will create a conda environment with all required packages and by default name it after the project directory (_sbg-weather_).

### 2. Load the data
    make sbg_data
    
loads datasets for 2011-2019 from the open government data website and stores it in the `data/raw` directory.

### 3. Data preparation and feature engineering
This is currently handled in the file `01-explore&prep.ipynb` in the notebooks directory. The raw data is processed and eventually stored across several csv files.
The datasets are written to seperate `train`, `val` and `test` folders within `data/processed`.

### 4. Model training
A first example of training an LSTM model is provided in the notebook `02-train.ipynb`.  
Input is a multivariate time-series containing hourly measurements of air pressure, temperature, humidity, as well as wind velocity and direction.  
These metrics are recorded across several locations throughout Salzburg.
The location is provided to the model via an embedding layer.  
72-hour-windows of input data are then used to predict temperature for the upcoming 24 hours.

#### Current model architecture
![Model architecture](plots/forecast_model_architecture.png)

#### Model output  
![Results](plots/forecast_output.png)


## Sources
* TensorFlow's 
[structured time series data tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
* The highly recommended book [Hands On Machine Learning With Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
