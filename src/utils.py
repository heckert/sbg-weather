import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import tensorflow as tf
from IPython.display import Image

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import metrics

def get_filenames_in_directory(path):
    f = []
    for (_, _, filenames) in os.walk(path):
        f.extend(filenames)
        break

    return f 

class SplitScaler:
    """Helper class to handle train-test splitting & location-wise scaling

    Data is split and scaled upon initialization and can then be accessed
    via the attributes `train`, `val` and `test`.

    Parameters
    ---------
    df : pandas.DataFrame
        Multiindexed DataFrame in which the 0th index level
        represents the location at which the data are measured,
        and the 1st level represents the time of measurement.

    train_val_ratios : tuple
        The ratios for training and validation dataset.
        Ratio for the test dataset is calculated as
        1 - (train ratio + validation ratio)

    """
    
    def __init__(self, df: pd.DataFrame, train_val_ratios: tuple):
        self.df = df
        self.train_share, self.val_share = train_val_ratios
        self.unique_locations = df.index.get_level_values(0).unique()
        self._split_data()
        self._scale()

    def _split_data(self):
        # create slices for training, validation & testing
        self.min_date = self.df.index.get_level_values(1).min().strftime('%Y-%m-%d')
        self.max_date = self.df.index.get_level_values(1).max().strftime('%Y-%m-%d')

        d_range = pd.date_range(self.min_date, self.max_date, freq='D')
        # days need to be string rather than datetime for 
        # dataframe sclicing to work as expected
        d_range = d_range.map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
        n_dates = len(d_range)

        train_range = d_range[:int(n_dates*self.train_share)]
        val_range = d_range[int(n_dates*self.train_share):int(n_dates*(self.train_share+self.val_share))]
        test_range = d_range[int(n_dates*(self.train_share+self.val_share)):]

        self.train_slice = slice(train_range[0], train_range[-1])
        self.val_slice = slice(val_range[0], val_range[-1])
        self.test_slice = slice(test_range[0], test_range[-1])

        self.train_unsc = self.df.loc[pd.IndexSlice[:,self.train_slice],:].copy()
        self.val_unsc = self.df.loc[pd.IndexSlice[:,self.val_slice],:].copy()
        self.test_unsc = self.df.loc[pd.IndexSlice[:,self.test_slice],:].copy()

        #self.split_data_unsc = train_unsc, val_unsc, test_unsc

    def plot(self):
        fig, ax = plt.subplots(figsize=(10,2))
        d_range = pd.date_range('2011-01-01', '2019-12-31')

        data = self.df.loc['St.Johann - Bezirkshauptmannschaft']

        data.loc[self.train_slice, 'Lufttemperatur [GradC]'].plot(ax=ax, color='tab:blue', label='Train')
        data.loc[self.val_slice, 'Lufttemperatur [GradC]'].plot(ax=ax, color='tab:orange', label='Validation')
        data.loc[self.test_slice, 'Lufttemperatur [GradC]'].plot(ax=ax, color='tab:green', label='Test')
        ax.legend()
        ax.set_title('Train, Validation & Test Set')
        plt.show()

    def _scale(self):
        self.scalers = {}
        ds_types = {
            'train': [],
            'val': [],
            'test': []
        }

        # ds_types = ('train', 'val', 'test')
        # containers = ([], [], [])  # empty arrays to store scaled dfs per location

        # for ds_type, df_unsc, container in zip(ds_types, self.split_data_unsc, containers):
        for ds_type, container in ds_types.items():
            for location in self.unique_locations:
                # get data per location 
                df_unsc = self.__getattribute__(f'{ds_type}_unsc')
                df_loc = df_unsc.loc[location]
                times = df_loc.index

                # fit scaler only on training data
                if ds_type == 'train':
                    scaler = StandardScaler()
                    scaler = scaler.fit(df_loc)
                    # Keep scaler per location
                    self.scalers[location] = scaler
                else:
                    scaler = self.scalers[location]
                    
                scaled_data = scaler.transform(df_loc)

                # create MultiIndex from location and array of measurement times
                index = pd.MultiIndex.from_product([[location], times], names=['Messort', 'Zeitpunkt'])
                df_loc_sc = pd.DataFrame(scaled_data, index=index, columns=df_loc.columns)

                container.append(df_loc_sc)

            concat_locations_df = pd.concat(container)
            self.__setattr__(ds_type, concat_locations_df)


class WindowGenerator:
    '''
    Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series
    '''

    def __init__(self, input_width=24, label_width=6, shift=None,
                 train_dir=None, val_dir=None, test_dir=None,
                 batch_size=32,
                 label_columns=['Lufttemperatur [GradC]'],
                 embedding_column='Location'):

        # store data dirs
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        # read columns from first file in train_dir
        self.train_files = get_filenames_in_directory(train_dir)

        with open(train_dir/self.train_files[0]) as f:
            self.columns = f.readline() \
                            .strip() \
                            .split(',')

        # work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                enumerate(self.columns)}

        # work out embedding column index, if available
        self.embedding_column = embedding_column
        if embedding_column is not None:
            self.embedding_index = self.columns.index(embedding_column)
        else:
            self.embedding_index = None

        self.batch_size = batch_size

        # work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        if shift is None:
            shift = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def _path2pattern(self, path):
        return str(path) + '/*.csv'

    def _preprocess(self, line):
        defs = [0.] * len(self.columns)
        fields = tf.io.decode_csv(line, record_defaults=defs)
        line = tf.stack(fields)
        return line

    def _create_window(self, window):
        return window.batch(self.total_window_size)

    def _split_xy(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        if self.embedding_index is not None:
            embeddings = inputs[:,self.embedding_index]
            inputs = inputs[:, :self.embedding_index]

            return (inputs, embeddings), labels

        return inputs, labels

    def make_dataset(self, dir_):
        pattern = self._path2pattern(dir_)
        files = tf.data.Dataset.list_files(pattern, shuffle=False)

        # Interleave windows generated from different files.
        dataset = files.interleave(lambda file: \
            tf.data.TextLineDataset(file).skip(1) \
                .map(self._preprocess) \
                .window(self.total_window_size, shift=3, drop_remainder=True) \
                .flat_map(self._create_window) \
                .map(self._split_xy) \
                .shuffle(200) \
                .batch(self.batch_size) \
                .prefetch(tf.data.AUTOTUNE),
            num_parallel_calls=len(self.train_files))

        return dataset

    def plot(self, model=None, max_subplots=3, rotate_examples=False):
        plot_col = self.label_columns[0]

        if rotate_examples:
            example = self.rotating_example
        else:
            example = self.example

        if self.embedding_column is not None:
            (inputs, embeddings), labels = example
        else:
            inputs, labels = example

        plt.figure(figsize=(9, 6))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(plot_col.split(' ')[0]+'\n[normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(self.label_indices, labels[n, :, label_col_index],
                     label='Labels', c='#2ca02c')
            if model is not None:
                if self.embedding_column is not None:
                    predictions = model.predict((inputs, embeddings))
                else:
                    predictions = model.predict(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Zeit [h]')

    @property
    def train(self):
        return self.make_dataset(self.train_dir)

    @property
    def val(self):
        return self.make_dataset(self.val_dir)

    @property
    def test(self):
        return self.make_dataset(self.test_dir)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.test` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        return result

    @property
    def rotating_example(self):
        result = next(iter(self.test))
        return result


class ModelTrainer:
    '''
    Helper class for model compilation, visualization, 
    training, evaluation and storing. 
    '''
    def __init__(
        self,
        model,
        name=None,
        train_dataset=None,
        validation_dataset=None,
        test_dataset=None, 
        batch_size=32,
        lr=1e-4, 
        patience=3,
        metrics=None,
        saved_model_dir=None,
        vis_dir=None,
        verbose=True
    ):

        self.model = model
        self.model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(lr=lr),
            metrics=metrics
        )
        self.name = name
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        # self.lr = lr
        # self.max_epochs = max_epochs
        self.patience = patience
        self.metrics = metrics
        self.saved_model_dir = saved_model_dir
        self.vis_dir = vis_dir
        self.verbose = verbose

        self.today = datetime.date.today().strftime('%Y-%m-%d')
        self.name_to_save = self.name.lower()\
                                     .replace(' ', '_')\
                                     .replace('/', '') + '_' + self.today

    def plot_model(self):
        if self.vis_dir is not None:
            to_file = self.vis_dir / f'{self.name_to_save}.png'

            tf.keras.utils.plot_model(
                self.model, to_file=to_file, show_shapes=True, 
                show_layer_names=False, rankdir='TB', expand_nested=False, dpi=64
            )

            return Image(to_file)

        else:
            raise ValueError('Parameter `vis_dir` not defined') 

    def fit(self, max_epochs=30):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.patience,
            mode='min', restore_best_weights=True
        )

        self.history = self.model.fit(
            self.train_dataset, epochs=max_epochs,
            validation_data=self.validation_dataset,
            callbacks=[early_stopping], batch_size=self.batch_size,
            verbose=self.verbose
        )

        if self.saved_model_dir is not None:
            self.model.save(self.saved_model_dir / self.name_to_save)
            print('Saved trained model at {}'.format(
                    self.saved_model_dir / self.name_to_save
                )
            )
    
    def plot_history(self):
        if self.history is not None:
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.legend(labels = ["loss", "validation loss"])
            plt.xlabel("Epochs")
            plt.ylabel("val_loss")
            plt.title(f'{self.name} (MSE)')

        else:
            raise ValueError('No history to plot')

    def evaluate(self):
        return self.model.evaluate(
            self.test_dataset, verbose=self.verbose
        )




    


