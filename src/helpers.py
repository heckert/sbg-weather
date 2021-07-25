import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def get_filenames_in_directory(path):
    f = []
    for (_, _, filenames) in os.walk(path):
        f.extend(filenames)
        break

    return f 

class WindowGenerator:
    '''
    Adapted from https://www.tensorflow.org/tutorials/structured_data/time_series
    '''

    def __init__(self, input_width=24, label_width=6, shift=6,
                 train_dir=None, val_dir=None, test_dir=None,
                 label_columns=['Lufttemperatur [GradC]'],
                 embedding_column='Location'):

        # Read columns from first file in train_dir.
        self.train_files = get_filenames_in_directory(train_dir)

        with open(train_dir/self.train_files[0]) as f:
            self.columns = f.readline() \
                            .strip() \
                            .split(',')

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                enumerate(self.columns)}

        self.embedding_column = embedding_column
        if embedding_column is not None:
            self.embedding_index = self.columns.index(embedding_column)

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
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

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        # inputs.set_shape([self.input_width, None])
        # labels.set_shape([self.label_width, None])

        return inputs, labels


    def csv_reader(self, dir_):
        pattern = self._path2pattern(dir_)
        files = tf.data.Dataset.list_files(pattern, shuffle=False)

        dataset = files.flat_map(lambda file: 
            tf.data.TextLineDataset(file).skip(1) \
                .map(self._preprocess) \
                .window(self.total_window_size, shift=1, drop_remainder=True) \
                .flat_map(self._create_window) \
                .map(self._split_xy)

        # TODO:
        # Batching
        # Prefetching
                
        )

        return dataset

  