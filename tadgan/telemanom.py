import numpy as np
import pandas as pd
import csv
from .tadgan import ProcessedDataset


def process(src_file, dist_file, window_size=100, step_size=1, train_range=float('inf')):
    values = np.load(src_file)[:, 0]
    min_value = np.min(values)
    max_value = np.max(values)
    train_range = min(train_range, len(values))

    def normalize(v):
        v = v - 0.5 * (max_value + min_value)
        if max_value - min_value > 0:
            v /= 0.5 * (max_value - min_value)
        return v

    with open(dist_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(np.append(['i'], [
                        'i_{}'.format(i) for i in range(window_size)]))

        start = 0
        end = window_size

        while end <= train_range:
            writer.writerow(np.append([start], [
                            normalize(v) for v in values[start:end]]))
            start += step_size
            end += step_size


class TelemanomDataset(ProcessedDataset):
    def __init__(self, processed_csv):
        self.df = pd.read_csv(processed_csv)

    def __getitem__(self, index):
        return (self.df.iloc[index][0], np.array([self.df.iloc[index][1:].values]))

    def __len__(self):
        return len(self.df)

    @property
    def window_size(self):
        col = len(self.df.columns)
        return max(col - 1, 0)

    @property
    def state_size(self):
        return 1
