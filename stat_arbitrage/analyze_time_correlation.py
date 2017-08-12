import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from download_bars import PAIRS


def get_pair_df(pair):
    df = pd.read_csv(open('datasets/{}.csv'.format(pair)), index_col='date').rename(columns={'close': pair})
    df.drop('open', 1, inplace=True)
    df.drop('high', 1, inplace=True)
    df.drop('low', 1, inplace=True)
    df = (df - df.mean()) / df.std()
    return df


for n in range(12):
    dfs = []
    for p in PAIRS:
        df = get_pair_df(p)
        dfs.append(df)
        for pair in PAIRS:
            if pair != p:
                dfs.append(df.shift(n))

        result = pd.concat(dfs, axis=1)
        corr = result.corr()

        corr = corr.abs().unstack()
        best_corr = corr[corr != 1.0].sort_values().tail(1).to_dict()
        if best_corr:
            keys = list(best_corr.keys())[0]
            if p in keys and keys[0] != keys[1]:
                print('{} shift from {} | Top correlations: '.format(n, p))
                print(best_corr, keys)