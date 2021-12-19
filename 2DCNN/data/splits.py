#! /usr/bin/env python3

"""
    code to create train data split
"""

import pandas

if __name__ == "__main__":
    df = pandas.read_csv("data/train.csv")
    for n in [1000, 2500, 5000]:
        df_ = df.sample(n, random_state=0)
        df_.to_csv(f"data/train_{n}.csv", index=False)
