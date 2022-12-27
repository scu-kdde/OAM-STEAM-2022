import pandas as pd


def save_best_result(file, auc, info):
    df = pd.read_csv(file)
    old_auc = float(df["auc"][df.shape[0] - 1])
    if auc >= old_auc:
        line = pd.Series(info)
        df = df.append(line, ignore_index=True)
        print(df)
        df.to_csv(file, index=False, sep=',')


info = {"auc": 0.94, "discriminator_out": 212, "discriminator_learning_rate": 0.6}
save_best_result('../logs/ACM.csv', 0.94, info)
