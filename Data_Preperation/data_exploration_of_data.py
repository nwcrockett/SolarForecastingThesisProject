import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import joypy
import os
import seaborn as sns
import numpy as np


def ridgeline_plot_of_minute_data(file_path):
    direct = os.listdir(file_path)
    direct.sort()
    df = []

    for item in direct:
        temp_df = pd.read_csv(file_path + "/" + item + "/minute_data_total_year.csv")
        temp_df["year"] = item
        df.append(temp_df)

    df = pd.concat(df, sort=False)
    fig, axes = joypy.joyplot(df, by="year", column="downwelling_shortwave", linewidth=0.05, colormap=cm.summer_r,
                                title="Ridgeline plot of GHI minute",
                                grid=True)
    plt.savefig(
        "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/"
        "Data_Preperation/graphs/Ridgeline plot minute.png")


def ridgeline_plot_of_hourly_data(file_path):
    direct = os.listdir(file_path)
    direct.sort()
    df = []
    split_for_ram = 1
    first_half = direct[:len(direct) // 2]
    second_half = direct[(len(direct) // 2):]
    final = second_half[(len(second_half) // 2):]
    second_half =second_half[:(len(second_half) // 2)]

    for item in first_half:
        find_file = os.listdir(file_path + "/" + item)
        find_file = pd.Series(find_file)
        file = find_file[find_file.str.startswith("nsaarmbecldradC1")]
        file = file[file.str.contains("nc")]
        file = file.reset_index(drop=True)
        print(file[0])
        temp_df = pd.read_csv(file_path + "/" + item + "/" + file[0])
        temp_df["year"] = item
        df.append(temp_df)
    df = pd.concat(df, sort=False)
    fig, axes = joypy.joyplot(df, by="year", column="swdn", linewidth=0.05, colormap=cm.summer_r,
                                title="Ridgeline plot of GHI hourly", grid=True)
    plt.savefig(
        "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/"
        "Data_Preperation/graphs/Ridgeline plot hourly" + str(split_for_ram) + ".png")
    split_for_ram += 1
    df = []

    for item in second_half:
        find_file = os.listdir(file_path + "/" + item)
        find_file = pd.Series(find_file)
        file = find_file[find_file.str.startswith("nsaarmbecldradC1")]
        file = file[file.str.contains("nc")]
        file = file.reset_index(drop=True)
        print(file[0])
        temp_df = pd.read_csv(file_path + "/" + item + "/" + file[0])
        temp_df["year"] = item
        df.append(temp_df)
    df = pd.concat(df, sort=False)
    fig, axes = joypy.joyplot(df, by="year", column="swdn", linewidth=0.05, colormap=cm.summer_r,
                                title="Ridgeline plot of GHI hourly", grid=True)
    plt.savefig(
        "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/"
        "Data_Preperation/graphs/Ridgeline plot hourly" + str(split_for_ram) + ".png")

    split_for_ram += 1
    df = []

    for item in final:
        find_file = os.listdir(file_path + "/" + item)
        find_file = pd.Series(find_file)
        file = find_file[find_file.str.startswith("nsaarmbecldradC1")]
        file = file[file.str.contains("nc")]
        file = file.reset_index(drop=True)
        print(file[0])
        temp_df = pd.read_csv(file_path + "/" + item + "/" + file[0])
        temp_df["year"] = item
        df.append(temp_df)
    df = pd.concat(df, sort=False)
    fig, axes = joypy.joyplot(df, by="year", column="swdn", linewidth=0.05, colormap=cm.summer_r,
                              title="Ridgeline plot of GHI hourly", grid=True)
    plt.savefig(
        "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/"
        "Data_Preperation/graphs/Ridgeline plot hourly" + str(split_for_ram) + ".png")


def make_correlation_matrix(df, photo, year):
    corr = df.corr()  # Use pandas to make the correlation calculations
    mask = np.zeros_like(corr, dtype=np.bool)  # mask to make the graph a triangle
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(year)
    sns.heatmap(corr, mask=mask, center=0, cmap="PiYG",
                square=True, linewidths=.5)
    plt.savefig(photo)
    plt.show()


if __name__ == "__main__":

    file_path = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/train"
    # ridgeline_plot_of_hourly_data(file_path)
    # ridgeline_plot_of_minute_data(file_path)


