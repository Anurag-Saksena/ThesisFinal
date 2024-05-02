import statistics
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import finplot as fplt
from pandas import Timestamp, DataFrame
import os
import cv2

def label_function(val):
    if val < -0.2:
        return 0
    elif val <-0.1:
        return 1
    elif val < 0:
        return 2
    elif val < 0.1:
        return 3
    elif val < 0.2:
        return 4
    else:
        return 5


def scale_values(df):
    for column in ["open", "high", "low", "close"]:
        if column == "date":
            continue
        df[column] = df[column] / statistics.mean(df[column])

df = pd.read_feather("data/NIFTY 50_data.feather")

df["percent_change"] = (df["close"].pct_change()).apply(lambda x: round(x * 100, 2))
df["percent_change_after"] = df["percent_change"].shift(-1)
print(df)
df["date"] = df["date"].apply(lambda x: Timestamp(str(x)))

df = df.dropna().reset_index(drop=True)

print(df)

bins = [round(val, 2) for val in np.arange(-15, 15.1, 0.1)]

counts, edges, bars = plt.hist(df["percent_change"], bins=bins)

plt.bar_label(bars)

data_labels = []

plt.show()

rolling_window_size = 6

for i in range(2):
    start_time = perf_counter()
    rolling_window = df.iloc[i:i+rolling_window_size].reset_index(drop=True)
    print(rolling_window)

    # scale_values(rolling_window)
    #
    # print(rolling_window)



    plt.figure()

    # "up" dataframe will store the stock_prices
    # when the closing stock price is greater
    # than or equal to the opening stock prices
    up = rolling_window[rolling_window.close >= rolling_window.open]

    # "down" dataframe will store the stock_prices
    # when the closing stock price is
    # lesser than the opening stock prices
    down = rolling_window[rolling_window.close < rolling_window.open]

    # When the stock prices have decreased, then it
    # will be represented by blue color candlestick
    col1 = 'green'

    # When the stock prices have increased, then it
    # will be represented by green color candlestick
    col2 = 'pink'

    # Setting width of candlestick elements
    width = .3
    width2 = .03

    # Plotting up prices of the stock
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

    # Plotting down prices of the stock
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)

    # rotating the x-axis tick labels at 30degree
    # towards right
    plt.xticks(rotation=30, ha='right')


    plt.savefig(f"data/images/image_{i}.png")

    plt.clf()
    plt.close()

    # displaying candlestick chart of stock data
    # of a week
    # plt.show()

    img = cv2.imread(f"data/images/image_{i}.png")

    print(img.shape)

    cropped_img = img[60:420, 80:560] # 360x480

    resized_img = cv2.resize(cropped_img, (30, 40), interpolation=cv2.INTER_AREA)

    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"data/images/image_{i}_cropped.png", cropped_img)
    cv2.imwrite(f"data/images/image_{i}_resized.png", resized_img)
    cv2.imwrite(f"data/images/image_{i}_grayscale.png", grayscale_img)

    percent_change_after = rolling_window["percent_change_after"][rolling_window_size-1]

    data_label = label_function(percent_change_after)

    data_labels.append(data_label)

    print("screenshot taken")
    print(f"image {i} saved")
    print(data_label)
    print(f"{round((perf_counter() - start_time) * 1000, 2)} ms")


print(data_labels)

# df = DataFrame()
# df["data_labels"] = data_labels
#
# df.to_csv("data/data_labels.csv", index=False)

from time import perf_counter

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import finplot as fplt
# from pandas import Timestamp, DataFrame
# import os
# import cv2
#
# def label_function(val):
#     if val < -0.2:
#         return 0
#     elif val <-0.1:
#         return 1
#     elif val < 0:
#         return 2
#     elif val < 0.1:
#         return 3
#     elif val < 0.2:
#         return 4
#     else:
#         return 5
#
# df = pd.read_feather("data/NIFTY 50_data.feather")
#
# df["percent_change"] = (df["close"].pct_change()).apply(lambda x: round(x * 100, 2))
# df["percent_change_after"] = df["percent_change"].shift(-1)
# print(df)
# df["date"] = df["date"].apply(lambda x: Timestamp(str(x)))
#
# df = df.dropna().reset_index(drop=True)
#
# print(df)
#
# bins = [round(val, 2) for val in np.arange(-15, 15.1, 0.1)]
#
# counts, edges, bars = plt.hist(df["percent_change"], bins=bins)
#
# plt.bar_label(bars)
#
# data_labels = []
#
# plt.show()
#
# df.to_csv("data/final_data.csv", index=False)
#
# df["range"] = df["high"] - df["low"]
# print(list(df["range"]).index(609))
#
# # rolling_window_size = 6
# #
# # for i in range(len(df)-rolling_window_size+1):
# #     start_time = perf_counter()
# #     rolling_window = df.iloc[i:i+rolling_window_size].reset_index(drop=True)
# #     # print(rolling_window)
# #
# #     percent_change_after = rolling_window["percent_change_after"][rolling_window_size-1]
# #
# #     data_label = label_function(percent_change_after)
# #
# #     data_labels.append(data_label)
# #     # print(data_label)
# #     print(f"image {i}")
# #     print(f"{round((perf_counter() - start_time) * 1000, 2)} ms")
# #
# #
# # print(data_labels)
# #
# # df = DataFrame()
# # df["data_labels"] = data_labels
# #
# # df.to_csv("data/data_labels.csv", index=False)
