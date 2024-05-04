# import statistics
# import sys
# from time import perf_counter
#
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
#
# def scale_values(df):
#     for column in ["open", "high", "low", "close"]:
#         if column == "date":
#             continue
#         df[column] = df[column] / statistics.mean(df[column])
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
# rolling_window_size = 6
#
# max_span_list = []
# for i in range(len(df)-rolling_window_size+1):
#     max_span =  max(df["high"][i:i+rolling_window_size]) - min(df["low"][i:i+rolling_window_size])
#     max_span_list.append(max_span)
#
# print(sorted(max_span_list, reverse=True))
#
# max_span = max(max_span_list)
#
# max_span = (max_span// 10 + 1) * 10
#
# print(max_span)
# # import sys
# # sys.exit()
#
# for i in range(len(df)-rolling_window_size+1):
#     start_time = perf_counter()
#     rolling_window = df.iloc[i:i+rolling_window_size].reset_index(drop=True)
#     print(rolling_window)
#
#     # scale_values(rolling_window)
#     #
#     # print(rolling_window)
#
#
#
#     plt.figure(figsize=(6.4, 12.8)) # 12.8x6.4 inches i.e. 1280x640 pixels
#
#     # "up" dataframe will store the stock_prices
#     # when the closing stock price is greater
#     # than or equal to the opening stock prices
#     up = rolling_window[rolling_window.close >= rolling_window.open]
#
#     # "down" dataframe will store the stock_prices
#     # when the closing stock price is
#     # lesser than the opening stock prices
#     down = rolling_window[rolling_window.close < rolling_window.open]
#
#     # When the stock prices have decreased, then it
#     # will be represented by blue color candlestick
#     col1 = 'green'
#
#     # When the stock prices have increased, then it
#     # will be represented by green color candlestick
#     col2 = 'pink'
#
#     # Setting width of candlestick elements
#     width = .3
#     width2 = .03
#
#     # Plotting up prices of the stock
#     plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
#     plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
#     plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)
#
#     # Plotting down prices of the stock
#     plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
#     plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
#     plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)
#
#     rolling_window_mid_value = (min(rolling_window["low"]) + max(rolling_window["high"]))/2
#
#     lower_bound = rolling_window_mid_value - max_span/2
#     upper_bound = rolling_window_mid_value + max_span/2
#
#
#     plt.ylim(lower_bound, upper_bound)
#
#     # rotating the x-axis tick labels at 30degree
#     # towards right
#     plt.xticks(rotation=30, ha='right')
#
#
#     plt.savefig(f"data/images/image_{i}.png")
#
#     plt.clf()
#     plt.close()
#
#     # displaying candlestick chart of stock data
#     # of a week
#     # plt.show()
#
#     img = cv2.imread(f"data/images/image_{i}.png")
#
#     print(img.shape)
#
#
#
#     cropped_img = img[155:1140, 90:570] # 985x480
#     resized_img = cv2.resize(cropped_img, (40, 400), interpolation=cv2.INTER_AREA)
#     grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#
#     cv2.imwrite(f"data/images/image_{i}_cropped.png", cropped_img)
#     cv2.imwrite(f"data/images/image_{i}_resized.png", resized_img)
#     cv2.imwrite(f"data/images/image_{i}_grayscale.png", grayscale_img)
#
#
#     print("screenshot taken")
#     print(f"image {i} saved")
#     print(f"{round((perf_counter() - start_time) * 1000, 2)} ms")
#
#
# print(data_labels)
#
# # df = DataFrame()
# # df["data_labels"] = data_labels
# #
# # df.to_csv("data/data_labels.csv", index=False)
#
# from time import perf_counter
#
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
# print(sorted(list(df["high"]-df["low"]), reverse=True))
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
# rolling_window_size = 6
#
# for i in range(len(df)-rolling_window_size+1):
#     start_time = perf_counter()
#     rolling_window = df.iloc[i:i+rolling_window_size].reset_index(drop=True)
#     # print(rolling_window)
#
#     percent_change_after = rolling_window["percent_change_after"][rolling_window_size-1]
#
#     data_label = label_function(percent_change_after)
#
#     data_labels.append(data_label)
#     # print(data_label)
#     print(f"image {i}")
#     print(f"{round((perf_counter() - start_time) * 1000, 2)} ms")
#
#
# print(data_labels)
#
# df = DataFrame()
# df["data_labels"] = data_labels
#
# df.to_csv("data/data_labels.csv", index=False)
import shutil

from pandas import DataFrame

# import os
# import shutil
#
# from pandas import DataFrame
# import pandas as pd
#
# list_of_files = os.listdir("data/images/all")
# list_of_files = [file for file in list_of_files if file.endswith("grayscale.png")]
#
# len_list_of_files = len(list_of_files)
#
# # use train test split
# from sklearn.model_selection import train_test_split
# train_files, test_files = train_test_split(list_of_files, test_size=0.2, random_state=42)
#
# print(len(train_files))
# print(train_files)
# print(len(test_files))
# print(test_files)
#
# print(len(train_files))
# print(len(test_files))
#
# assert set(train_files).intersection(set(test_files)) == set()
#
# data_labels_df = pd.read_csv("data/data_labels.csv")
# data_labels = list(data_labels_df["data_labels"])
#
# train_data_labels = []
# test_data_labels = []
#
# train_files = sorted(train_files, key=lambda x: int(x.split("_")[1]))
# print(train_files)
#
# for file in train_files:
#     file_number = int(file.split("_")[1])
#     train_data_labels.append(data_labels[file_number])
#
# train_data_labels_df = DataFrame()
# train_data_labels_df["data_labels"] = train_data_labels
# train_data_labels_df.to_csv("data/train_data_labels.csv", index=False)
#
# test_files = sorted(test_files, key=lambda x: int(x.split("_")[1]))
# print(test_files)
#
# for file in test_files:
#     file_number = int(file.split("_")[1])
#     test_data_labels.append(data_labels[file_number])
#
# test_data_labels_df = DataFrame()
# test_data_labels_df["data_labels"] = test_data_labels
# test_data_labels_df.to_csv("data/test_data_labels.csv", index=False)

# import shutil
#
# for file in train_files:
#     shutil.copy(f"data/images/{file}", f"data/images/train/{file}")
#
# for file in test_files:
#     shutil.copy(f"data/images/{file}", f"data/images/test/{file}")

# import os
# import shutil
#
# from sklearn.model_selection import train_test_split
#
# current_dir = os.getcwd()
#
# os.chdir(r"D:\Pycharm Projects")
#
# train_images_list = os.listdir(r'ThesisFinalData\scaled_images_rolling_window_6\train')
# train_images_list.sort(key=lambda x: int(x.split('_')[1]))
#
# for file in train_images_list:



# from time import perf_counter
#
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
#
# def label_function_absolute(val):
#     if val < -30:
#         return 0
#     elif val <-20:
#         return 1
#     elif val < -10:
#         return 2
#     elif val < 0:
#         return 3
#     elif val < 10:
#         return 4
#     elif val < 20:
#         return 5
#     elif val < 30:
#         return 6
#     else:
#         return 7
#
# def label_function_absolute_binary(val):
#     if val < 0:
#         return 0
#     else:
#         return 1
#
# df = pd.read_feather("data/NIFTY 50_data.feather")
#
# df["open_to_close"] = df["close"] - df["open"]
# df["open_to_close_after"] = df["open_to_close"].shift(-1)
# print(df)
# df["date"] = df["date"].apply(lambda x: Timestamp(str(x)))
#
# df = df.dropna().reset_index(drop=True)
#
# print(df)
#
# bins = [round(val, 2) for val in np.arange(-100, 100, 10)]
#
# counts, edges, bars = plt.hist(df["open_to_close"], bins=bins)
#
# plt.bar_label(bars)
#
# data_labels = []
#
# plt.show()
#
# df.to_csv("data/final_data.csv", index=False)
# #
# # df["range"] = df["high"] - df["low"]
# # print(list(df["range"]).index(609))
#
# rolling_window_size = 6
#
# for i in range(len(df)-rolling_window_size+1):
#     start_time = perf_counter()
#     rolling_window = df.iloc[i:i+rolling_window_size].reset_index(drop=True)
#     # print(rolling_window)
#
#     open_to_close_after = rolling_window["open_to_close_after"][rolling_window_size-1]
#
#     data_label = label_function_absolute_binary(open_to_close_after)
#
#     data_labels.append(data_label)
#     print(data_label)
#     print(f"image {i}")
#     print(f"{round((perf_counter() - start_time) * 1000, 2)} ms")
#
#
# print(data_labels)
#
# df = DataFrame()
# df["data_labels"] = data_labels
#
# df.to_csv("data/data_labels_absolute_binary.csv", index=False)

import os
os.chdir(r"D:\Pycharm Projects")
import pandas as pd

data_labels_df = pd.read_csv(rf"ThesisFinal\Game Theoretical Adversarial Deep Learning\data\data_labels_absolute_binary.csv")

data_labels = list(data_labels_df["data_labels"])

train_files = os.listdir(rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\train\images")

train_data_labels = []
test_data_labels = []

train_files = sorted(train_files, key=lambda x: int(x.split("_")[1]))
print(train_files)

index = 0
for file in train_files:
    print(f"1__{index}")
    index += 1
    file_number = int(file.split("_")[1])
    train_data_labels.append(data_labels[file_number])

train_data_labels_df = DataFrame()
train_data_labels_df["data_labels"] = train_data_labels
train_data_labels_df.to_csv(rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\train\train_data_labels_absolute_binary.csv", index=False)

test_files = os.listdir(rf"ThesisFinalData\scaled_images_rolling_window_6\test\images")

test_files = sorted(test_files, key=lambda x: int(x.split("_")[1]))
print(test_files)

index = 0
for file in test_files:
    print(f"2__{index}")
    index += 1
    file_number = int(file.split("_")[1])
    test_data_labels.append(data_labels[file_number])

test_data_labels_df = DataFrame()
test_data_labels_df["data_labels"] = test_data_labels
test_data_labels_df.to_csv("ThesisFinalData/smaller_scaled_images_rolling_window_6/test/test_data_labels_absolute_binary.csv", index=False)


# import os
# os.chdir(r"D:\Pycharm Projects")
#
# index = 0
# for file in os.listdir(rf"ThesisFinalData\scaled_images_rolling_window_6\train"):
#     print(f"1__{index}")
#     index += 1
#     if file.endswith(".png.png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file[:-12]}")
#     if file.endswith(".png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file[:-8]}")
#     if file.endswith(".png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file[:-4]}")
#     if file.endswith(".png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file}")
#
# index = 0
# for file in os.listdir(rf"ThesisFinalData\scaled_images_rolling_window_6\train\images"):
#     print(f"2__{index}")
#     index += 1
#     if file.endswith(".png.png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file[:-12]}")
#     if file.endswith(".png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file[:-8]}")
#     if file.endswith(".png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\train\images\{file[:-4]}")
#
# index = 0
# for file in os.listdir(rf"ThesisFinalData\scaled_images_rolling_window_6\test"):
#     print(f"3__{index}")
#     index += 1
#     if file.endswith(".png.png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file[:-12]}")
#     if file.endswith(".png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file[:-8]}")
#     if file.endswith(".png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file[:-4]}")
#     if file.endswith(".png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file}")
#
# index = 0
# for file in os.listdir(rf"ThesisFinalData\scaled_images_rolling_window_6\test\images"):
#     print(f"4__{index}")
#     index += 1
#     if file.endswith(".png.png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file[:-12]}")
#     if file.endswith(".png.png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file[:-8]}")
#     if file.endswith(".png.png"):
#         shutil.move(rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file}", rf"ThesisFinalData\scaled_images_rolling_window_6\test\images\{file[:-4]}")

# import os
# import shutil
#
# from sklearn.model_selection import train_test_split
#
# os.chdir(r"D:\Pycharm Projects")
#
# images_list = os.listdir(rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\all images")
#
# images_list = [image for image in images_list if image.endswith("grayscale.png")]
#
# train_images, test_images = train_test_split(images_list, test_size=0.2, random_state=42)
#
# index = 0
# for image in train_images:
#     print(f"1__{index}")
#     index += 1
#     shutil.copy(rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\all images\{image}",
#                 rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\train\images\{image}")
#
# index = 0
# for image in test_images:
#     print(f"2__{index}")
#     index += 1
#     shutil.copy(rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\all images\{image}",
#                 rf"ThesisFinalData\smaller_scaled_images_rolling_window_6\test\images\{image}")
