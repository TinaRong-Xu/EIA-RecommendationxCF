# -*- "coding: utf-8" -*-

import numpy as np
import copy
import requests
from datetime import datetime
import xlsxwriter

api = 'https://restapi.amap.com/v3/geocode/regeo'
# 只需要一个密钥
# 高德api改版, 单个账号单日6000次调用量, 而不是单个密钥单日6000次调用量
keys = (
        "18d2c317cc7ccd4e4559fc8f113ce213",  # key0
        "a9b6d53af81e02ef0a9e0e864d058165",  # key1
        "94fd72799197f0b146c37c3c8fc323b6",  # key2
        "1f8a39aa445e51b959449b818e4a4a4c",  # key3
        "171e07fc8a407379e5394aebd7ada8d1",  # key4
        "0546b2077916fc48f99ab1cb752ffa44",  # key5
        "9bce418222ffe4c9520542b2817ff915",  # key6
        "b501637b6323845adb81b5a0946f6a91",  # key7
        "49ea5b64a8e6c2ff7594160d31fdfa69",  # key8
        "c9fde9674bdb8ab7b3770d71c0e4aac4",  # key9
    )
key_pool = (key for key in keys)
key = ''
filepath = "../data/20200729/siteinfo.csv"
savepath = "../data/20200729/siteinfo-1.csv"
batch_size = 20

log = open("log.txt", "a", encoding="utf-8", buffering=1)

def coordinate2address(longitudes, latitudes, step):

    provinces = []
    cities = []
    districts = []

    coordinates = "|".join(
            [str(round(longitude, 6))+","+str(round(latitude, 6)) for longitude, latitude in zip(longitudes, latitudes)])
    
    try:
        response = requests.get(api, params={"key": key, "location": coordinates, "batch": "true"}).json()
        if len(response["regeocodes"]) != step:
            raise Exception("contains invalid coordinate.")
    except Exception as e:
        print(type(e), ":", e, file=log)
        # exit(1)
        return (["[]"]*step, ["[]"]*step, ["[]"]*step)
    for regeocode in response["regeocodes"]:
        province = str(regeocode["addressComponent"]["province"])[:2]  # eg: 新疆
        city = str(regeocode["addressComponent"]["city"])
        if city == "[]":
            city = province + "市"
        district = str(regeocode["addressComponent"]["district"])

        provinces.append(province)
        cities.append(city)
        districts.append(district)

    return provinces, cities, districts

def process_data():

    global key
    key = next(key_pool)  # 获取第一把密钥

    with open(savepath, "w", encoding="utf-8", buffering=1) as fout:
        with open(filepath, "r", encoding="utf-8") as fin:
            fout.write(fin.readline())  # 跳过第一行

            line_cnt = 0
            line_total = 1081571
            request_cnt = 0
            
            lines = []
            batch = []
            line_count = 0
            for line in fin:

                line = line.replace('"',"")

                line_cnt += 1
                print("{}/{}, {}".format(line_cnt, line_total, request_cnt), end="\r")

                segments = line.strip().split(",")
                lines.append(segments)
                if segments[-3] not in ["异常", ""]:  # 跳过已有数据
                    continue
                batch.append(segments[1:3])
                line_count += 1
                if line_count == batch_size: # 凑够20个再一起处理
                    longitudes = [float(item[0]) for item in batch]
                    latitudes = [float(item[1]) for item in batch]
                    provinces, cities, districts = coordinate2address(longitudes, latitudes, line_count)
                    i = 0
                    for item in lines:
                        if item[-3] in ["异常", ""]:
                            item[-3] = provinces[i]
                            item[-2] = cities[i]
                            item[-1] = districts[i]
                            i += 1
                        fout.write(",".join(item)+"\n")
                    lines = []
                    batch = []
                    line_count = 0
                    request_cnt += 1
                    # 更新高德地图密钥
                    # if request_cnt % 5000 == 0:
                        # key = next(key_pool)

            if len(batch) > 0:  # 处理结尾部分
                longitudes = [float(item[0]) for item in batch]
                latitudes = [float(item[1]) for item in batch]
                provinces, cities, districts = coordinate2address(longitudes, latitudes, line_count)
                i = 0
                for item in lines:
                    if item[-3] in ["异常", ""]:
                        item[-3] = provinces[i]
                        item[-2] = cities[i]
                        item[-1] = districts[i]
                        i += 1
                    fout.write(",".join(item)+"\n")
                lines = []
                batch = []
                line_count = 0


def padding_file():
    with open(savepath, "r", encoding="utf-8") as fout:
        length = len(fout.readlines())
    with open(savepath, "a", encoding="utf-8", buffering=1) as fout:
        with open(filepath, "r", encoding="utf-8") as fin:
            cnt = 0
            for line in fin:
                if cnt >= length:
                    fout.write(line)
                cnt += 1


def find_difference():
    with open(savepath, "r", encoding="utf-8") as fout:
        with open(filepath, "r", encoding="utf-8") as fin:
            cnt = 0
            while True:
                a = fin.readline().replace('"', "")
                b = fout.readline().replace('"', "")
                if a.split(",")[0] != b.split(",")[0]:
                    print(a)
                    print(b)
                    break
                if not a:
                    break
                cnt += 1
            print(cnt)


if __name__ == "__main__":
    process_data()
    # padding_file()
    # find_difference()
