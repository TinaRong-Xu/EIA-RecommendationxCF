# -*- "coding: utf-8" -*-

import pandas as pd
import numpy as np

from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

import datetime
import logging
import os

from config import Config


def user_division():
    """
    将用户划分为 高级用户、中级用户、初级用户
    
    高级用户：
        频繁出差（过去半年打开位置大于等于3个，精确到地级市）+ 近一月使用且使用频率高（>10次/月）
    中级用户：
        基本不发生移动（近半年打开位置小于三个，精确到地级市）+ 近一月打开且打开频率中及以上（>5次/月）
        + 频繁出差（过去半年打开位置大于等于3个，精确到地级市）近一月使用且使用频率较低（<=10次/月）        
    初级用户：
        不属于上述两类的用户
    
    1. 下面类似“10次/月”的条件，是针对近一月使用次数的统计
    2. 目前只提供了最近三个月的数据，所以实际只统计了过去三个月内的位置变动情况，
       只要将来提供的数据超过了半年，就将统计半年的数据，不需要修改代码
    """

    # 统计打开位置
    login_records = pd.read_csv(
            Config.login_records_path, 
            usecols=["telphone", "createtime", "city"], 
            na_values=["[]", "[]市"], 
            dtype=str
        )
    login_records.dropna(inplace=True)
    login_records.sort_values(by=["createtime"], inplace=True)

    last_date = login_records.iloc[-1]["createtime"]  # 取数据中的最后一天
    half_year_ago = datetime.datetime.strftime(
            datetime.datetime.strptime(last_date, "%Y/%m/%d %H:%M:%S")-datetime.timedelta(days=6*30),
            "%Y/%m/%d %H:%M:%S"
        )  # 时间前推 6*30天，作为最近半年数据的分界线
    login_records = login_records[login_records["createtime"]>half_year_ago]  # 只保留最近半年的数据
    user_positions = login_records.groupby(by=["telphone"])["city"].unique()  # 按用户名分组，得到每个用户的登录地点
    user_positions.rename("position", inplace=True)

    # 统计打开频率
    a_month_ago = datetime.datetime.strftime(
            datetime.datetime.strptime(last_date, "%Y/%m/%d %H:%M:%S")-datetime.timedelta(days=30),
            "%Y/%m/%d %H:%M:%S"
        )  # 时间前推 30天，作为最近一个月数据的分界线
    a_month_data = login_records[login_records["createtime"]>a_month_ago]  # 过滤出最近一个月的数据
    user_count = a_month_data.groupby(by=["telphone"]).size()  # 按用户名分组，得到每个用户的登录次数
    user_count.rename("count", inplace=True)

    # 合并结果
    table = pd.concat([user_positions, user_count], axis=1).reset_index()

    # 划分用户
    result = {"senior":set(), "middle":set(), "primary":set()}
    def loop(line):
        position_count = len(line["position"])
        login_count = line["count"]
        username = line["telphone"]
        if position_count >= 3 and login_count > 10:
            result["senior"].add(username)
        elif (position_count >= 3 and login_count <= 10) or (position_count < 3 and login_count > 5):
            result["middle"].add(username)
        else:
            result["primary"].add(username)
    table.apply(loop, axis=1)
    # print(len(result["senior"]), len(result["middle"]), len(result["primary"]))

    return result


def cal_clv():
    """
    计算每个用户的CLV值
    """

    # 准备数据
    login_records = pd.read_csv(
            Config.login_records_path, 
            usecols=["telphone", "createtime"], 
            dtype={"telphone":str, "createtime":str}
        )
    login_records.dropna(inplace=True)
    last_date = datetime.datetime.strptime(login_records.iloc[-1]["createtime"], "%Y/%m/%d %H:%M:%S")
    last_date = "{}-{}-{}".format(last_date.year, last_date.month, last_date.day)
    half_year_ago = datetime.datetime.strftime(datetime.datetime.strptime(last_date, "%Y-%m-%d")-datetime.timedelta(days=6*30), "%Y/%m/%d %H:%M:%S")
    login_records = login_records[login_records["createtime"]>half_year_ago]  # 为保证用户名能一一对应，同样只保留最近半年的数据
    login_records["monetary_value"] = [1]*login_records.shape[0]  # 用户每登录一次，视为消费金额加1，用于通过GGF计算CLV值
    login_records.sort_values(by=["createtime"], inplace=True)

    # 统计: frequency, recency, T, monetary_value
    summary = summary_data_from_transaction_data(login_records, "telphone", "createtime", "monetary_value", observation_period_end=last_date)
    returning_summary = summary[summary["frequency"]>0]  # 只使用frequency大于0的数据对GGF建模

    # BGF模型
    bgf = BetaGeoFitter(penalizer_coef=0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

    # GGF模型
    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(returning_summary['frequency'], returning_summary['monetary_value'])

    # 计算 clv 值
    username_clv = ggf.customer_lifetime_value(
            bgf, # the model to use to predict the number of future transactions
            summary['frequency'],
            summary['recency'],
            summary['T'],
            summary['monetary_value'],
            time=12, # months
            discount_rate=0 # monthly discount rate
    )

    return username_clv


def save_result(category_username, username_clv):
    """
    根据用户的等级和CLV值，将用户划分为一、二、三类群体

    结果包含这几列：
        username: 用户名
        category: 高级用户(senior)、中级用户(middle)、初级用户(primary);
        clv: 高CLV(high), 低CLV(low);
        tag: 一类群体(1), 二类群体(2), 三类群体(3);
    """
    
    mean_clv = username_clv.mean()
    with open("./temp/username-category-clv_table.csv", "w", encoding="utf-8-sig") as f:
        f.write("username,category,clv,tag\n")
        for category in category_username:
            for username in category_username[category]:
                clv_tag = "high" if username_clv.loc[username] > mean_clv else "low"  # clv 大于均值的就认为是高 clv
                if category == "senior" and clv_tag == "high":
                    tag = 1
                elif category == "senior" and clv_tag == "low":
                    tag = 2
                elif category == "middle" and clv_tag == "high":
                    tag = 2
                elif category == "middle" and clv_tag == "low":
                    tag = 3
                elif category == "primary" and clv_tag == "high":
                    tag = 2
                elif category == "primary" and clv_tag == "low":
                    tag = 3
                f.write("{},{},{},{}\n".format(username, category, clv_tag, tag))


def run():
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists("./temp/"):
        os.mkdir("./temp/")

    logging.info("user division...")
    category_username = user_division()
    logging.info("calculate clv...")
    username_clv = cal_clv()
    logging.info("save result...")
    save_result(category_username, username_clv)


if __name__ == "__main__":
    run()
