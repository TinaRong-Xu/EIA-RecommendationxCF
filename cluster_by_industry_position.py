# -*- "coding: utf-8" -*-

from collections import defaultdict
import pickle
import os
import logging
import sys

import numpy as np
from sklearn.cluster import AffinityPropagation

from config import Config


def get_user_industry(users=None):
    """
    找出每个用户所在的行业

    统计用户过去三个月浏览文件（Config.view_records_path）的适用行业字段，频率最高的行业作为用户所在行业

    view_records_path对应的文件包含这些列：
    "lawId","nameCN","classificationTwo","documentNum","pubDate",
    "implDate","isFail","area","departmenttext","nec",
    "industry","drafttext","openTime","username","phoneId"
    """

    user_industy_count = defaultdict(lambda :defaultdict(int))
    with open(Config.view_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split('","')]
        # username_index = columns.index("phoneId")
        username_index = columns.index("username")  # 使用 用户名username 而不是 设备编号phoneId 作为 用户标识
        industry_index = columns.index("industry")
        for line in f:
            segments = line.strip().split('","')
            username = segments[username_index].replace('"',"")
            industry = segments[industry_index].replace('"',"").replace(",","，").split("，")[0]  # 对于逗号分隔的多个行业只取第一个
            if users != None and username not in users:  # 只统计当前类别群体（如：一类群体）的数据
                continue
            if username != "" and industry != "所有行业" and industry != "":  # 不统计 空用户 空行业 和 所有行业
                user_industy_count[username][industry] += 1

    # 保存统计结果
    with open("./temp/user_big_table.csv", "w", encoding="utf-8-sig") as f:
        for username in user_industy_count:
            most_common_industry = None
            most_common_count = 0
            for industry in user_industy_count[username]:
                cur_count = user_industy_count[username][industry]
                if most_common_industry is None or cur_count > most_common_count:
                    most_common_count = cur_count
                    most_common_industry = industry
            if most_common_industry is None:  # 没有结果则为所有行业
                most_common_industry = "所有行业"
            f.write("{},{}\n".format(username, most_common_industry))


def get_user_position(users=None):
    """
    获取所有用户所在的省份
    
    登录数据（Config.login_records_path）中，记录最后一次登录的省份。

    login_records_path对应的文件包含这些列：
    "id","longitude","latitude","createtime","telphone","telephoneid","telphoneType","province","city","county"
    """
    
    user_position = defaultdict(str)
    with open(Config.login_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split(',')]
        # username_index = columns.index("phoneId")
        username_index = columns.index("telphone")  # 使用 username 而不是 phoneId 作为 用户标识
        position_index = columns.index("province")
        for line in f:
            segments = line.strip().split(',')
            username = segments[username_index].replace('"',"")
            position = segments[position_index].replace('"',"")
            if users != None and username not in users:  # 只统计当前类别群体的数据
                continue
            if username != "" and position not in ("[]", "", "异常"):
                user_position[username] = position  # 只保留最新的结果，相当于只保留最近登录的数据

    # 保存结果
    with open("./temp/user_big_table1.csv", "w", encoding="utf-8-sig") as fin:
        with open("./temp/user_big_table.csv", "r", encoding="utf-8-sig") as fout:
            for line in fout:
                username = line.strip().split(",")[0]
                new_line = line.strip() + ",{}\n".format(user_position[username])
                fin.write(new_line)


def generate_similarity_matrix():
    """
    根据用户所在行业和位置，按照Config中设定的权重，累加分值

    如：行业权重0.7，位置权重0.3，初始相似度均为0
    则：当两个用户的行业相同时，两者之间的相似度加 1*0.7
        当两个用户的位置相同时，两者之间的相似度加 1*0.3
    """

    data = []
    with open("./temp/user_big_table1.csv", "r", encoding="utf-8-sig") as f:
        for line in f:
            segments = line.strip().split(",")
            data.append((segments[1], segments[2]))  # (industry, position)
    n_users = len(data)
    data_mat = np.zeros((n_users, n_users), dtype=np.float32)
    cnt = 0
    for i in range(n_users):
        cnt += 1
        print("{}/{}".format(cnt, n_users), end="\r")
        for j in range(i, n_users):
            if data[i][0] == data[j][0]:  # industry
                data_mat[i,j] += 1*Config.industry_weight
                data_mat[j,i] = data_mat[i,j]
            if data[i][1] != "" and data[i][1] == data[j][1]:  # position (position != "")
                data_mat[i,j] += 1*Config.position_weight
                data_mat[j,i] = data_mat[i,j]
    print("{}/{}".format(cnt, n_users))
    
    # with open("./temp/user_similarity_mat.pkl", "wb") as f:  # 保存相似度矩阵
        # pickle.dump(data_mat, f, -1)
    return data_mat


def AP_cluster(similarities, preference=1):
    """
    根据预先计算好的用户间的相似度，使用近邻传播聚类算法，将用户聚成数量不定的簇
    """

    # fit的输入X是计算好的相似度矩阵, 因此affinity为precomputed, 而affinity默认值为euclidean, 表示使用欧氏距离的负数
    # preference为相似度矩阵对角线上的值, cosine 相似度为1
    af = AffinityPropagation(preference=preference, affinity="precomputed").fit(similarities)
    centers = af.cluster_centers_indices_  # 聚类中心, 指示item的位置
    labels = af.labels_  # 标签列表, 每个item赋予一个类别标签
    return centers, labels
  

def save_cluster_result(labels, centers, similarities, tag, users):
    """
    保存聚类结果

    结果包含这几列：
    簇内的第一个用户名（用于标识每一簇）, 簇内用户的相似度均值, 簇内的所有用户名列表

    """

    result = {}
    for center in centers:
        result[center] = []
    label2center = {idx:center for idx, center in enumerate(centers)}
    for idx, label in enumerate(labels):
        result[label2center[label]].append(idx)

    print("save csv file...")
    with open("./temp/userid-userlist_cluster{}.csv".format(tag), "w", encoding="utf-8") as f:
        for center, line in result.items():
            t_similarities = []
            for i in range(len(line)):
                for j in range(i+1, len(line)):
                    t_similarities.append(similarities[line[i]][line[j]])
            if sum(t_similarities) == 0:
                t_mean = 0
            else:
                t_mean = np.mean(t_similarities)
            f.write("{},{},{}\n".format(users[center], t_mean, list(map(lambda x: users[x], line))))
    return result


def filter_users(tag):
    """
    过滤出属于当前群体（如：一类群体）的所有用户名，便于后续只针对当前群体做计算
    """

    users = []
    with open("./temp/username-category-clv_table.csv", "r", encoding="utf-8-sig") as f:
        tag_index = f.readline().strip().split(",").index("tag")
        for line in f:
            segments = line.strip().split(",")
            username = segments[0]
            val = segments[tag_index]
            if val == str(tag):
                users.append(username)

    return users


def run(tag):
    logging.info("in cluster_by_industry_position.py:")
    logging.info("industry_weight:{}, position_weight:{}".format(Config.industry_weight, Config.position_weight))

    if tag in (1, 2):
        users = filter_users(tag)
    else:
        print("Invalid tag value: {}, tag should in (1, 2)".format(tag), file=sys.stderr)

    logging.info("get user industry...")
    get_user_industry(users)
    logging.info("get user position...")
    get_user_position(users)
    logging.info("generate similarity matrix...")
    similarities = generate_similarity_matrix()
    print(similarities.shape)
    logging.info("AP cluster and save result...")
    centers, labels = AP_cluster(similarities)
    save_cluster_result(labels, centers, similarities, tag, users)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(1)
