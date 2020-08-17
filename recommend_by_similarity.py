# -*- "coding: utf-8" -*-

from collections import defaultdict
import logging
import pickle
import os

import numpy as np
import pandas as pd

from config import Config


def load_view_records(user_list=None):
    """
    读取所有浏览记录和所有下载记录，按Config中设置的权重计算分值

    如：
    """

    users = set()
    laws = set()

    ##########
    # 找出浏览记录和下载记录所涉及的所有用户和法律法规，便于构建用户的偏好矩阵
    ##########

    # logging.info("filter and save all users and laws...")
    with open(Config.view_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split('","')]
        username_index = columns.index("username")
        law_id_index = columns.index("lawId")
        for line in f:
            segments = line.strip().split('","')
            username = segments[username_index].replace('"',"")
            if user_list != None and username not in user_list:
                continue
            law_id = segments[law_id_index].replace('"',"")
            if law_id == "undefined":
                continue
            if username != "":
                users.add(username)
                laws.add(law_id)

    with open(Config.download_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split('","')]
        username_index = columns.index("username")
        law_id_index = columns.index("lawId")
        for line in f:
            segments = line.strip().split('","')
            username = segments[username_index].replace('"',"")
            if user_list != None and username not in user_list:
                continue
            law_id = segments[law_id_index].replace('"',"")
            if law_id == "undefined":
                continue
            if username != "":
                users.add(username)
                laws.add(law_id)

    users2id = {item:idx for idx, item in enumerate(users)}
    laws2id = {item:idx for idx, item in enumerate(laws)}

    with open("./temp/users2id.pkl", "wb") as f:
        pickle.dump(users2id, f, -1)
    with open("./temp/laws2id.pkl", "wb") as f:
        pickle.dump(laws2id, f, -1)


    ##########
    # 构建并保存用户的偏好矩阵
    ##########
    # logging.info("generate and save user-law score matrix...")
    users = tuple(users)  # set类型无序，转为tuple类型固定次序
    laws = tuple(laws)
    n_user = len(users)
    n_law = len(laws)
    user_law_mat = np.zeros((n_user, n_law), dtype=np.float32)  # 偏好矩阵，评分初始化为0，使用float32以上类型加快矩阵运算

    with open(Config.view_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split('","')]
        username_index = columns.index("username")
        law_id_index = columns.index("lawId")
        for line in f:
            segments = line.strip().split('","')
            username = segments[username_index].replace('"',"")
            if user_list != None and username not in user_list:
                continue
            law_id = segments[law_id_index].replace('"',"")
            if law_id == "undefined":
                continue
            if username != "":
                user_law_mat[users2id[username]][laws2id[law_id]] += Config.view_weight * 1

    with open(Config.download_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split('","')]
        username_index = columns.index("username")
        law_id_index = columns.index("lawId")
        for line in f:
            segments = line.strip().split('","')
            username = segments[username_index].replace('"',"")
            if user_list != None and username not in user_list:
                continue
            law_id = segments[law_id_index].replace('"',"")
            if law_id == "undefined":
                continue
            if username != "":
                user_law_mat[users2id[username]][laws2id[law_id]] += Config.download_weight * 1

    with open("./temp/user_law_mat.pkl", "wb") as f:  # 保存用户的偏好矩阵，将在推荐法律法规时用到
        pickle.dump(user_law_mat, f, -1)

    return user_law_mat, users, laws


def _cal_cosine(X, Y):
    '''
    计算余弦相似度，X 为左矩阵，Y 为右矩阵
    Args:
        X: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
        Y: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
    Returns:
        cosine: ndarray 第i行第j列的值是第i个元素和第j个元素之间的相似度
    '''
    numerator = X.dot(Y.T)  # 向量 a 乘以向量 b (分子)
    norm_X = np.linalg.norm(X, axis=1)  # 每行一个范数，结果为一维 ndarray 数据
    norm_Y = np.linalg.norm(Y, axis=1)  # 每行一个范数，结果为一维 ndarray 数据
    # 为防止零作除数做的处理
    cosine = numerator /np.maximum(np.expand_dims(norm_X, axis=1),10e-20) /np.maximum(np.expand_dims(norm_Y, axis=0),10e-20)
    return cosine


def find_neighbors(X, Y):
    '''
    组织数据分块计算相似度
    Args:
        X: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
        Y: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
    Returns:
        similarities: ndarray 第i行第j列的值是第i个元素和第j个元素之间的相似度
    '''

    f = open("./temp/user_neighbors.csv", "w", encoding="utf-8-sig")

    pointer = 0
    cnt = 0
    total = X.shape[0]//Config.step +1
    user_index = 0
    while pointer <= X.shape[0]:
        cnt += 1
        # logging.info("{}/{}".format(cnt, total))

        similarity = _cal_cosine(X[pointer:pointer+Config.step], Y)
        pointer += Config.step

        ##########
        # save result
        # 保存用户的邻居列表，将在推荐法律法规时用到
        # 每计算一步保存一次，避免持续增加内存占用
        ##########
        indices = np.argsort(similarity, axis=1)[:, ::-1][:, :Config.n_neighbors+1]  # 包含自身
        scores = np.sort(similarity, axis=1)[:, ::-1][:, :Config.n_neighbors+1]  # 包含自身
        for row in range(indices.shape[0]):
            for col in range(indices.shape[1]):
                user = user_index
                neighbor = indices[row, col]
                score = scores[row, col]
                if user != neighbor:  # 不考虑自身
                    if score > 0:  # 只保留相似度大于0的邻居
                        f.write("{},{},{}\n".format(user,neighbor,score))
                    else:
                        break
            user_index += 1
    f.close()


def recommendation_laws(users, laws, fout):
    """
    根据用户的偏好矩阵和用户的邻居列表生成推荐结果，Config中设置了推荐的结果数量
    """

    ##########
    # 加载用户的偏好矩阵 load_view_records 中生成
    ##########
    with open("./temp/user_law_mat.pkl", "rb") as f:
        user_law_mat = pickle.load(f)

    ##########
    # 加载用户的邻居列表 find_neighbors 中生成
    ##########
    law_score = defaultdict(float)
    with open("./temp/user_neighbors.csv", "r", encoding="utf-8-sig") as f:
        cur_user = None
        for line in f:
            user, neighbor, similarity = line.strip().split(",")
            user, neighbor, similarity = int(user), int(neighbor), float(similarity)
            # 找出邻居看过或下载过(分值>1)但当前用户没看过也每下载过(分值=0)的法律法规
            difference = user_law_mat[neighbor] - user_law_mat[user]  # 用 neighbor - user
            law_indices = np.where(difference>1)[0]  # 筛选结果中分值>1的法律法规即为所求
            for law_idx in law_indices:
                # 当前法律法规的得分 = 邻居对该项法律法规的偏好分值 * 当前用户和邻居之间的相似度
                law_score[law_idx] += user_law_mat[neighbor][law_idx] * similarity

            if cur_user is None:
                cur_user = user
            elif user != cur_user:  # 保存上一个用户的推荐结果
                sorted_law_score = sorted(law_score.items(), key=lambda x:x[1], reverse=True)[:Config.n_recommendation]
                # # 归一化 score
                # try:
                #     max_score = sorted_law_score[0][1]
                #     min_score = sorted_law_score[-1][1]
                # except IndexError as e:  # 无推荐结果则跳过（无相似用户 或相似用户的浏览记录中无当前用户未浏览过的文件）
                #     print(user, users[user], sorted_law_score)
                # for law, score in sorted_law_score:
                #     fout.write("{},{},{}\n".format(users[user], laws[law], (score-min_score)/max(max_score-min_score, 1e-5) ))
                # 不对 score 归一化
                for law, score in sorted_law_score:
                    fout.write("{},{},{}\n".format(users[user], laws[law], score))
                cur_user = user
                law_score = defaultdict(float)


def convert_recommendation(tag):
    """
    对推荐结果稍作转换，增加法律法规的中文名、标准号这两项数据
    """

    # load all laws
    cols = ["lawId", "nameCN", "documentNum"]
    laws = pd.read_csv(Config.view_records_path, encoding="utf-8", usecols=cols)
    laws.drop_duplicates(inplace=True)
    laws.fillna('', inplace=True)  # 将NaN都替换为空字符串
    laws.set_index("lawId", inplace=True)  # 将law id 设为索引

    # save
    with open("./temp/user_recommendation.csv", "r", encoding="utf-8-sig") as fin:
        with open("./temp/recommendation_cn{}.csv".format(tag), "w", encoding="utf-8-sig", buffering=1) as fout:
            fout.write("用户名,推荐文件中文名,推荐文件id,推荐文件标准号,相关度\n")
            cnt = 0
            for line in fin:
                username, lawid, score = line.strip().split(",")
                if lawid == "undefined":
                    continue
                name_cn = laws.loc[lawid, "nameCN"].replace(",","，")
                document_num = laws.loc[lawid, "documentNum"]
                score = round(float(score), 8)
                cnt += 1
                print("{}".format(cnt), end="\r")
                fout.write("{},{},{},{},{}\n".format(username, name_cn, lawid, document_num, score))
            print("{}".format(cnt))


def run(tag):
    logging.info("in recommend_by_similarity.py:")
    logging.info("industry_weight:{}, position_weight:{}".format(Config.industry_weight, Config.position_weight))

    logging.info("recommend for each group of users...")
    fout = open("./temp/user_recommendation.csv", "w", encoding="utf-8-sig", buffering=1)
    with open("./temp/userid-userlist_cluster{}.csv".format(tag), "r", encoding="utf-8") as f:
        cnt = 0
        # 对于每一簇的用户单独进行推荐，即，只在同一簇内寻找相似用户
        for line in f:
            cnt += 1
            print("{}".format(cnt), end="\r")
            user_list = eval(",".join(line.strip().split(",")[2:]))
            if len(user_list) == 1:  # 一个元素单独一类的不寻找相似用户
                continue
            user_law_mat, users, laws = load_view_records(user_list)
            # logging.info("find neighbors...")
            find_neighbors(user_law_mat, user_law_mat)
            # logging.info("generate and save recommendations...")
            recommendation_laws(users, laws, fout)
            # logging.info("convert recommendation...")
        print("{}".format(cnt))
    fout.close()
    convert_recommendation(tag)


if __name__ == "__main__":

    # logging.basicConfig(filename="log.txt", filemode='a',
                        # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        # datefmt='%H:%M:%S', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)
    run(1)
