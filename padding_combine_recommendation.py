# -*- "coding: utf-8" -*-

from collections import Counter, defaultdict
import logging

from config import Config

def get_all_users():
    """
    找出所有用户名，便于后续对没有推荐结果的用户做补齐
    """

    users = []
    with open("./temp/username-category-clv_table.csv", "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            segments = line.strip().split(",")
            username = segments[0]
            users.append(username)
    return users


def get_popular_laws():
    """
    找出阅读量高的记录作为热门法律法规，用于第三类群体的推荐，以及前两类用户无推荐或推荐数量不足的补齐
    """
    
    laws = []
    with open(Config.view_records_path, "r", encoding="utf-8") as f:
        columns = [item.replace('"',"") for item in f.readline().strip().split('","')]
        law_index = columns.index("lawId")
        name_index = columns.index("nameCN")
        num_index = columns.index("documentNum")
        for line in f:
            segments = line.strip().split('","')
            law = segments[law_index].replace('"',"")
            name = segments[name_index].replace('"',"")
            num = segments[num_index].replace('"',"")
            laws.append((law,name,num))
    law_counter = sorted(dict(Counter(laws)).items(), key=lambda x:x[1], reverse=True)
    popular_laws = [item[0] for item in law_counter[:Config.n_recommendation]]
    return popular_laws


def combine():
    """
    合并前两类群体的推荐结果
    """

    success_users = defaultdict(int)
    cnt = 1
    with open(Config.result_path, "w", encoding="utf-8-sig") as fout:
        with open("./temp/recommendation_cn1.csv", "r", encoding="utf-8-sig") as f1:
            fout.write("序号,用户名,推荐文件中文名,推荐文件id,推荐文件标准号,相关度\n")
            f1.readline()
            for line in f1:
                username = line.strip().split(",")[0]
                success_users[username] += 1
                fout.write(str(cnt)+","+line)
                cnt += 1
        with open("./temp/recommendation_cn2.csv", "r", encoding="utf-8-sig") as f2:
            f2.readline()
            for line in f2:
                username = line.strip().split(",")[0]
                success_users[username] += 1
                fout.write(str(cnt)+","+line)
                cnt += 1
    return success_users, cnt


def padding(users, success_users, popular_laws, cnt):
    """
    追加第三类群体的推荐结果，同时补齐前两类群体中无推荐或推荐数量不足的情况
    """

    with open(Config.result_path, "a", encoding="utf-8-sig") as fout:
        for user in users:
            if user not in success_users:  # 单独为一类的用户 以及 第三类用户 直接推荐热门数据
                for law in popular_laws:
                    fout.write("{},{},{},{},{},{}\n".format(cnt, user, *law, 0))
                    cnt += 1
            elif success_users[user] < Config.n_recommendation:
                for index in range(Config.n_recommendation-success_users[user]):
                    law = popular_laws[index]
                    fout.write("{},{},{},{},{},{}\n".format(cnt, user, *law, 0))
                    cnt += 1


def run():
    logging.info("padding and combine result...")
    users = get_all_users()
    popular_laws = get_popular_laws()
    success_users, cnt = combine()
    padding(users, success_users, popular_laws, cnt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
