# -*- "coding: utf-8" -*-

class Config(object):
    login_records_path = "../data/20200729/siteinfo.csv"  # 用户登录记录所在路径
    view_records_path = "../data/20200729/lawView.csv"  # 用户浏览记录所在路径
    download_records_path = "../data/20200729/lawAttachmentDownload.csv"  # 用户附件下载记录所在路径
    result_path = "./recommendation_cn.csv"  # 推荐结果保存路径

    industry_weight = 0.7
    position_weight = 0.3
    view_weight = 0.5
    download_weight = 0.5

    n_neighbors = 50
    n_recommendation = 10
    step = 2000
