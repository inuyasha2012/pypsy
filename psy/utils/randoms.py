# coding=utf-8
import numpy as np


def _gen_item_dt(items):
    """
    生成试题字典，key为试题顺序，value为试题所在维度
    :param items: list(int), 试题维度排列列表
    :return: dict(int: int) 试题字典
    """
    item_dt = {}
    for i, item in enumerate(items):
        item_dt[i] = item
    return item_dt


def gen_item_bank(trait_size, item_size, block_size=3, lower=1, upper=4, avg=0, std=1):
    """
    生成用于自适应测验的题库
    :param trait_size: int
    :param item_size: int
    :param block_size: int
    :param lower: int|float
    :param upper: int|float
    :param avg: int|float
    :param std: int|float
    :return: 
    """
    if not isinstance(trait_size, int):
        raise ValueError('trait_size must be int')
    if not isinstance(item_size, int):
        raise ValueError('trait_size must be int')
    if block_size not in (2, 3):
        raise ValueError('block_size must be 2 or 3')
    if not isinstance(lower, (int, float)):
        raise ValueError('lower must be int or float')
    if not isinstance(upper, (int, float)):
        raise ValueError('upper must be int or float')
    if not isinstance(avg, (int, float)):
        raise ValueError('avg must be int or float')
    if not isinstance(std, (int, float)):
        raise ValueError('std must be int or float')

    trait_list = range(trait_size)
    item_bank = []
    for i in range(item_size):
        _item_list = np.random.choice(trait_list, block_size, False)
        _item_dt = _gen_item_dt(_item_list)
        params = random_params(_item_dt, trait_size, block_size=block_size, lower=lower, upper=upper, avg=avg, std=std)
        item_bank.append({'dim': _item_list, 'params': params})
    return np.array(item_bank)


def _pair_random_params(item_dt, trait_size, lower=1, upper=4, avg=0, std=1):
    """
    生成block_size为2的多维随机斜率（区分度）,和一维阈值（通俗度）
    :param std: 多维正态分布的标准差
    :param avg: 多维正态分布的期望值
    :param item_dt: dict(int:int), 试题字典
    :param trait_size: int，特质数量
    :param lower: int(>0), uniform分布的下界
    :param upper: int(>lower), uniform分布的上界
    :return: tuple(ndarray, ndarray), 斜率和阈值
    """
    keys = item_dt.keys()
    pair_nums = len(keys) / 2
    keys.sort()
    a = np.zeros((pair_nums, trait_size))
    a1 = np.random.uniform(lower, upper, pair_nums * 2)
    a2 = np.random.uniform(lower, upper, pair_nums * 2)

    for i in range(pair_nums):
        i1 = item_dt[2 * i]
        i2 = item_dt[2 * i + 1]
        a[i][i1] = a1[i]
        a[i][i2] = a2[2 * i] * -1
    b = np.random.normal(avg, std, pair_nums)
    return a, b


def _triplet_random_params(item_dt, trait_size, lower=1, upper=4, avg=0, std=1):
    """
    生成block_size为3的多维随机斜率（区分度）,和一维阈值（通俗度）
    :param std: 多维正态分布的标准差
    :param avg: 多维正态分布的期望值
    :param item_dt: dict(int:int), 试题字典
    :param trait_size: int，特质数量
    :param lower: int(>0), uniform分布的下界
    :param upper: int(>lower), uniform分布的上界
    :return: tuple(ndarray, ndarray), 斜率和阈值
    """
    keys = item_dt.keys()
    pair_nums = len(keys)
    keys.sort()
    a = np.zeros((pair_nums, trait_size))
    a1 = np.random.uniform(lower, upper, pair_nums)
    a2 = np.random.uniform(lower, upper, pair_nums)

    for i in range(len(keys) / 3):
        i1 = item_dt[3 * i]
        i2 = item_dt[3 * i + 1]
        i3 = item_dt[3 * i + 2]
        a[3 * i][i1] = a1[3 * i]
        a[3 * i][i2] = a2[3 * i] * -1
        a[3 * i + 1][i1] = a1[3 * i + 1]
        a[3 * i + 1][i3] = a2[3 * i + 1] * -1
        a[3 * i + 2][i2] = a1[3 * i + 2]
        a[3 * i + 2][i3] = a2[3 * i + 2] * -1
    b = np.random.normal(avg, std, pair_nums)
    return a, b


def random_params(item_dt, trait_size, block_size=3, lower=1, upper=4, avg=0, std=1):
    """
    生成随机参数
    :param item_dt: dict,试题字典，例如题块为3的0:1,1:0,2:2}代表第1题的第一个陈述测的是特质1，
    第二个陈述测的是特质0，第三个陈述测的是特质2
    :param trait_size: int
    :param block_size: int
    :param lower: int|float
    :param upper: int|float
    :param avg: int|float
    :param std: int|float
    :return: 
    """
    if not isinstance(item_dt, dict):
        raise ValueError('item_dt must be dict')
    if not isinstance(trait_size, int):
        raise ValueError('trait_size must be int')
    if block_size not in (2, 3):
        raise ValueError('block_size must be 2 or 3')
    if not isinstance(lower, (int, float)):
        raise ValueError('lower must be int or float')
    if not isinstance(upper, (int, float)):
        raise ValueError('upper must be int or float')
    if not isinstance(avg, (int, float)):
        raise ValueError('avg must be int or float')
    if not isinstance(std, (int, float)):
        raise ValueError('std must be int or float')

    if block_size == 3:
        return _triplet_random_params(item_dt, trait_size, lower=lower, upper=upper, avg=avg, std=std)
    elif block_size == 2:
        return _pair_random_params(item_dt, trait_size, lower=lower, upper=upper, avg=avg, std=std)