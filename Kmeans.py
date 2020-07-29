import numpy as np
import datetime

def init_k(data_list,k):
    centure_list = set()
    while len(centure_list) < k:
        centure_list.add(int(np.random.random(1) * len(data_list)))
    centure = []
    for i in centure_list:
        centure.append(np.array(data_list[i]))
    return centure

def distance(list1,list2):
    return np.dot(list1,list2)

def re_choose_centure(centure,data_list,data_label):
    for k in range(len(centure)):
        add_list = []
        for index in range(len(data_label)):
            if data_label[index] == k:
                add_list.append(np.array(data_list[index]))
        centure[k] = np.array(sum(add_list) / len(add_list)) if add_list else np.array(data_list[int(np.random.random(1))])
    return centure


def caculate_distance(data_list,k,round):
    data_label = [0] * len(data_list)
    centure = init_k(data_list,k)
    for r in range(round):
        for i in range(len(data_list)):
            min_dis = distance(data_list[i],centure[data_label[i]])
            for d in centure:
                if distance(data_list[i],d) < min_dis:
                    for ind in range(len(centure)):
                        if all(centure[ind] == d):
                            break
                    data_label[i] = ind

        centure = re_choose_centure(centure,data_list,data_label)
    return centure,data_label


def kmeans_xufive(ds, k, round):
    """k-means聚类算法

    k       - 指定分簇数量
    ds      - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
    """

    m, n = ds.shape  # m：样本数量，n：每个样本的属性值个数
    result = np.empty(m, dtype=np.int)  # m个样本的聚类结果
    cores = ds[np.random.choice(np.arange(m), k, replace=False)]  # 从m个数据样本中不重复地随机选择k个样本作为质心

    #for rou in range(round):  #  迭代计算
    count = 0
    mini = 100
    while True:
        count += 1
        ramdom_index = np.random.choice(np.arange(m), mini, replace=False)  #mini_batch
        mini_ds = ds[ramdom_index]
        d = np.square(np.repeat(mini_ds, k, axis=0).reshape(mini, k, n) - cores)
        distance = np.sqrt(np.sum(d, axis=2))  # ndarray(m, k)，每个样本距离k个质心的距离，共有m行
        index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号
        if (index_min == result[ramdom_index]).all():  # 如果样本聚类没有改变
            return result, cores,count  # 则返回聚类结果和质心数据

        result[ramdom_index] = index_min  # 重新分类
        for i in range(k):  # 遍历质心集
            items = ds[result == i]  # 找出对应当前质心的子样本集
            cores[i] = np.mean(items, axis=0)  # 以子样本集的均值作为当前质心的位置


if __name__ == '__main__':
    k = 10
    round = 20
    data_list = np.random.rand(100000, 100)
    start_time = datetime.datetime.now()
    #centure, data_label = caculate_distance(data_list,k,round)
    result, cores, count = kmeans_xufive(data_list,k,round)
    cost_time = (datetime.datetime.now() - start_time).seconds
    print ('cost_time: ',cost_time)
    print ('count: ',count)


