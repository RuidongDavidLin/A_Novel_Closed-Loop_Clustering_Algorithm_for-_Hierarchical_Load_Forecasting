import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def count_clusters(num_cluster,cluster_list):
    """
    统计每个簇里面的用户
    \ncount users in each cluster
    """
    user_belong = {str(index):[] for index in range(num_cluster)}
    for user_id, index in enumerate(cluster_list):
        user_belong[str(index)].append(user_id)
    return user_belong

def aggregate_by_cluster_list(load, labels, num_cluster, agg="sum", sparse=False):
    """
    基于 one-hot 的簇聚合：
    sums = H^T @ load
    其中 H[i,k] = 1 if labels[i]==k else 0

    参数
    ----
    load : (N, T) 每个用户的负荷曲线
    labels : (N,)  每个用户的簇标签，取值范围 [0, num_cluster-1]
    num_cluster : 簇的数量 K
    agg : "sum" | "mean" | "median"
    sparse : 是否使用稀疏 one-hot（scipy.sparse），适合大规模数据

    返回
    ----
    (K, T) 的聚合矩阵
    """
    load = np.asarray(load)
    labels = np.asarray(labels)
    N, T = load.shape
    K = num_cluster

    if sparse:
        # 稀疏 one-hot：更省内存（需要 scipy）
        rows = np.arange(N)
        data = np.ones(N, dtype=load.dtype)
        # H: (N, K)，每行只有一个 1
        H = sp.coo_matrix((data, (rows, labels)), shape=(N, K)).tocsr()
        # (K, N) @ (N, T) -> (K, T)
        sums = (H.T @ load)
        # 确保是 ndarray
        sums = sums if isinstance(sums, np.ndarray) else sums.toarray()
    else:
        # 纯 NumPy：H = eye(K)[labels] -> (N, K)
        H = np.eye(K, dtype=load.dtype)[labels]
        # (K, N) @ (N, T) -> (K, T)
        sums = H.T @ load

    if agg == "sum":
        return sums

    # 各簇样本数 (K,)
    counts = np.bincount(labels, minlength=K).astype(load.dtype)

    if agg == "mean":
        return np.divide(sums, counts[:, None],
                    out=np.zeros_like(sums, dtype=load.dtype),
                    where=counts[:, None] > 0)


    elif agg == "median":
        # 中位数无法用矩阵乘法一次完成；逐簇取 median
        out = np.zeros((K, T), dtype=load.dtype)
        for k in range(K):
            idx = (labels == k)
            if np.any(idx):
                out[k] = np.median(load[idx], axis=0)
            # 空簇保持 0
        return out

    else:
        raise ValueError("agg 只能是 'sum' | 'mean' | 'median'")

def change_rate_simple(old_labels, new_labels):
    old = np.asarray(old_labels).ravel()
    new = np.asarray(new_labels).ravel()
    assert old.shape == new.shape
    changed = np.count_nonzero(old != new)
    rate = changed / old.size
    return changed


def train_model(cluster_num, cluster_list, aggr_data, lag, train_ratio, Radiation, Calendar):
    """
    cluster_num: 初始聚类簇数 num of initial clulsters \n
    cluster_list: 每个用户所属簇列表 list including cluster_id of each user \n
    aggr_data: 每个簇聚合后负荷 aggregated load curve of each clusters \n
    lag: 时间滞后步长 timestep lag \n
    train_ratio: 训练集数据占比 \n
    temp_data: 对应温度数据 corresponding temperature data \n
    time_index: 时间戳 time index
    """
    model_dict = {str(index):None for index in range(cluster_num)}
    counts = np.bincount(cluster_list, minlength=cluster_num)
    for index in range(cluster_num):
        # 空簇不训练模型 no model trained for empty cluster
        if counts[index] == 0:
            pass
        else:
            x_data = aggr_data[index][:-lag-1].reshape(-1,1)
            x_data = np.concatenate((x_data,aggr_data[index][1:-lag].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,aggr_data[index][2:-lag+1].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,Radiation[1:-lag].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,Calendar[lag+1:].reshape(-1,8)),axis=1)
            
            y_data = aggr_data[index][lag+1:]
            x_train = x_data[:int(train_ratio*x_data.shape[0])]
            y_train = y_data[:int(train_ratio*y_data.shape[0])]
            model = LinearRegression()
            model.fit(X=x_train, y=y_train)
            model_dict[str(index)] = model
    return model_dict

def error_feedback(load_data, cluster_num, model_dict, lag, train_ratio, val_ratio, Radiation, Calendar):
    matrix = np.full((load_data.shape[0],cluster_num), 1e9)
    for index in range(cluster_num):
        for user_id in range(load_data.shape[0]):
            if model_dict[str(index)] != None:
                x_data = load_data[user_id][:-lag-1].reshape(-1,1)
                x_data = np.concatenate((x_data,load_data[user_id][1:-lag].reshape(-1,1)),axis=1)
                x_data = np.concatenate((x_data,load_data[user_id][2:-lag+1].reshape(-1,1)),axis=1)
                x_data = np.concatenate((x_data,Radiation[1:-lag].reshape(-1,1)),axis=1)
                x_data = np.concatenate((x_data,Calendar[lag+1:].reshape(-1,8)),axis=1)
                
                y_data = load_data[user_id][lag+1:]
                x_train = x_data[int(train_ratio*x_data.shape[0]):int((train_ratio+val_ratio)*x_data.shape[0])]
                y_train = y_data[int(train_ratio*y_data.shape[0]):int((train_ratio+val_ratio)*x_data.shape[0])]
                y_pred = model_dict[str(index)].predict(x_train)
                error = np.mean(np.abs(y_pred-y_train)/y_train)
                matrix[user_id, index] = error
    return matrix


def Predict(cluster_num, cluster_list, aggr_data, lag, train_ratio, val_ratio, Radiation, Calendar, model_dict):
    """
    cluster_num: 聚类簇数 num of clulsters \n
    cluster_list: 每个用户所属簇列表 list including cluster_id of each user \n
    aggr_data: 每个簇聚合后负荷 aggregated load curve of each clusters \n
    lag: 时间滞后步长 timestep lag \n
    train_ratio: 训练集数据占比 ratio of training dataset\n
    val_ratio: 验证集数据占比 ratio of validation dataset \n
    temp_data: 对应温度数据 corresponding temperature data \n
    time_index: 时间戳 time index
    """
    predict_result = []
    test_result = []
    counts = np.bincount(cluster_list, minlength=cluster_num)
    for index in range(cluster_num):
        # 空簇不训练模型 no model trained for empty cluster
        if counts[index] == 0:
            pass
        else:
            x_data = aggr_data[index][:-lag-1].reshape(-1,1)
            x_data = np.concatenate((x_data,aggr_data[index][1:-lag].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,aggr_data[index][2:-lag+1].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,Radiation[1:-lag].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,Calendar[lag+1:].reshape(-1,8)),axis=1)
            y_data = aggr_data[index][lag+1:]
            
            x_test = x_data[int((train_ratio+val_ratio)*x_data.shape[0]):]
            y_test = y_data[int((train_ratio+val_ratio)*x_data.shape[0]):]
            y_pred  = model_dict[str(index)].predict(x_test)
            predict_result.append(y_pred)
            test_result.append(y_test)
    return np.asarray(predict_result), np.asarray(test_result)

def Predict_val(cluster_num, cluster_list, aggr_data, lag, train_ratio, val_ratio, Radiation, Calendar, model_dict):
    """
    cluster_num: 聚类簇数 num of clulsters \n
    cluster_list: 每个用户所属簇列表 list including cluster_id of each user \n
    aggr_data: 每个簇聚合后负荷 aggregated load curve of each clusters \n
    lag: 时间滞后步长 timestep lag \n
    train_ratio: 训练集数据占比 ratio of training dataset\n
    val_ratio: 验证集数据占比 ratio of validation dataset \n
    temp_data: 对应温度数据 corresponding temperature data \n
    time_index: 时间戳 time index
    """
    predict_result = []
    test_result = []
    counts = np.bincount(cluster_list, minlength=cluster_num)
    for index in range(cluster_num):
        # 空簇不训练模型 no model trained for empty cluster
        if counts[index] == 0:
            pass
        else:
            x_data = aggr_data[index][:-lag-1].reshape(-1,1)
            x_data = np.concatenate((x_data,aggr_data[index][1:-lag].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,aggr_data[index][2:-lag+1].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,Radiation[1:-lag].reshape(-1,1)),axis=1)
            x_data = np.concatenate((x_data,Calendar[lag+1:].reshape(-1,8)),axis=1)
            y_data = aggr_data[index][lag+1:]
            
            x_val = x_data[int((train_ratio)*x_data.shape[0]):int((train_ratio+val_ratio)*x_data.shape[0])]
            y_val = y_data[int((train_ratio)*x_data.shape[0]):int((train_ratio+val_ratio)*x_data.shape[0])]
            y_pred  = model_dict[str(index)].predict(x_val)
            predict_result.append(y_pred)
            test_result.append(y_val)
    return np.asarray(predict_result), np.asarray(test_result)