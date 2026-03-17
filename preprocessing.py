# from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
#
# # from keras.utils import to_categorical
# from os.path import join
# import torch
# from sklearn import preprocessing
# import numpy as np
# import pandas as pd
# import glob as gb
#
# import matplotlib.pyplot as plt
#
# # global variable
# seed = 124
# eps = 1e-15
#
#
# # read all the csv files
# def read_data(dataroot, file_ending='*.csv'):
#     if file_ending is None:
#         print("please specify file ending pattern for glob")
#         exit()
#     print(join(dataroot, file_ending))
#     filenames = [i for i in gb.glob(join(dataroot, file_ending))]
#     combined_csv = pd.concat([pd.read_csv(f, dtype=object) for f in filenames], sort=False,
#                              ignore_index=True)  # dopisałem ignore_index=True
#     return combined_csv
#
#
# # read one csv file
# def read_single_data(datafile):
#     out_csv = pd.read_csv(datafile)
#     return out_csv
# def clean(data,label,index):
#     data=data.to_numpy()
#     label=label.to_numpy()
#     data=np.delete(data,index,axis=0)
#     label=np.delete(label,index,axis=0)
#     return data,label
# def normalization(M):
#     M=M/np.sqrt(np.dot(M,M.T))
#     return M
# # load the csv files in data frame and start normalizing
# def load_data_train(*params):
#     # d = {"1.0": 0, "3.0": 1, "5.0": 2, "6.0": 3, "7.0": 4, "9.0": 5, "11.0": 6, "13.0": 7, "22.0": 8, "25.0": 9, "28.0": 10,
#     #      "32.0": 11, "33.0": 12, "34.0": 13, "35.0": 14, "42.0": 15, "44.0": 16, "47.0": 17, "49.0": 18, "51.0": 19}
#     d = {'Collecting': 0, 'bowing': 1, 'cleaning': 2, 'drinking': 3, 'eating': 4, 'looking': 5, 'opening': 6,
#          'passing': 7, 'picking': 8, 'placing': 9, 'pushing': 10, 'reading': 11, 'sitting': 12, 'standing': 13,
#          'standing_up': 14, 'talking': 15, 'turing_front': 16, 'turning': 17, 'walking': 18}
#     # dataroot = ".../train_data/pixel/"
#     dataroot = params[0]
#     end=params[1]
#     idx=params[2]
#     # data_path = read_data(dataroot, '*.csv')
#     # num_records, num_features = data_path.shape
#     # print("there are {} flow records with {} feature dimension".format(num_records, num_features))
#     # # there is white spaces in columns names e.g. ' Destination Port'
#     # # So strip the whitespace from  column names
#     # data = data_path.rename(columns=lambda x: x.strip())
#     # print('stripped column names')
#     # print('dropped bad columns')
#     # data.replace([np.inf, -np.inf], np.nan).dropna()
#     # num_records, num_features = data.shape
#     # print("there are new {} flow records with {} feature dimension".format(num_records, num_features))
#     # data = data.drop(data[data.label =="not specified"].index)
#     # data = data.sample(frac=1.0)  # 打乱所有数据
#     # data = data.reset_index(drop=True)
#     if idx==1:
#         data=pd.read_csv("weight/train-all-correct-2.csv")
#         # data = pd.read_csv(
#         #     '/mnt/storage/buildwin/desk_backword/sota/DIVA-master/dataset/wut/pku_train_new20_filted.csv')
#
#
#     else:
#         data = pd.read_csv("weight/test-all-correct-2.csv")
#         # data = pd.read_csv( '/mnt/storage/buildwin/desk_backword/pku mmd/video-tets/data_all.csv')
#     # if idx == 1:
#     #     data.to_csv("weight/train-all-correct-2.csv",index=False)
#     # else:
#     #     data.to_csv("weight/test-all-correct-2.csv",index=False)
#     label=data.iloc[:,-1]
#     data=data.iloc[:,:-1]
#     # data,label=clean(data,label)
#     data = data.astype('float64')
#     nan_count = data.isnull().sum().sum()
#     if nan_count > 0:
#         print("nan")
#         data.fillna(data.mean(), inplace=True)
#         # print('filled NAN')
#         # Normalising all numerical features:
#         # cols_to_norm = list(data.columns.values)[:68]
#         # print('cols_to_norm:\n', cols_to_norm)
#     data = data.astype(np.float32)
#     data = data.astype(float).apply(pd.to_numeric)
#     for i, r in data.iterrows():
#         t = normalization(r)
#         data.at[i] = t
#     data,label=clean(data,label,data[data.isnull().T.any()].index.to_numpy())
#     print(np.unique(label))
#
#     label2 = []
#     for i in label:
#         label2.append(d[str(i)])
#     label2 = torch.from_numpy(np.array(label2).astype(np.float32)).clone()
#     print(np.unique(label2.to('cpu').detach().numpy().copy()))
#     print(label2.unique(),"label2")
#     data = data.reshape((data.shape[0],  57))
#     data = torch.from_numpy(data.astype(np.float32)).clone()
#     print(data.type,213)
#     return data,label2,label
import pandas as pd
import numpy as np
import torch
import os
import glob

# ---------------------------------------------------------
# 1. 数据集配置 (Label Maps)
# ---------------------------------------------------------
LABEL_MAP_WUT = {
    'Collecting': 0, 'bowing': 1, 'cleaning': 2, 'drinking': 3, 'eating': 4,
    'looking': 5, 'opening': 6, 'passing': 7, 'picking': 8, 'placing': 9,
    'pushing': 10, 'reading': 11, 'sitting': 12, 'standing': 13,
    'standing_up': 14, 'talking': 15, 'turing_front': 16, 'turning': 17, 'walking': 18
}

LABEL_MAP_PKU = {
    "1.0": 0, "3.0": 1, "5.0": 2, "6.0": 3, "7.0": 4,
    "9.0": 5, "11.0": 6, "13.0": 7, "22.0": 8, "25.0": 9,
    "28.0": 10, "32.0": 11, "33.0": 12, "34.0": 13, "35.0": 14,
    "42.0": 15, "44.0": 16, "47.0": 17, "49.0": 18, "51.0": 19
}

DATASET_CONFIGS = {
    'wut': LABEL_MAP_WUT,
    'pku': LABEL_MAP_PKU
}

# [修改] 移除了强制全局常量的校验，后续在 get_dataset 内可根据参数灵活设置


# ---------------------------------------------------------
# 2. 内部逻辑
# ---------------------------------------------------------

def _read_raw_folder(folder_path):
    print(f"   [IO]正在扫描文件夹: {folder_path} ...")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"错误: 在 {folder_path} 下未找到任何 .csv 文件")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, dtype=object)
            df_list.append(df)
        except Exception as e:
            print(f"   警告: 跳过文件 {f} ({e})")

    if not df_list: raise ValueError("没有读取到数据")

    data = pd.concat(df_list, ignore_index=True, sort=False)
    data = data.rename(columns=lambda x: str(x).strip())
    data = data.apply(pd.to_numeric, errors='ignore')

    if data.columns[-1] in data.columns:
        label_col = data.columns[-1]
        data = data[data[label_col] != "not specified"]

    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return data


def _l2_normalize(df):
    arr = df.to_numpy().astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return pd.DataFrame(arr / norms, columns=df.columns, index=df.index)


def _preprocess_and_save(raw_folder_path, save_path, label_map):
    df = _read_raw_folder(raw_folder_path)

    features = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    labels = df.iloc[:, -1]

    if features.isnull().sum().sum() > 0:
        features.fillna(features.mean(), inplace=True)

    print("   [处理] 执行 L2 归一化...")
    features = _l2_normalize(features)

    print(f"   [处理] 标签编码 (Map size: {len(label_map)})...")
    labels_encoded = labels.map(lambda x: label_map.get(str(x), -1))

    valid_mask = (labels_encoded != -1)
    if (~valid_mask).sum() > 0:
        print(f"   [处理] 剔除 {(~valid_mask).sum()} 行无效标签数据")
        features = features[valid_mask]
        labels_encoded = labels_encoded[valid_mask]

    full_df = pd.concat([features, labels_encoded.rename('label')], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_df.to_csv(save_path, index=False)
    print(f"   [完成] 已保存至: {save_path}")


# ---------------------------------------------------------
# 3. 对外接口
# ---------------------------------------------------------

# [修改] 添加了 feature_dim 默认参数，不再强制约束为57
def get_dataset(raw_path, processed_path, dataset_name='wut', force_update=False, feature_dim=57):
    """
    Args:
        raw_path: 原始数据文件夹路径
        processed_path: 缓存文件路径 (.csv)
        dataset_name: 'wut' 或 'pku'
        force_update: 是否强制重新生成缓存
        feature_dim: 特征的预期维度
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    target_map = DATASET_CONFIGS[dataset_name]

    print(f"\n>>> 正在加载数据集 [{dataset_name.upper()}]")

    if force_update or not os.path.exists(processed_path):
        print(f"   (需要重新生成缓存)")
        _preprocess_and_save(raw_path, processed_path, target_map)
    else:
        print(f"   (读取现有缓存: {os.path.basename(processed_path)})")

    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        raise IOError(f"读取 CSV 失败: {processed_path}. 错误: {e}")

    X_part = df.iloc[:, :-1]
    y_part = df.iloc[:, -1]

    if y_part.dtype == 'object' or isinstance(y_part.iloc[0], str):
        print(f"   [警告] 缓存文件中的标签是字符串 (例如 '{y_part.iloc[0]}')，正在进行实时映射...")
        y_numeric = y_part.map(lambda x: target_map.get(str(x), -1))
        if (y_numeric == -1).all():
            raise ValueError(f"所有标签映射失败！请检查 dataset_name='{dataset_name}' 是否正确，或者 CSV 里的标签是否在 MAP 中。")
        y_part = y_numeric

    try:
        X_np = X_part.values.astype(np.float32)
        y_np = y_part.values.astype(np.float32)
    except ValueError as e:
        print(f"   [致命错误] 无法将数据转换为浮点数: {e}")
        print("   检查是否有非数字字符混入特征列...")
        raise e

    # [修改] 维度修正逻辑：利用传进来的 feature_dim 而非全局常量
    if X_np.shape[1] != feature_dim:
        try:
            X_np = X_np.reshape(-1, feature_dim)
        except ValueError:
            pass

    print(f"   加载完毕: X={X_np.shape}, y={y_np.shape}")
    return torch.from_numpy(X_np), torch.from_numpy(y_np)


