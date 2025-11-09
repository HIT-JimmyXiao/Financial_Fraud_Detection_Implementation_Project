"""
深度数据清理脚本
- 读取 13-preprocessed.csv
- 缺失值分层处理（删除/KNN/中位数）
- Indcd类别变量KNN填充（编码→填充→解码）
- GPU加速KNN填充（如果cuML可用）
- 方差过滤
- VIF共线性过滤
- 保存清理后的数据到 13-preprocessed_final.csv

缺失值处理策略（参考model/data_preprocessing.py）：
  1. 删除：缺失率 > 50%
  2. KNN填充：缺失率 < 30%（k=5）
     - 支持GPU加速（cuML，如果可用）
     - Indcd类别变量：编码→KNN填充→解码
  3. 中位数填充：缺失率 30%-50%
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.impute import KNNImputer, SimpleImputer
import multiprocessing
import time
import logging
warnings.filterwarnings('ignore')

# 抑制 joblib/loky 的调试输出（Windows 上检测 CPU 核心数时的调用栈信息）
logging.getLogger('joblib').setLevel(logging.WARNING)
logging.getLogger('loky').setLevel(logging.WARNING)

# 尝试导入tqdm显示进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm未安装，将使用简单进度显示。建议安装: pip install tqdm")

# 并行处理配置
N_JOBS = -1  # -1表示使用所有CPU核心，也可以指定数字如4
try:
    CPU_COUNT = multiprocessing.cpu_count()
    print(f"[INFO] 检测到CPU核心数: {CPU_COUNT}")
except:
    CPU_COUNT = 4
    print(f"[INFO] 无法检测CPU核心数，默认使用: {CPU_COUNT}")

print("=" * 80)
print("财务舞弊识别 - 深度数据清理")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 获取脚本所在目录（Insight_output目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 切换到脚本所在目录
print(f"[INFO] 工作目录: {os.getcwd()}")

# 1. 加载数据
print("\n" + "=" * 80)
print("步骤1: 加载数据")
print("=" * 80)

input_file = os.path.join(script_dir, '13-preprocessed.csv')
output_file = os.path.join(script_dir, '13-preprocessed_final.csv')

df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"[OK] 数据加载成功:")
print(f"  原始数据形状: {df.shape}")
print(f"  列名: {df.columns.tolist()[:10]}... (共{len(df.columns)}列)")

# 2. 分离特征和标签
print("\n" + "=" * 80)
print("步骤2: 分离特征和标签")
print("=" * 80)

# 主键列（Indcd需要填充，不放在主键列中）
id_cols = ['Stkcd', 'Accper', 'Typrep']
# 标签列
label_cols = ['isviolation', 'isST']
# 特征列（所有F开头的列 + Indcd）
feature_cols = [col for col in df.columns if col.startswith('F')]
# Indcd需要KNN填充，加入特征列
if 'Indcd' in df.columns:
    feature_cols.append('Indcd')

print(f"[INFO] 主键列: {id_cols}")
print(f"[INFO] 标签列: {label_cols}")
print(f"[INFO] 特征列数量: {len(feature_cols)}（包含Indcd）")
print(f"[INFO] 特征列: {feature_cols[:10]}... (共{len(feature_cols)}列)")

# 分离数据
ids = df[id_cols].copy()
labels = df[label_cols].copy()
features = df[feature_cols].copy()

print(f"[OK] 数据分离完成:")
print(f"  主键: {ids.shape}")
print(f"  标签: {labels.shape}")
print(f"  特征: {features.shape}")

# 3. 缺失值统计和处理（分层策略：删除/KNN/中位数）
print("\n" + "=" * 80)
print("步骤3: 缺失值分析与处理")
print("=" * 80)

missing_pct = (features.isnull().sum() / len(features) * 100).sort_values(ascending=False)
print(f"[INFO] 缺失率前10的特征:")
for col, pct in missing_pct.head(10).items():
    print(f"  {col}: {pct:.2f}%")

# 缺失值处理配置（参考model/config.py）
DELETE_THRESHOLD = 0.5  # 缺失率>50%的列删除
KNN_THRESHOLD = 0.3     # 缺失率<30%的列使用KNN
KNN_NEIGHBORS = 5       # KNN邻居数

# 1. 删除缺失率过高的列（>50%）
high_missing_cols = missing_pct[missing_pct > DELETE_THRESHOLD * 100].index.tolist()
if high_missing_cols:
    print(f"\n[步骤3.1] 删除缺失率>{DELETE_THRESHOLD*100}%的列: {len(high_missing_cols)}个")
    for col in high_missing_cols:
        print(f"  {col}: {missing_pct[col]:.2f}%")
    features = features.drop(columns=high_missing_cols)
    print(f"[OK] 删除后特征数: {len(features.columns)}")
else:
    print(f"\n[步骤3.1] 没有缺失率>{DELETE_THRESHOLD*100}%的列")

# 分离Indcd，单独处理（类别变量，使用mode策略）
indcd_col = None
indcd_original = None
if 'Indcd' in features.columns:
    indcd_col = 'Indcd'
    indcd_original = features['Indcd'].copy()
    # 从features中移除Indcd，单独处理
    features = features.drop(columns=['Indcd'])

# 重新计算缺失率（Indcd已移除）
missing_pct = (features.isnull().sum() / len(features) * 100)

# 2. 对缺失率较低的列使用KNN填充（<30%），不包括Indcd
low_missing_cols = missing_pct[
    (missing_pct > 0) & 
    (missing_pct <= KNN_THRESHOLD * 100)
].index.tolist()

# 初始化GPU标志
USE_GPU_KNN = False

if low_missing_cols:
    print(f"\n[步骤3.2] 使用KNN填充缺失率<{KNN_THRESHOLD*100}%的列: {len(low_missing_cols)}个")
    print(f"  KNN参数: n_neighbors={KNN_NEIGHBORS}")
    
    # 尝试使用GPU加速的KNN（cuML）
    # 注意：cuML主要支持Linux，Windows上安装困难，建议使用WSL2或Docker
    try:
        import platform
        if platform.system() == 'Windows':
            # Windows上cuML通常不可用，直接使用CPU版本
            raise ImportError("cuML在Windows上不可用")
        from cuml.impute import KNNImputer as cuKNNImputer
        import cudf
        USE_GPU_KNN = True
        print(f"  [OK] 检测到cuML GPU支持，将使用GPU加速KNN填充")
    except ImportError:
        print(f"  [INFO] cuML未安装或不可用，使用sklearn KNNImputer（CPU）")
        print(f"  [INFO] 注意：cuML主要支持Linux系统，Windows上建议使用WSL2或Docker")
        print(f"  [INFO] Linux安装: conda install -c rapidsai -c conda-forge -c nvidia cuml")
        from sklearn.impute import KNNImputer
    
    if USE_GPU_KNN:
        # 使用cuML GPU加速
        print(f"  [INFO] 使用GPU加速KNN填充...")
        try:
            # 转换为cudf DataFrame
            features_gpu = cudf.DataFrame(features[low_missing_cols])
            knn_imputer = cuKNNImputer(n_neighbors=KNN_NEIGHBORS)
            features_gpu_filled = knn_imputer.fit_transform(features_gpu)
            # 转换回pandas DataFrame
            features[low_missing_cols] = features_gpu_filled.to_pandas()
            print(f"  [OK] GPU加速KNN填充完成")
        except Exception as e:
            print(f"  [WARN] GPU加速失败: {e}，回退到CPU")
            USE_GPU_KNN = False
    
    if not USE_GPU_KNN:
        # 使用sklearn CPU版本（分批处理以显示进度）
        # 注意：sklearn 1.6.1的KNNImputer不支持n_jobs参数，使用默认单核
        from sklearn.impute import KNNImputer
        
        print(f"  [INFO] 使用sklearn KNNImputer（CPU单核）")
        print(f"  [INFO] 数据量: {len(features)}行 × {len(low_missing_cols)}列")
        
        # 分批处理策略：每批5000行，以显示进度
        batch_size = 5000
        n_batches = (len(features) + batch_size - 1) // batch_size
        print(f"  [INFO] 采用分批处理策略，每批{batch_size}行，共{n_batches}批")
        
        # 先在所有数据上训练KNN模型（一次性）
        print(f"  [INFO] 训练KNN模型...")
        knn_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        start_time = time.time()
        knn_imputer.fit(features[low_missing_cols])
        fit_time = time.time() - start_time
        print(f"  [OK] KNN模型训练完成，耗时: {fit_time:.1f}秒")
        
        # 分批转换数据
        print(f"  [INFO] 开始分批填充缺失值...")
        filled_data = []
        batch_start_time = time.time()
        
        if HAS_TQDM:
            batch_iterator = tqdm(range(n_batches), desc="  KNN填充", ncols=80)
        else:
            batch_iterator = range(n_batches)
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(features))
            
            # 转换当前批次
            batch_data = features[low_missing_cols].iloc[start_idx:end_idx]
            batch_filled = knn_imputer.transform(batch_data)
            filled_data.append(pd.DataFrame(batch_filled, 
                                           index=batch_data.index, 
                                           columns=low_missing_cols))
            
            # 显示进度（如果没有tqdm）
            if not HAS_TQDM:
                elapsed = time.time() - batch_start_time
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
                progress = 100 * (batch_idx + 1) / n_batches
                print(f"\r  进度: {batch_idx+1}/{n_batches} ({progress:.1f}%), "
                      f"已用: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒", end='', flush=True)
        
        if not HAS_TQDM:
            print()  # 换行
        
        # 合并所有批次
        features[low_missing_cols] = pd.concat(filled_data)
        total_time = time.time() - start_time
        print(f"  [OK] CPU KNN填充完成，总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
else:
    print(f"\n[步骤3.2] 没有缺失率<{KNN_THRESHOLD*100}%的列需要KNN填充")

# 3. 单独处理Indcd：KNN最近5个取众数，如果没有多数就不填
if indcd_col and indcd_original is not None:
    print(f"\n[步骤3.3] 处理Indcd类别变量（KNN mode策略）")
    
    # 检查Indcd缺失情况
    indcd_missing_mask = indcd_original.isnull()
    indcd_missing_count = indcd_missing_mask.sum()
    
    if indcd_missing_count > 0:
        print(f"  Indcd缺失记录数: {indcd_missing_count}")
        
        # 准备用于KNN的特征（不包括Indcd本身）
        # 使用已填充的数值特征
        knn_features = features.copy()
        
        # 找到有Indcd值的记录索引
        has_indcd_mask = indcd_original.notna()
        has_indcd_indices = np.where(has_indcd_mask)[0]
        missing_indcd_indices = np.where(indcd_missing_mask)[0]
        
        if len(has_indcd_indices) > 0 and len(missing_indcd_indices) > 0:
            print(f"  有Indcd值的记录: {len(has_indcd_indices)}")
            print(f"  缺失Indcd的记录: {len(missing_indcd_indices)}")
            
            # 尝试使用GPU加速的KNN查找
            USE_GPU_KNN_SEARCH = False
            try:
                import platform
                if platform.system() == 'Windows':
                    # Windows上cuML通常不可用，直接使用CPU版本
                    raise ImportError("cuML在Windows上不可用")
                from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
                USE_GPU_KNN_SEARCH = True
                print(f"  [OK] 检测到cuML GPU支持，将使用GPU加速KNN查找")
            except ImportError:
                from sklearn.neighbors import NearestNeighbors
                print(f"  [INFO] 使用sklearn NearestNeighbors（CPU多核并行）")
            
            # 准备训练数据（有Indcd值的记录）
            X_train = knn_features.iloc[has_indcd_indices].values
            y_train = indcd_original.iloc[has_indcd_indices].values  # Indcd值
            
            # 准备查询数据（缺失Indcd的记录）
            X_query = knn_features.iloc[missing_indcd_indices].values
            
            # 处理NaN值（用0填充，因为KNN不能处理NaN）
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_query = imputer.transform(X_query)
            
            if USE_GPU_KNN_SEARCH:
                try:
                    # GPU加速KNN查找
                    knn_model = cuNearestNeighbors(n_neighbors=KNN_NEIGHBORS)
                    knn_model.fit(X_train)
                    distances, indices = knn_model.kneighbors(X_query)
                    print(f"  [OK] GPU加速KNN查找完成")
                except Exception as e:
                    print(f"  [WARN] GPU加速失败: {e}，回退到CPU")
                    USE_GPU_KNN_SEARCH = False
            
            if not USE_GPU_KNN_SEARCH:
                # CPU版本（多核并行）
                # 临时抑制 joblib 的输出
                import sys
                from io import StringIO
                old_stderr = sys.stderr
                sys.stderr = StringIO()
                
                try:
                    knn_model = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, n_jobs=N_JOBS)
                    print(f"  [INFO] 使用CPU多核并行（n_jobs={N_JOBS if N_JOBS > 0 else CPU_COUNT}）")
                    print(f"  [INFO] 开始KNN查找，训练集: {len(X_train)}行，查询集: {len(X_query)}行...")
                    start_time = time.time()
                    knn_model.fit(X_train)
                    distances, indices = knn_model.kneighbors(X_query)
                    elapsed_time = time.time() - start_time
                    print(f"  [OK] KNN查找完成，耗时: {elapsed_time:.1f}秒")
                finally:
                    sys.stderr = old_stderr
            
            # 对每个缺失Indcd的记录，找到最近5个邻居的Indcd值，取众数
            print(f"  [INFO] 开始填充Indcd值，共{len(missing_indcd_indices)}条记录...")
            filled_count = 0
            # 使用tqdm显示进度
            if HAS_TQDM:
                iterator = tqdm(enumerate(missing_indcd_indices), total=len(missing_indcd_indices), 
                               desc="  填充Indcd", ncols=80)
            else:
                iterator = enumerate(missing_indcd_indices)
                print(f"  进度: 0/{len(missing_indcd_indices)}", end='', flush=True)
            
            start_time = time.time()
            for i, missing_idx in iterator:
                # 获取最近5个邻居的索引（在has_indcd_indices中的位置）
                neighbor_indices_in_train = indices[i]
                # 获取这些邻居的Indcd值
                neighbor_indcds = y_train[neighbor_indices_in_train]
                # 去除NaN值
                neighbor_indcds = neighbor_indcds[pd.notna(neighbor_indcds)]
                
                if len(neighbor_indcds) > 0:
                    # 计算众数（mode）
                    from collections import Counter
                    indcd_counts = Counter(neighbor_indcds)
                    most_common = indcd_counts.most_common(1)
                    
                    # 如果有众数（出现次数>1，或者只有1个值），则填充
                    if len(most_common) > 0:
                        mode_indcd, mode_count = most_common[0]
                        # 如果众数出现次数>1，或者所有邻居的Indcd都一样，则填充
                        if mode_count > 1 or len(set(neighbor_indcds)) == 1:
                            indcd_original.iloc[missing_idx] = mode_indcd
                            filled_count += 1
                
                # 每100条记录显示一次进度（如果没有tqdm）
                if not HAS_TQDM and (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (len(missing_indcd_indices) - i - 1) / rate if rate > 0 else 0
                    print(f"\r  进度: {i+1}/{len(missing_indcd_indices)} ({100*(i+1)/len(missing_indcd_indices):.1f}%), "
                          f"已用: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒", end='', flush=True)
            
            if not HAS_TQDM:
                print()  # 换行
            elapsed_time = time.time() - start_time
            print(f"  [OK] Indcd KNN mode填充完成: {filled_count}/{len(missing_indcd_indices)}个记录被填充，耗时: {elapsed_time:.1f}秒")
        else:
            print(f"  [WARN] 没有足够的参考数据来填充Indcd")
    
    # 4. 对于每个Stkcd公司，取Indcd里最高的一个（如果都一样就不填）
    print(f"\n[步骤3.4] 按Stkcd公司处理Indcd（取最高值）")
    
    # 合并Stkcd信息（从ids中获取）
    if 'Stkcd' in ids.columns:
        # 创建DataFrame用于处理
        indcd_with_stkcd = pd.DataFrame({
            'Stkcd': ids['Stkcd'].values,
            'Indcd': indcd_original.values
        }, index=indcd_original.index)
        
        # 对每个Stkcd分组处理
        stkcd_filled_count = 0
        unique_stkcds = indcd_with_stkcd['Stkcd'].unique()
        print(f"  [INFO] 处理{len(unique_stkcds)}个公司的Indcd...")
        
        # 使用tqdm显示进度
        if HAS_TQDM:
            stkcd_iterator = tqdm(unique_stkcds, desc="  处理Stkcd", ncols=80)
        else:
            stkcd_iterator = unique_stkcds
            print(f"  进度: 0/{len(unique_stkcds)}", end='', flush=True)
        
        start_time = time.time()
        for i, stkcd in enumerate(stkcd_iterator):
            stkcd_mask = indcd_with_stkcd['Stkcd'] == stkcd
            stkcd_indcds = indcd_with_stkcd.loc[stkcd_mask, 'Indcd']
            
            # 去除NaN值
            non_null_indcds = stkcd_indcds.dropna()
            
            if len(non_null_indcds) > 0:
                # 如果所有Indcd值都一样，就不填（保持原样）
                unique_indcds = non_null_indcds.unique()
                if len(unique_indcds) > 1:
                    # 有不同值，取最高的一个（按字符串排序，取最大值）
                    highest_indcd = sorted(unique_indcds)[-1]
                    # 填充该Stkcd的所有空Indcd
                    null_mask = stkcd_mask & indcd_with_stkcd['Indcd'].isnull()
                    if null_mask.sum() > 0:
                        # 使用索引更新indcd_original
                        null_indices = indcd_with_stkcd.loc[null_mask].index
                        indcd_original.loc[null_indices] = highest_indcd
                        stkcd_filled_count += null_mask.sum()
            
            # 每1000个公司显示一次进度（如果没有tqdm）
            if not HAS_TQDM and (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(unique_stkcds) - i - 1) / rate if rate > 0 else 0
                print(f"\r  进度: {i+1}/{len(unique_stkcds)} ({100*(i+1)/len(unique_stkcds):.1f}%), "
                      f"已用: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒", end='', flush=True)
        
        if not HAS_TQDM:
            print()  # 换行
        elapsed_time = time.time() - start_time
        print(f"  [OK] 按Stkcd填充完成: {stkcd_filled_count}个记录被填充，耗时: {elapsed_time:.1f}秒")
    
    # 5. 确保每个公司只有一个Indcd（取出现次数最多的）
    print(f"\n[步骤3.4.5] 确保每个公司Indcd唯一性（同一公司统一为出现次数最多的Indcd）")
    
    # 检查当前情况
    if 'Stkcd' in ids.columns:
        # 创建DataFrame用于检查
        check_df = pd.DataFrame({
            'Stkcd': ids['Stkcd'].values,
            'Indcd': indcd_original.values
        })
        
        # 检查有多少公司有多个Indcd
        stkcd_indcd_counts = check_df.groupby('Stkcd')['Indcd'].nunique()
        multi_indcd_count = (stkcd_indcd_counts > 1).sum()
        
        if multi_indcd_count > 0:
            print(f"  [INFO] 发现{multi_indcd_count}个公司有多个Indcd类别，需要统一")
            
            # 为每个Stkcd选择出现次数最多的Indcd
            stkcd_to_mode_indcd = {}
            for stkcd in check_df['Stkcd'].unique():
                stkcd_mask = check_df['Stkcd'] == stkcd
                indcd_counts = check_df.loc[stkcd_mask, 'Indcd'].value_counts()
                # 选择出现次数最多的Indcd
                mode_indcd = indcd_counts.index[0]
                stkcd_to_mode_indcd[stkcd] = mode_indcd
            
            # 应用统一Indcd
            original_before_unify = indcd_original.copy()
            indcd_original = ids['Stkcd'].map(stkcd_to_mode_indcd)
            changed_count = (original_before_unify != indcd_original).sum()
            
            print(f"  [OK] Indcd唯一性修正完成，共修改{changed_count}条记录")
            
            # 验证结果
            check_df['Indcd'] = indcd_original.values
            stkcd_indcd_counts_after = check_df.groupby('Stkcd')['Indcd'].nunique()
            multi_indcd_after = (stkcd_indcd_counts_after > 1).sum()
            
            if multi_indcd_after == 0:
                print(f"  [OK] 验证通过：所有公司现在都只有一个Indcd类别")
            else:
                print(f"  [WARN] 仍有{multi_indcd_after}个公司有多个Indcd")
        else:
            print(f"  [OK] 所有公司已经只有一个Indcd类别，无需修正")
    
    # 将处理后的Indcd重新加入features
    features[indcd_col] = indcd_original.values
    
    # 检查Indcd是否还有缺失值，如果有则使用整体众数填充（兜底策略）
    remaining_indcd_missing = features[indcd_col].isnull().sum()
    if remaining_indcd_missing > 0:
        print(f"\n[步骤3.5] Indcd剩余缺失值处理（兜底策略）")
        print(f"  剩余缺失记录数: {remaining_indcd_missing}")
        # 使用整体众数填充
        indcd_mode = features[indcd_col].mode()
        if len(indcd_mode) > 0:
            fill_value = indcd_mode[0]
            features[indcd_col] = features[indcd_col].fillna(fill_value)
            print(f"  [OK] 使用整体众数填充: {fill_value}")
        else:
            # 如果没有众数，使用最常见的值
            fill_value = features[indcd_col].value_counts().index[0]
            features[indcd_col] = features[indcd_col].fillna(fill_value)
            print(f"  [OK] 使用最常见值填充: {fill_value}")

# 6. 对缺失率中等的列使用中位数填充（30%-50%）
# 重新计算缺失率（因为Indcd已重新加入并处理完毕）
missing_pct = (features.isnull().sum() / len(features) * 100)

medium_missing_cols = missing_pct[
    (missing_pct > KNN_THRESHOLD * 100) &
    (missing_pct <= DELETE_THRESHOLD * 100)
].index.tolist()

if medium_missing_cols:
    print(f"\n[步骤3.6] 使用中位数填充缺失率{KNN_THRESHOLD*100}%-{DELETE_THRESHOLD*100}%的列: {len(medium_missing_cols)}个")
    from sklearn.impute import SimpleImputer
    median_imputer = SimpleImputer(strategy='median')
    features[medium_missing_cols] = median_imputer.fit_transform(features[medium_missing_cols])
    print(f"[OK] 中位数填充完成")
else:
    print(f"\n[步骤3.6] 没有缺失率{KNN_THRESHOLD*100}%-{DELETE_THRESHOLD*100}%的列需要中位数填充")

# 确保没有缺失值（添加调试信息）
final_missing = features.isnull().sum().sum()
if final_missing > 0:
    print(f"\n[WARN] 仍有{final_missing}个缺失值未处理！")
    print(f"[INFO] 各列缺失值统计:")
    missing_by_col = features.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    for col, count in missing_by_col.items():
        print(f"  {col}: {count}个缺失值")
    # 强制填充剩余缺失值（兜底）
    print(f"[INFO] 使用兜底策略填充所有剩余缺失值...")
    for col in missing_by_col.index:
        if features[col].dtype == 'object' or col == 'Indcd':
            # 类别列用众数
            mode_val = features[col].mode()
            if len(mode_val) > 0:
                features[col].fillna(mode_val[0], inplace=True)
                print(f"  {col}: 使用众数填充: {mode_val[0]}")
            else:
                features[col].fillna('Unknown', inplace=True)
                print(f"  {col}: 使用默认值 'Unknown' 填充")
        else:
            # 数值列用中位数
            median_val = features[col].median()
            features[col].fillna(median_val, inplace=True)
            print(f"  {col}: 使用中位数填充: {median_val:.4f}")
    final_missing_after = features.isnull().sum().sum()
    print(f"[OK] 兜底策略完成，剩余缺失值: {final_missing_after}")

assert features.isnull().sum().sum() == 0, f"仍有{features.isnull().sum().sum()}个缺失值未处理！"
print(f"\n[OK] 缺失值处理完成，所有缺失值已填充，剩余特征数: {len(features.columns)}")

# 6. 方差过滤（不包括Indcd类别变量）
print("\n" + "=" * 80)
print("步骤4: 方差过滤")
print("=" * 80)

from sklearn.feature_selection import VarianceThreshold

# 分离Indcd（不参与方差过滤）
indcd_for_variance = None
if indcd_col and indcd_col in features.columns:
    indcd_for_variance = features[indcd_col].copy()
    features_for_variance = features.drop(columns=[indcd_col])
else:
    features_for_variance = features.copy()

# 计算方差（只对数值特征）
variances = features_for_variance.var()
print(f"[INFO] 特征方差统计:")
print(f"  方差最小的10个特征:")
for col, var in variances.sort_values().head(10).items():
    print(f"    {col}: {var:.6f}")

# 使用VarianceThreshold删除低方差特征
# 阈值设为0.01（方差小于0.01的特征将被删除）
variance_threshold = 0.01
selector = VarianceThreshold(threshold=variance_threshold)
features_variance_filtered = selector.fit_transform(features_for_variance)

# 获取保留的特征名
selected_features = features_for_variance.columns[selector.get_support()].tolist()
removed_features = [col for col in features_for_variance.columns if col not in selected_features]

# 如果Indcd存在，重新加入
if indcd_for_variance is not None:
    selected_features.append(indcd_col)

print(f"\n[INFO] 方差过滤结果（阈值={variance_threshold}）:")
print(f"  原始特征数: {len(features.columns)}")
print(f"  保留特征数: {len(selected_features)}")
print(f"  删除特征数: {len(removed_features)}")

if removed_features:
    print(f"  删除的低方差特征:")
    for col in removed_features:
        print(f"    {col}: 方差={variances[col]:.6f}")

# 更新features为过滤后的DataFrame
features = features[selected_features].copy()
print(f"[OK] 方差过滤后特征形状: {features.shape}")

# 7. VIF共线性过滤（不包括Indcd类别变量）
print("\n" + "=" * 80)
print("步骤5: VIF共线性过滤")
print("=" * 80)

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # 分离Indcd（不参与VIF过滤）
    indcd_for_vif = None
    if indcd_col and indcd_col in features.columns:
        indcd_for_vif = features[indcd_col].copy()
        features_for_vif = features.drop(columns=[indcd_col])
    else:
        features_for_vif = features.copy()
    
    print(f"[INFO] 开始VIF计算（初始特征数: {len(features_for_vif.columns)}，不包括Indcd）...")
    
    # VIF逐步过滤（阈值设为10）
    vif_threshold = 10
    max_iterations = 50  # 最大迭代次数
    iteration = 0
    
    while iteration < max_iterations:
        # 计算所有特征的VIF（只对数值特征）
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features_for_vif.columns
        
        # 显示VIF计算进度
        if HAS_TQDM:
            vif_iterator = tqdm(range(len(features_for_vif.columns)), 
                                desc=f"  计算VIF (迭代{iteration+1})", ncols=80, leave=False)
        else:
            vif_iterator = range(len(features_for_vif.columns))
            if iteration == 0 or iteration % 5 == 0:
                print(f"  迭代{iteration+1}: 计算{len(features_for_vif.columns)}个特征的VIF...", end='', flush=True)
        
        vif_scores = []
        start_time = time.time()
        for i in vif_iterator:
            vif_score = variance_inflation_factor(features_for_vif.values, i)
            vif_scores.append(vif_score)
        
        if not HAS_TQDM and (iteration == 0 or iteration % 5 == 0):
            elapsed = time.time() - start_time
            print(f" 完成，耗时: {elapsed:.1f}秒")
        
        vif_data["VIF"] = vif_scores
        
        # 找到最大VIF
        max_vif = vif_data["VIF"].max()
        max_vif_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        
        # 如果最大VIF小于阈值，停止迭代
        if max_vif <= vif_threshold:
            print(f"\n[OK] VIF过滤完成（第{iteration+1}次迭代）")
            print(f"  所有特征的VIF均 <= {vif_threshold}")
            break
        
        # 删除最大VIF的特征
        print(f"  迭代{iteration+1}: 删除特征 {max_vif_feature} (VIF={max_vif:.2f})")
        features_for_vif = features_for_vif.drop(columns=[max_vif_feature])
        iteration += 1
        
        # 每10次迭代显示进度
        if iteration % 10 == 0:
            print(f"    进度: 已迭代{iteration}次，剩余特征数: {len(features_for_vif.columns)}")
    
    if iteration >= max_iterations:
        print(f"[WARN] 达到最大迭代次数（{max_iterations}），停止VIF过滤")
    
    # 显示最终VIF结果
    print(f"\n[INFO] 最终VIF统计:")
    final_vif_data = pd.DataFrame()
    final_vif_data["Feature"] = features_for_vif.columns
    final_vif_data["VIF"] = [variance_inflation_factor(features_for_vif.values, i) 
                             for i in range(len(features_for_vif.columns))]
    final_vif_data = final_vif_data.sort_values("VIF", ascending=False)
    
    print(f"  VIF最大的10个特征:")
    for idx, row in final_vif_data.head(10).iterrows():
        print(f"    {row['Feature']}: {row['VIF']:.2f}")
    
    # 如果Indcd存在，重新加入
    if indcd_for_vif is not None:
        features_for_vif[indcd_col] = indcd_for_vif.values
    
    features = features_for_vif.copy()
    print(f"\n[OK] VIF过滤后特征数: {len(features.columns)}（包括Indcd）")
    
except ImportError:
    print("[WARN] statsmodels未安装，跳过VIF过滤")
    print("[INFO] 可以运行: pip install statsmodels")
except Exception as e:
    print(f"[ERROR] VIF计算失败: {e}")
    print("[WARN] 跳过VIF过滤，继续后续步骤")

# 6. 异常值处理（跳过，不进行Winsorize）
print("\n" + "=" * 80)
print("步骤6: 异常值处理")
print("=" * 80)
print("[INFO] 跳过异常值处理，保留原始数据")
print("[INFO] 异常值将在建模阶段根据具体模型需求处理")

# 7. 标准化（可选，这里先不做，留给后续建模）
print("\n" + "=" * 80)
print("步骤7: 数据标准化（跳过，留给建模阶段）")
print("=" * 80)
print("[INFO] 数据标准化将在建模阶段进行")

# 8. 合并数据并保存
print("\n" + "=" * 80)
print("步骤8: 合并数据并保存")
print("=" * 80)

# 从features中提取Indcd（如果存在）
if 'Indcd' in features.columns:
    ids['Indcd'] = features['Indcd']
    features = features.drop(columns=['Indcd'])

# 合并主键、标签和清理后的特征
df_final = pd.concat([ids, labels, features], axis=1)

print(f"[INFO] 最终数据形状: {df_final.shape}")
print(f"  主键列: {len(id_cols) + (1 if 'Indcd' in ids.columns else 0)}（包含Indcd）")
print(f"  标签列: {len(label_cols)}")
print(f"  特征列: {len(features.columns)}")
print(f"  总列数: {len(df_final.columns)}")

# 保存到CSV
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] 数据已保存: {output_file}")

# 9. 生成清理报告
print("\n" + "=" * 80)
print("步骤9: 生成清理报告")
print("=" * 80)

report_file = os.path.join(script_dir, 'deep-cleaning-report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("财务舞弊识别 - 深度数据清理报告\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("数据清理流程\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"1. 原始数据:\n")
    f.write(f"   文件: {input_file}\n")
    f.write(f"   形状: {df.shape}\n")
    f.write(f"   原始特征数: {len(feature_cols)}\n\n")
    
    f.write(f"2. 缺失值处理（分层策略）:\n")
    f.write(f"   配置参数:\n")
    f.write(f"     - 删除阈值: >{DELETE_THRESHOLD*100}%\n")
    f.write(f"     - KNN阈值: <{KNN_THRESHOLD*100}%（邻居数={KNN_NEIGHBORS}）\n")
    f.write(f"     - 中位数填充: {KNN_THRESHOLD*100}%-{DELETE_THRESHOLD*100}%\n")
    f.write(f"   删除高缺失率列（>{DELETE_THRESHOLD*100}%）: {len(high_missing_cols)}个\n")
    if high_missing_cols:
        for col in high_missing_cols:
            f.write(f"     - {col}: 缺失率未记录（已删除）\n")
    f.write(f"   KNN填充列数: {len(low_missing_cols)}个\n")
    if low_missing_cols:
        f.write(f"     - 列: {', '.join(low_missing_cols)}\n")
    f.write(f"   Indcd处理: 单独使用KNN mode策略（最近5个取众数，无众数不填）+ 按Stkcd取最高值 + 确保每个公司只有一个Indcd\n")
    # 记录GPU使用情况
    if 'USE_GPU_KNN' in locals() or 'USE_GPU_KNN_SEARCH' in locals():
        gpu_used = USE_GPU_KNN if 'USE_GPU_KNN' in locals() else (USE_GPU_KNN_SEARCH if 'USE_GPU_KNN_SEARCH' in locals() else False)
        f.write(f"   GPU加速: {'是（cuML）' if gpu_used else '否（sklearn CPU）'}\n")
    f.write(f"   中位数填充列数: {len(medium_missing_cols)}个\n")
    if medium_missing_cols:
        f.write(f"     - 列: {', '.join(medium_missing_cols)}\n")
    f.write(f"   最终缺失值: 0\n\n")
    
    f.write(f"3. 方差过滤:\n")
    f.write(f"   阈值: {variance_threshold}\n")
    f.write(f"   删除特征数: {len(removed_features)}\n")
    if removed_features:
        for col in removed_features:
            f.write(f"     - {col}: 方差={variances[col]:.6f}\n")
    f.write(f"   保留特征数: {len(selected_features)}\n\n")
    
    f.write(f"4. VIF共线性过滤:\n")
    f.write(f"   阈值: {vif_threshold}\n")
    f.write(f"   迭代次数: {iteration}\n")
    f.write(f"   最终特征数: {len(features.columns)}\n\n")
    
    f.write(f"5. 异常值处理:\n")
    f.write(f"   方法: 跳过（保留原始数据）\n")
    f.write(f"   说明: 异常值将在建模阶段根据具体模型需求处理\n\n")
    
    f.write(f"6. 最终数据:\n")
    f.write(f"   文件: {output_file}\n")
    f.write(f"   形状: {df_final.shape}\n")
    f.write(f"   主键列: {id_cols}\n")
    f.write(f"   标签列: {label_cols}\n")
    f.write(f"   特征列数: {len(features.columns)}\n")
    f.write(f"   保留的特征列:\n")
    for col in features.columns:
        f.write(f"     - {col}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("数据质量统计\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"1. 标签分布:\n")
    f.write(f"   违规样本: {labels['isviolation'].sum()} ({labels['isviolation'].mean():.2%})\n")
    f.write(f"   ST样本: {labels['isST'].sum()} ({labels['isST'].mean():.2%})\n\n")
    
    f.write(f"2. 特征统计:\n")
    f.write(f"   特征数: {len(features.columns)}\n")
    f.write(f"   样本数: {len(features)}\n")
    f.write(f"   缺失值总数: {features.isnull().sum().sum()}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("清理完成\n")
    f.write("=" * 80 + "\n")

print(f"[OK] 清理报告已保存: {report_file}")

# 10. 输出摘要
print("\n" + "=" * 80)
print("[OK] 深度数据清理完成")
print("=" * 80)
print(f"")
print(f"数据清理摘要:")
print(f"  原始特征数: {len(feature_cols)}")
print(f"  最终特征数: {len(features.columns)}")
print(f"  删除特征数: {len(feature_cols) - len(features.columns)}")
print(f"  特征保留率: {len(features.columns)/len(feature_cols)*100:.2f}%")
print(f"")
print(f"文件保存位置:")
print(f"  清理后数据: {output_file}")
print(f"  清理报告: {report_file}")
print(f"")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

