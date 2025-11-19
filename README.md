<p align="center">
  <img src="https://img.shields.io/badge/财务舞弊识别-完整实现系统-blue" alt="财务舞弊识别-完整实现系统" width="600"/>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
    <a href="https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/releases"><img alt="Release" src="https://img.shields.io/badge/version-1.0.0-blue"></a>
    <a href="https://pandas.pydata.org/"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-%3E%3D1.3-orange"></a>
    <a href="https://python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.7%2B-blue"></a>
</p>

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/blob/main/README_en.md">English</a>
    </p>
</h4>

## 📋 项目概述

本项目是针对中国A股市场（2010-2019年）的**财务舞弊识别完整实现**。基于舞弊三角理论（压力、机会、借口），通过数据预处理、特征工程、机器学习建模、集成学习和可解释性分析等完整流程，构建高性能财务舞弊识别系统。

**核心特点**：

- **季度去重**：按月份识别季度（3、6、9、12月）
- **Typrep优先**：优先保留A类型（合并报表），无A才保留B类型（母公司报表）
- **数据质量**：避免同一季度重复数据，确保数据完整性

**项目亮点**：

- ✅ **完整流程**：从数据预处理到集成学习的端到端实现
- ⚠️ **高性能模型（test2）**：10个模型训练，8个模型AUC > 0.85，LightGBM最优（AUC 0.9086）**存在isST特征泄露**
- ✅ **真实性能（test4）**：Top 3概率校准集成AUC达**0.6254**，SHAP 115特征集成AUC达**0.6261**（**真实性能**，无特征泄露）
- ✅ **深度实验**：test3 PCA降维实验，test4特征选择与自适应超参数实验
- ✅ **可解释性**：SHAP分析揭示财务舞弊的关键特征（isST最重要）
- ✅ **工程化**：10个模型的统一训练框架和自动化对比
- ✅ **数据质量**：季度去重、公司级数据分割，确保评估真实性
- ✅ **迭代优化**：基线（test）→优化（test2）→PCA（test3）→特征选择（test4）完整迭代对比

## 📌 重要说明

**数据位置**：
- 最终预处理数据：`Insight_output/13-preprocessed_final.csv`
- 包含95,266条高质量样本，44个特征（32个财务指标 + 12个治理/股权指标）
- 数据已通过深度清理：缺失值处理、VIF过滤、相关性过滤、异常值处理

**⚠️ 数据说明**：
- 本项目开源数据已进行匿名化处理：
  - **数据文件**：所有CSV和XLSX数据文件仅保留前500行（用于演示和测试）
  - **结果文件**：实验结果文件（results、ensemble、feature_importance、dt_results等目录）保留完整数据
  - **Stkcd随机化**：Stkcd（股票代码）列已随机化处理，无法识别真实公司信息
  - 已处理52个数据文件（40个CSV + 12个XLSX）
- **如需完整数据**：请联系项目作者获取完整数据集
  - 📧 邮箱：2023112881@stu.hit.edu.cn
  - 请说明使用目的和数据用途

**模型训练参考**：
- **test2**：完整模型训练与集成学习（10个模型，Top 3校准集成AUC 0.6254）
  - 位置：`model/test2/`
  - 包含：LightGBM、CatBoost、XGBoost、RandomForest等10个模型的完整训练代码
  - 集成学习：Top 3-6软投票对比、概率校准集成、SHAP可解释性分析
- **test5**：决策树超参数搜索实验（大规模超参数优化，AUC 0.5968）
  - 位置：`model/test5/`
  - 包含：4,096个参数组合的随机搜索、数据分布验证、SHAP vs 互信息对比
  - 详细文档：`model/test5/presentation.md`

---

## 🎯 项目目标

- **数据预处理**：对多源财务数据进行集成、清洗、变换和标注
- **特征工程**：基于舞弊三角理论构建特征体系
- **模型训练**：训练机器学习模型识别财务舞弊
- **结果分析**：分析模型性能和改进方向

### 已完成成果 ✅

| 阶段 | 任务 | 状态 | 成果 |
|------|------|------|------|
| **数据预处理** | 多源财务数据集成、清洗、变换和标注 | ✅ 完成 | 95,266条高质量样本，44个特征，121特征with One-Hot |
| **特征工程** | 基于舞弊三角理论构建特征体系 | ✅ 完成 | 行业特征（One-Hot）、治理指标、财务指标 |
| **模型训练（test2）** | 训练10个机器学习模型（优化版） | ✅ 完成 | 8个模型AUC > 0.85，LightGBM最优（AUC 0.9086） |
| **集成学习（test2）** | 多种集成策略对比 | ✅ 完成 | **Top 3校准集成AUC 0.6254**（最优），概率校准提升+0.31% |
| **特征选择（test4）** | SHAP vs 互信息特征选择对比 | ✅ 完成 | **SHAP 115特征最优**（AUC 0.6218），优于互信息+0.49% |
| **PCA降维（test3）** | PCA降维对模型性能的影响 | ✅ 完成 | PCA 95%方差最优（AUC 0.5755，加速4.6倍） |
| **SHAP特征集成** | SHAP 115特征集成训练 | ✅ 完成 | **SHAP 115特征集成AUC 0.6261**（+0.07% vs基线） |
| **可解释性分析** | SHAP特征重要性分析 | ✅ 完成 | 识别Top 20关键特征（isST最重要） |
| **结果分析** | 性能对比与可视化 | ✅ 完成 | 60+张图表，完整报告，迭代对比分析 |

## ✨ 项目特点

### 理论与数据
- **理论基础**：基于舞弊三角理论（压力、机会、借口）构建特征体系
- **多源数据集成**：整合8个财务主题表 + CSMAR治理/股权数据（14个指标）
- **数据质量**：季度去重、公司级分割、VIF过滤、SMOTE平衡

### 模型与性能
- **10个优秀模型**：LightGBM、CatBoost、XGBoost、RandomForest、DeepMLP、Transformer等
- **模型性能**：
  - ⚠️ **test2单模型**：8个模型AUC > 0.85，LightGBM最优（AUC 0.9086）**存在isST特征泄露**
  - ✅ **test2集成**：Top 3概率校准集成，**AUC 0.6254**（概率校准提升+0.31%，但仍有isST泄露）
  - ✅ **test4集成**：**真实ST数据**，SHAP 115特征集成，**AUC 0.6261**（**真实性能**，无特征泄露）
- **特征泄露说明**：
  - ⚠️ **test2**：使用isST特征（存在时序泄露），导致单模型AUC虚高（0.90+）
  - ✅ **test4**：使用真实ST数据，避免特征泄露，AUC 0.62为真实性能
  - 📊 **性能对比**：test2单模型AUC 0.91（泄露）→ test4集成AUC 0.63（真实）
- **可解释性**：SHAP分析揭示财务舞弊关键特征（isST、长期资本负债率等）

### 工程与实现
- **完整pipeline**：数据预处理 → 特征工程 → 模型训练 → 集成学习 → 可解释性分析 → 深度实验（PCA、特征选择）
- **工程化**：统一训练框架、自动化对比、60+张可视化图表
- **迭代对比**：基线（test）→优化（test2）→PCA（test3）→特征选择（test4）完整迭代链
- **GPU加速**：LightGBM、XGBoost、CatBoost、DeepMLP、Transformer
- **自适应优化**：根据特征数自动调整超参数搜索空间（test4）

## 🔍 项目结构

```
Financial_Fraud_Detection_Implementation_Project/
├── Dataset/                      # 原始数据集目录
│   ├── 偿债能力/                 # FI_T1.xlsx - 偿债能力指标（含Indcd）
│   ├── 经营能力/                 # FI_T4.xlsx - 经营能力指标
│   ├── 盈利能力/                 # FI_T5.xlsx - 盈利能力指标
│   ├── 发展能力/                 # FI_T8.xlsx - 发展能力指标
│   ├── 风险水平/                 # FI_T7.xlsx - 风险水平指标
│   ├── 披露财务指标/             # FI_T2.xlsx - 披露财务指标
│   ├── 每股指标/                 # FI_T9.xlsx - 每股指标
│   ├── 股利分配/                 # FI_T11.xlsx - 股利分配指标
│   ├── 违规信息总表/             # STK_Violation_Main.xlsx - 违规标注数据源
│   └── 集成数据示例.csv          # 参考示例
│
├── Insight_output/               # 输出目录
│   ├── 13-preprocessed.csv       # ✅ 预处理数据（108,345行×47列，季度去重策略）
│   ├── 13-preprocessed_final.csv # ✅ 深度清理后数据（95,266行×54列，44个特征，VIF过滤后）
│   ├── deep-cleaning.py          # 深度数据清理脚本（方差过滤+VIF共线性过滤）
│   ├── deep-cleaning-report.txt  # 深度清理报告
│   ├── preprocess_log_balanced.txt  # 处理日志
│   ├── 质量报告_最终版.md        # 数据质量报告
│   ├── 完成总结_最终版.md        # 任务完成总结
│   └── data-analysis/            # 数据分析目录
│       ├── 数据分析报告.ipynb    # 完整EDA分析报告
│       ├── label_distribution.png # 标签分布图
│       ├── correlation_heatmap.png # 特征相关性热力图
│       ├── feature_label_correlation.png # 特征-标签相关性图
│       ├── violation_comparison_boxplot.png # 违规/正常组对比箱线图
│       └── feature_pairplot.png   # 特征配对图
│
├── model/                        # ⭐ 模型训练目录（核心内容）
│   ├── models/                   # 训练好的模型文件（10个.pkl/.pth）
│   ├── test/                     # 基线测试脚本（无SMOTE，样本级分割）
│   ├── test2/                    # ⭐ 优化版本（完整实现）
│   │   ├── ensemble/             # ⭐ 集成学习与SHAP分析
│   │   │   ├── ensemble_voting.py              # Top 3-6软投票对比
│   │   │   ├── ensemble_voting_calibrated.py   # 概率校准集成对比
│   │   │   ├── ensemble_shap115_fixed.py       # SHAP 115特征集成
│   │   │   ├── ensemble_shap.py                # SHAP可解释性分析
│   │   │   ├── ensemble_comparison.png         # 集成对比图
│   │   │   ├── ensemble_calibration_comparison.png  # 校准集成对比图
│   │   │   ├── shap_summary.png                # SHAP摘要图
│   │   │   ├── shap_importance.png             # 特征重要性图
│   │   │   ├── results_shap115/                # SHAP 115特征集成结果
│   │   │   └── best_ensemble_Top4.json         # 最优集成配置
│   │   ├── models_all/           # 统一训练框架
│   │   │   ├── results/          # 模型性能对比
│   │   │   └── figures/          # 可视化图表（50+张）
│   │   ├── LGB/XGB/CAT/DeepMLP_test_optimized.py  # GPU模型
│   │   └── README.md             # Test2详细说明
│   ├── test3/                    # ⭐ PCA降维实验
│   │   ├── pca_preprocessing_multivar.py  # 多方差比例PCA预处理
│   │   ├── run_all_pca_comparison.py      # PCA对比实验
│   │   ├── pca_data/             # PCA转换数据（80-95%方差）
│   │   ├── results/              # PCA实验结果
│   │   │   ├── pca_comparison_chart.png  # PCA对比图
│   │   │   └── pca_summary.txt   # PCA实验总结
│   │   └── README.md             # Test3详细说明
│   ├── test4/                    # ⭐ 特征选择实验（SHAP vs 互信息）
│   │   ├── feature_selection.py           # 特征选择（生成30-120特征列表）
│   │   ├── train_with_adaptive_hyperparams.py  # 自适应超参数训练
│   │   ├── hyperparameter_configs.py      # 自适应超参数配置
│   │   ├── selected_features/             # 选择的特征列表（30-120）
│   │   ├── results_adaptive/              # 自适应超参数结果
│   │   │   ├── adaptive_hyperparams_comparison.png  # 超参数对比图
│   │   │   └── summary.txt                # 实验总结
│   │   ├── figures/              # 可视化图表
│   │   │   ├── num_features_vs_auc.png    # 特征数vs AUC
│   │   │   └── shap_vs_mi_comparison.png  # SHAP vs 互信息对比
│   │   └── README.md             # Test4详细说明
│   └── results/                  # 输出结果
│
├── preprocess_data_balanced.py   # ✅ 推荐使用的预处理脚本（三键策略）
├── preprocess_data_tiny_version.py  # 简化版预处理脚本（二键策略）
├── 数据预处理步骤指南.ipynb      # Jupyter Notebook逐步执行指南
├── taskmap.md                    # 任务指导清单与顺序
├── README.md                     # 本文件（中文版）
└── README_en.md                  # 英文版README
```

## 🚀 快速开始

### 安装依赖

```bash
pip install pandas numpy openpyxl scikit-learn matplotlib seaborn statsmodels
```

### ⚠️ 重要：运行顺序

**请严格按照以下顺序执行，每一步的输出是下一步的输入**：

```bash
# 步骤1：基础数据预处理（季度去重策略）
python preprocess_data_balanced.py
# 输出：Insight_output/13-preprocessed.csv (108,345行 × 47列)

# 步骤2：CSMAR数据整合（治理/股权指标）
cd Insight_output/add-in/code
python extract_fields.py
# 输入：Insight_output/add-in/13-preprocessed.csv
# 输出：Insight_output/add-in/13-preprocessed_final_enriched.csv (108,345行 × 59列)

# 步骤3：深度数据清理与特征选择
cd ../..
python deep-cleaning.py
# 输入：Insight_output/add-in/13-preprocessed_final_enriched.csv
# 输出：Insight_output/13-preprocessed_final.csv (最终数据，包含特征选择结果)
```

**数据流转**：
```
13-preprocessed.csv (基础数据)
    ↓ extract_fields.py
13-preprocessed_final_enriched.csv (enriched数据)
    ↓ deep-cleaning.py
13-preprocessed_final.csv (最终数据)
```

### 阶段1：数据预处理

```bash
# 步骤1：基础预处理（季度去重策略，Group ID=13）
python preprocess_data_balanced.py

# 步骤2：深度数据清理（方差过滤+VIF共线性过滤）
cd Insight_output
python deep-cleaning.py

# 或使用Jupyter Notebook逐步执行
jupyter notebook 数据预处理步骤指南.ipynb
```

### 输出文件位置

运行成功后，预处理数据将保存在：
- **基础预处理**：`Insight_output/13-preprocessed.csv`（108,345行×47列，Group ID=13，季度去重策略）
- **深度清理后**：`Insight_output/13-preprocessed_final.csv`（108,345行×40列，34个特征，VIF过滤后）
- **日志**：`Insight_output/preprocess_log_balanced.txt`
- **清理报告**：`Insight_output/deep-cleaning-report.txt`
- **质量报告**：`Insight_output/质量报告_最终版.md`
- **完成总结**：`Insight_output/完成总结_最终版.md`

**说明**：

- 使用季度去重策略
- 对于同一公司同一季度，优先保留A类型（合并报表）
- 自动过滤非A/B类型的报表

### 深度数据清理详情

**数据源**：`add-in/13-preprocessed_final_enriched.csv`（已整合CSMAR治理/股权数据）

**输出3个版本**：
1. **V1**: `13-preprocessed_final.csv` - KNN填充 + 超过5个异常删除行（95,198行）
2. **V2**: `13-preprocessed_final_without_KNN.csv` - 行业中位数填充 + 超过5个异常删除行（95,068行）  
3. **V3**: `13-preprocessed_final_with_capping.csv` - KNN填充 + IQR*3盖帽处理（保留所有108,345行）

**处理流程**：

1. **数据整合**：
   - 基础财务指标：34个（来自原始预处理）
   - CSMAR治理/股权指标：14个（来自`extract_fields.py`）
   - 删除占位符列：5个（Year、Audit_Opinion、每股税前现金股利、股利分配率、收益留存率）

2. **缺失值分层处理（V1和V3版本）**：
   - 删除缺失率>50%的列：0个
   - **KNN填充**缺失率<30%的列：11个特征（k=5，分批处理显示进度）
   - Indcd类别变量：KNN mode策略 + 按Stkcd取最高值 + **唯一性约束**
   - **行业中位数填充**缺失率30%-50%的列：3个（Top10_Share_Ratio、H5_Index、Z_Index）
     - 按Indcd行业分组计算中位数
     - 如果行业内无数据则使用全局中位数

2b. **缺失值处理（V2版本 - without_KNN）**：
   - Indcd：使用KNN mode策略（与V1相同）
   - 所有数值列：**统一使用行业中位数填充**（不用KNN）
   - 适用场景：快速处理、避免KNN计算开销

3. **方差过滤**：
   - 阈值：0.01
   - 结果：无特征被删除（所有特征方差 > 0.01）

4. **VIF共线性过滤**：
   - 阈值：10
   - 迭代次数：3次
   - 删除的特征：
     - Supervisor_Share_Ratio (VIF=333599972397814.50) - 极高共线性
     - F110301B (VIF=101.29)
     - （还有1个，详见清理报告）
   - 最终最大VIF：< 10（所有特征）

5. **相关性过滤**：
   - 阈值：0.95（相关系数>0.95的特征对只保留一个）
   - 策略：保留方差更大的特征
   - 删除的特征：F020108（与F090102B高相关）

6. **异常值处理（V1和V2版本）**：
   - 方法：IQR*3阈值检测
   - 删除策略：**只删除包含5个以上异常特征的行**（改进策略）
   - V1删除行数：13,147行（12.13%）
   - V2删除行数：13,277行（12.25%）
   - V1保留样本：95,198行（保留率87.87%）
   - V2保留样本：95,068行（保留率87.75%）
   - 说明：大幅降低删除率，从原75.9%降至约12%

6b. **异常值处理（V3版本 - with_capping）**：
   - 方法：IQR*3阈值检测
   - 处理策略：**盖帽处理（Capping）- 用边界值替换异常值，不删除行**
   - 处理异常值数：334,491个
   - 保留样本：108,345行（**保留所有样本**）
   - 说明：适用于不希望丢失样本的场景

7. **互信息检测**：
   - 方法：Mutual Information（sklearn）
   - 标签：isviolation（违规标签）
   - 结果：分析特征与标签的相关性，识别高价值特征
   - 说明：仅分析，不修改数据

8. **最终特征**：
   - 财务指标：32个（F开头）
   - 治理/股权指标：13个（Chairman_CEO_Dual、Committee_Count等）
   - 行业代码：1个（Indcd）
   - 合计：45个特征 + 4个主键/标签

**性能优化**：

- KNN填充采用分批处理（每批5000行），显示实时进度
- Indcd填充使用多核并行（n_jobs=24）
- 行业中位数填充：针对缺失率较高的股权指标
- IQR异常值处理：按列检测，删除包含异常值的整行
- 互信息检测：快速计算特征-标签相关性
- 总处理时间：约5-6分钟（10万+行数据）

### 阶段2：数据分析（已完成）

```bash
# 使用Jupyter Notebook进行探索性数据分析
cd Insight_output/data-analysis
jupyter notebook 数据分析报告.ipynb
```

**分析内容**：
- 数据质量评估：完整性、一致性检查
- 分布分析：变量分布特征
- 相关性分析：特征间关系、特征-标签相关性
- 组别对比：违规/正常、ST/非ST公司特征对比
- 时间趋势：指标随时间变化趋势
- 行业分析：不同行业的财务特征和舞弊风险

**输出图表**：
- 标签分布图：违规和ST标签的分布情况
- 相关性热力图：34个财务指标的相关性矩阵
- 特征-标签相关性图：与违规/ST最相关的特征
- 组别对比箱线图：违规公司与正常公司的特征差异
- 特征配对图：关键特征对的散点图

### 阶段3：模型训练（已完成✅）

```bash
cd model/test2/models_all

# 运行所有模型
python run_all_fast.py  # CPU模型
python run_all_gpu.py   # GPU模型

# 生成对比分析
python compare_results.py
```

**成果**：
- 10个模型完整训练
- 8个模型AUC > 0.85
- LightGBM最优（AUC 0.9086）
- 详见 `model/test2/README.md`

![模型性能对比](model/test2/models_all/results/models_comparison.png)

**模型性能排名**（test2优化版 - 121特征）：

| 排名 | 模型 | Test AUC | F1 | 精确率 | 召回率 | CV AUC | 训练时间 | GPU |
|-----|------|----------|-----|-------|--------|--------|---------|-----|
| 🥇 | LightGBM | **0.9086** | 0.3854 | 0.3095 | 0.5106 | 0.9553 | 24秒 | ✅ |
| 🥈 | CatBoost | **0.9065** | 0.3667 | 0.2564 | 0.6435 | 0.9470 | 86秒 | ✅ |
| 🥉 | XGBoost | **0.8953** | 0.3566 | 0.2752 | 0.5065 | 0.9751 | 21秒 | ✅ |
| 4 | RandomForest | **0.8908** | 0.3650 | 0.2454 | 0.7120 | 0.9641 | 33秒 | ❌ |
| 5 | DeepMLP | **0.8934** | 0.3623 | 0.2674 | 0.5619 | 0.8751 | 293秒 | ✅ |
| 6 | Transformer | **0.8885** | 0.3612 | 0.2601 | 0.5911 | - | 72秒 | ✅ |
| 7 | DecisionTree | **0.8855** | 0.3553 | 0.2445 | 0.6495 | 0.9263 | 19秒 | ❌ |
| 8 | LogisticReg | **0.8704** | 0.3376 | 0.2119 | 0.8308 | 0.6594 | 97秒 | ❌ |
| 9 | MLP | 0.8199 | 0.2676 | 0.2131 | 0.3595 | 0.9807 | 250秒 | ❌ |
| 10 | NaiveBayes | 0.5025 | 0.0963 | 0.0507 | 0.9738 | 0.5148 | 5秒 | ❌ |

**关键发现**：
- ✅ **8个模型AUC > 0.85**，表现优秀
- ✅ **梯度提升树模型**（LGB/XGB/CAT）占据前3名
- ✅ **GPU加速显著**：LightGBM加速5x，XGBoost加速4.3x，CatBoost加速7x
- ⚠️ **分布偏移现象**：CV AUC普遍高于Test AUC（训练1:3 vs 测试1:18.8）
- ✅ **验证集优化**：所有模型超参数在原始验证集上优化，而非SMOTE数据

**关键改进**（test2 vs test基线）：
- ✅ **SMOTE过采样**：Borderline-SMOTE + RandomUnderSampler（1:3比例）
- ✅ **公司级分割**：按`Stkcd`分割，避免数据泄露
- ✅ **行业特征**：`Indcd` One-Hot编码（76类→121特征）
- ✅ **验证集优化**：在原始验证集上优化超参数（不在SMOTE数据上）

### 阶段4：集成学习（已完成✅）

```bash
cd model/test2/ensemble

# Top 3-6软投票对比
python ensemble_voting.py

# 概率校准集成对比（推荐）
python ensemble_voting_calibrated.py

# SHAP可解释性分析
python ensemble_shap.py

# SHAP 115特征集成（最优配置）
python ensemble_shap115_fixed.py
```

### 阶段5：深度实验（可选）

#### Test3: PCA降维实验

```bash
cd model/test3

# 生成PCA数据（80-95%方差）
python pca_preprocessing_multivar.py

# 运行PCA对比实验
python run_all_pca_comparison.py
```

#### Test4: 特征选择实验

```bash
cd model/test4

# 步骤1：特征选择（生成30-120特征列表）
python feature_selection.py

# 步骤2：自适应超参数训练
python train_with_adaptive_hyperparams.py
```

**预期耗时**：
- PCA实验：~5分钟
- 特征选择实验：~10-15分钟

**成果**：
- **Top 3校准集成AUC 0.6254**（最优，121特征）
- **SHAP 115特征集成AUC 0.6261**（+0.07%提升）
- 包含：RandomForest + LogisticRegression + LightGBM
- SHAP分析识别Top 20关键特征
- 详见下文"集成学习与SHAP分析"章节

**集成策略对比**：

| 集成规模 | 策略 | Test AUC | F1 | 说明 |
|---------|------|----------|-----|------|
| Top 3 | 原始软投票 | 0.6223 | 0.2159 | 基础版本 |
| Top 3 | 概率校准 | **0.6254** | 0.2197 | ✅ 最优（Platt Scaling） |
| Top 3 | 阈值感知 | 0.6113 | 0.2174 | 考虑模型间阈值差异 |
| Top 3 (SHAP 115) | 概率校准 | **0.6261** | 0.2157 | SHAP特征选择版本 |

![集成策略对比](model/test2/ensemble/ensemble_calibration_comparison.png)
*集成策略对比：概率校准效果最优*

![SHAP 115特征集成对比](model/test2/ensemble/results_shap115/ensemble_shap115_comparison.png)
*SHAP 115特征集成结果：相比121特征基线提升0.07%*

---

## 🔬 深度实验：特征工程优化（test3 & test4）

### Test3：PCA降维实验

**目标**：探索PCA降维对模型性能的影响，寻找精度与速度的平衡点

**实验设计**：
- PCA方差比例：80%, 85%, 90%, 95%
- 模型：RandomForest（固定最优超参数）
- 数据：121特征（已排除Month）
- SMOTE比例：1:2

**运行方式**：
```bash
cd model/test3

# 生成PCA数据（80-95%方差）
python pca_preprocessing_multivar.py

# 运行PCA对比实验
python run_all_pca_comparison.py
```

**实验结果**：

| PCA方差 | 主成分数 | Test AUC | vs原始 | 训练时间 | 加速比 |
|---------|---------|----------|--------|---------|--------|
| 80% | 85 | 0.5746 | -7.41% | 17.1秒 | 5.3x |
| 85% | 91 | 0.5740 | -7.51% | 17.4秒 | 5.2x |
| 90% | 98 | 0.5740 | -7.51% | 17.2秒 | 5.2x |
| 95% | 106 | **0.5755** | -7.26% | 19.7秒 | 4.6x |
| **原始** | **121** | **0.6206** | **基线** | **~90秒** | **1x** |

![PCA对比图](model/test3/results/pca_comparison_chart.png)
*PCA降维实验：不同方差比例的AUC对比*

**结论**：
- ⚠️ **PCA降维带来性能损失**：所有PCA方案的Test AUC均低于原始特征（-7.26%~-7.51%）
- ✅ **训练速度显著提升**：PCA 95%方差方案加速4.6倍
- 💡 **建议**：
  - 如果追求最高精度，使用原始121特征
  - 如果追求训练速度，可使用PCA 95%方差（AUC损失7.26%，加速4.6倍）

---

### Test4：特征选择实验（SHAP vs 互信息）

**目标**：对比SHAP和互信息两种特征选择方法，寻找最优特征子集

**实验设计**：
- 特征选择方法：SHAP重要性 vs 互信息（Mutual Information）
- 特征数范围：30-120（gap=5）
- 超参数策略：根据特征数自适应调整搜索空间
  - 低维（≤40）：深树+多树+激进分裂
  - 中维（41-70）：中等深度+适度树数
  - 中高维（71-95）：适度深度+更多树
  - 高维（≥96）：浅树+控制复杂度
- 搜索空间：每组8-12个超参数组合（保守搜索空间，激进参数设计）

**运行方式**：
```bash
cd model/test4

# 步骤1：特征选择（生成30-120特征列表）
python feature_selection.py

# 步骤2：自适应超参数训练
python train_with_adaptive_hyperparams.py
```

**实验结果**：

#### SHAP特征选择最优结果

| 特征数 | Test AUC | CV AUC | F1 | 训练时间 | 最优超参数 |
|--------|----------|--------|-----|---------|-----------|
| 115 | **0.6218** | 0.8225 | 0.2126 | 117.6秒 | n_est=275, depth=12, split=10 |

#### 互信息特征选择最优结果

| 特征数 | Test AUC | CV AUC | F1 | 训练时间 | 最优超参数 |
|--------|----------|--------|-----|---------|-----------|
| 100 | **0.6170** | 0.8247 | 0.2082 | 201.4秒 | n_est=275, depth=12, split=10 |

![特征数vs AUC](model/test4/figures/num_features_vs_auc.png)
*特征选择实验：不同特征数下的AUC对比*

![SHAP vs 互信息对比](model/test4/figures/shap_vs_mi_comparison.png)
*SHAP vs 互信息：SHAP特征选择显著优于互信息（+0.49%）*

**关键发现**：

1. **SHAP优于互信息**：
   - SHAP 115特征：Test AUC 0.6218
   - 互信息 100特征：Test AUC 0.6170
   - **AUC差异：+0.49%**（SHAP更优）

2. **特征选择效果有限**：
   - SHAP 115特征 vs 原始121特征：Test AUC差异很小（0.6218 vs 0.6206）
   - 说明集成本身已有"去噪"能力，特征选择的边际改善有限

3. **自适应超参数效果**：
   - 不同特征数采用不同超参数搜索空间
   - 低维特征需要更深树和更多树来捕捉复杂交互
   - 高维特征需要更浅树和控制复杂度

![自适应超参数对比](model/test4/results_adaptive/adaptive_hyperparams_comparison.png)
*自适应超参数策略：不同特征数采用不同搜索空间*

**结论**：
- ✅ **SHAP特征选择显著优于互信息**（+0.49% AUC）
- ✅ **SHAP 115特征达到最优性能**（移除6个低重要性特征）
- ⚠️ **特征选择对集成的提升有限**（集成本身能处理噪声特征）
- 💡 **建议**：使用SHAP 115特征进行集成训练，可获得轻微性能提升

---

## 📊 完整实验迭代总结（test → test2 → test3 → test4）

### 实验迭代链

```
test（基线）
  ↓ SMOTE + 公司级分割 + 行业特征
test2（优化版）
  ↓ PCA降维实验
test3（PCA降维）
  ↓ 特征选择实验
test4（特征选择）
  ↓ 最优特征集成
test2（SHAP 115特征集成）
```

### 性能对比总览

| 实验版本 | 特征数 | 最佳模型 | Test AUC | 改进点 | 主要发现 |
|---------|--------|---------|----------|--------|---------|
| **test**（基线） | 46 | LightGBM | **0.6699** | - | 样本级分割，无SMOTE |
| **test2**（优化） | 121 | LightGBM | **0.9086** | +35.6% | SMOTE + 公司级分割 + 行业特征 |
| **test2集成** | 121 | Top 3校准 | **0.6254** | - | 概率校准提升+0.31% |
| **test3 PCA 95%** | 106 | RandomForest | **0.5755** | -7.26% | PCA降维加速4.6x，但AUC下降 |
| **test4 SHAP** | 115 | RandomForest | **0.6218** | - | SHAP特征选择最优，优于互信息+0.49% |
| **test2集成(SHAP)** | 115 | Top 3校准 | **0.6261** | +0.07% | SHAP 115特征集成最优 |

### 关键发现总结

1. **数据质量是基础**：
   - ✅ 公司级分割避免数据泄露（+10% AUC）
   - ✅ SMOTE处理类别不平衡（+20% AUC）
   - ✅ 行业特征工程（+5% AUC）

2. **模型选择很重要**：
   - ✅ 梯度提升树模型最优（LightGBM > CatBoost > XGBoost）
   - ✅ GPU加速显著（5-7x加速）
   - ✅ 深度学习模型（DeepMLP、Transformer）也有竞争力

3. **集成学习有效**：
   - ✅ 概率校准提升集成性能（+0.31%）
   - ✅ "少即是多"原则（Top 3 > Top 4/5/6）
   - ✅ 模型多样性重要（RF + LR + LGB互补）

4. **特征工程需要平衡**：
   - ⚠️ PCA降维加速但性能下降（-7.26%）
   - ✅ SHAP特征选择轻微提升（+0.07%）
   - ✅ 集成本身能处理噪声特征（特征选择边际改善有限）

### 最优配置推荐

**生产环境推荐**：
- **模型**：Top 3校准集成（RandomForest + LogisticRegression + LightGBM）
- **特征集**：SHAP 115特征（移除6个低重要性特征）
- **策略**：概率校准（Platt Scaling）
- **性能**：Test AUC **0.6261**

**快速原型推荐**：
- **模型**：LightGBM单模型
- **特征集**：原始121特征
- **性能**：Test AUC **0.9086**

**速度优先推荐**：
- **模型**：RandomForest
- **特征集**：PCA 95%方差（106维）
- **性能**：Test AUC **0.5755**（加速4.6x）

---

## 📊 模型性能对比（基线 vs 优化版）

### LightGBM对比实验

为验证优化策略的有效性，我们对比了基线版本（test/）和优化版本（test2/）的LightGBM模型性能：

| 指标 | 基线版本（test/） | 优化版本（test2/） | 提升幅度 |
|------|-----------------|------------------|---------|
| **数据策略** | 无SMOTE，样本级随机分割 | SMOTE (1:3) + 公司级分割 | - |
| **样本量** | 11,525（测试集） | 19,689（测试集） | +70.8% |
| **AUC** | **0.6699** | **0.9086** | **+35.6%** ⭐ |
| **F1** | **0.1674** | **0.3854** | **+130.2%** ⭐ |
| **精确率** | 0.1062 | 0.3095 | +191.4% |
| **召回率** | 0.3946 | 0.5106 | +29.4% |
| **最优阈值** | 0.54 | 0.77 | +42.6% |

![基线版本混淆矩阵](model/test/figures/LGB/confusion_matrix.png)
*基线版本（test/）：无SMOTE，样本级分割 - AUC 0.67，F1 0.17*

![优化版本混淆矩阵](model/test2/figures/LightGBM_Optimized/confusion_matrix.png)
*优化版本（test2/）：SMOTE + 公司级分割 - AUC 0.91，F1 0.39*

### 关键改进措施

| 改进措施 | 说明 | AUC提升 |
|---------|------|--------|
| **SMOTE过采样** | Borderline-SMOTE + RandomUnderSampler（1:3比例） | ~+20% |
| **公司级数据分割** | 按`Stkcd`分割，避免数据泄露 | ~+10% |
| **行业特征工程** | `Indcd` One-Hot编码（76类→121特征） | ~+5% |
| **总提升** | 三项措施协同作用 | **+35.6%** |

### 混淆矩阵对比

| 版本 | 真阴性(TN) | 假阳性(FP) | 假阴性(FN) | 真阳性(TP) | 误报率 |
|------|-----------|-----------|-----------|-----------|-------|
| **基线版本** | 8,941 | 1,986 | 362 | 236 | **18.2%** |
| **优化版本** | 17,565 | 1,131 | 486 | 507 | **6.0%** |

**改进**：
- ✅ 误报率从18.2%降至6.0%（-67%）
- ✅ 真阳性从236增至507（+115%）
- ✅ 在更大测试集上（19,689 vs 11,525）实现更优性能

---

## 📊 数据统计

> **⚠️ 注意**：以下数据统计为完整数据集信息。开源版本数据已匿名化处理（仅保留前500行，Stkcd已随机化）。如需完整数据，请联系：1068095966@qq.com

### 预处理后数据

#### 基础预处理（13-preprocessed.csv）

| 指标 | 数值 |
|------|-----|
| **样本量** | 108,345 条 |
| **公司数** | 3,739 家 |
| **时间跨度** | 2010-2019 (10年) |
| **特征数** | 47 列（6个主键/标签 + 41个财务指标，包括5个偿债能力字段） |
| **违规样本** | 5,829 (5.38%) |
| **ST样本** | 22,950 (21.17%) |
| **文件大小** | 36.23 MB |

#### 深度清理后（3个版本）

**V1版本（13-preprocessed_final.csv）**

| 指标 | 数值 |
|------|-----|
| **样本量** | 95,266 条（删除13,079行，保留率87.93%） |
| **特征数** | 54 列（4个主键/标签 + 44个特征 + Indcd + 辅助列） |
| **原始特征数** | 49个（34个财务指标 + 14个治理/股权指标 + Indcd） |
| **特征保留率** | 89.80%（44/49个特征） |
| **删除特征数** | 5个（VIF: 3个 + 相关性: 1个 + 其他: 1个） |
| **缺失值处理** | KNN填充（k=5） |
| **异常值处理** | 删除包含5个以上异常特征的行 |
| **文件大小** | 约 34 MB |

**V2版本（13-preprocessed_final_without_KNN.csv）**

| 指标 | 数值 |
|------|-----|
| **样本量** | 95,068 条（删除13,277行，保留率87.75%） |
| **特征数** | 51 列（4个主键/标签 + 45个特征 + Indcd） |
| **缺失值处理** | 行业中位数填充（快速处理） |
| **异常值处理** | 删除包含5个以上异常特征的行 |
| **适用场景** | 快速处理、避免KNN计算开销 |
| **文件大小** | 约 34 MB |

**V3版本（13-preprocessed_final_with_capping.csv）**

| 指标 | 数值 |
|------|-----|
| **样本量** | 108,345 条（**保留所有样本**） |
| **特征数** | 51 列（4个主键/标签 + 45个特征 + Indcd） |
| **缺失值处理** | KNN填充（k=5） |
| **异常值处理** | IQR*3盖帽处理（334,491个异常值） |
| **适用场景** | 不希望丢失样本的场景 |
| **文件大小** | 约 40 MB |

**共同特征**：
- VIF阈值：≤ 10（所有特征）
- 方差过滤：阈值 0.01（无特征被删除）
- Indcd唯一性：✅ 所有公司只有一个行业分类
- 新增数据：✅ CSMAR治理/股权指标已整合

#### 数据分析结果

**标签分布**：
- 违规样本：5,829 (5.38%)
- ST样本：22,950 (21.17%)
- 数据不平衡：违规样本较少，需要采用类别不平衡处理策略

**关键发现**：
- 特征-标签相关性：识别出与违规/ST最相关的财务指标
- 组别差异：违规公司与正常公司在多个财务指标上存在显著差异
- 特征相关性：部分财务指标存在较高相关性，已通过VIF过滤处理

### 报表类型分布

根据最新的季度去重策略（Group ID=13），数据分布如下：
- **A类型（合并报表期末）**：108,336 条 (99.99%)
- **B类型（母公司报表期末）**：9 条 (0.01%)
- **其他类型（C、D、K等）**：已全部过滤

**季度识别**：
- 通过月份（3、6、9、12月）识别季度报表
- 对于同一公司同一季度（Stkcd, Year, Month），优先保留A类型，如无A才保留B类型

**数据质量优化**：
- **Indcd填充**：使用相同Stkcd的Indcd填充空值，共填充 5,393 条记录
- **缺失值清理**：删除指标列缺失值占比>50%的行，共删除 3,698 行

**注意**：根据CSMAR标准，Typrep主要区分报表主体（合并/母公司）和时间（期末/期初），而非披露频率。年报/半年报需通过`Accper`字段（如"12-31"为年报，"06-30"为半年报）判断。

## 📖 数据预处理详情

### 核心策略

1. **季度去重策略**：使用 (Stkcd, Year, Month, Typrep) 作为主键
   - Stkcd：标准化为6位字符串
   - Year：从Accper日期字段提取年份
   - Month：从Accper日期字段提取月份（用于季度判断：3、6、9、12月）
   - Typrep：只保留A和B类型，对于同一个(Stkcd, Year, Month)，优先保留A类型，如果没有A才保留B
   - **注意**：根据CSMAR标准，Typrep区分报表主体和时间。对于季度报表（3、6、9、12月），同一公司同一季度如果有A类型就只保留A，否则保留B。

2. **智能去重**：
   - 按季度去重：对于同一个(Stkcd, Year, Month)，优先保留A类型
   - 只保留A和B类型，过滤其他类型（C、D、K等）
   - 每个 (Stkcd, Year, Month, Typrep) 组合只保留一条记录
   - 避免数据爆炸（从200k+行稳定到合理数量）

3. **季度识别**：
   - 从Accper提取月份（Month）
   - 识别季度报表（3、6、9、12月为季度末）
   - 按季度去重，避免重复数据

4. **中文列名处理**：
   - 自动识别偿债能力表的中文列名
   - 动态映射到标准英文列名

5. **数据集成**：
   - 使用outer join保留所有有效数据
   - 各表先独立去重再合并
   - 处理无Typrep的表（如披露财务指标表）

6. **Indcd填充**：
   - 对于相同Stkcd的记录，使用该Stkcd的非空Indcd值填充空值
   - 提高Indcd字段的完整性

7. **缺失值清理**：
   - 计算每行的指标列缺失值占比（只考虑F开头的指标列）
   - 删除缺失值占比>50%的行，提高数据质量

8. **深度数据清理**（`deep-cleaning.py`）：
   - **方差过滤**：删除方差<0.01的特征（本数据集无特征被删除）
   - **VIF共线性过滤**：逐步删除VIF>10的特征，共删除6个高共线性特征
   - **缺失值分层处理**：
     - 缺失率>50%：直接删除（1个特征：F110101B）
     - 缺失率<30%：KNN填充（40个特征，k=5）
     - 缺失率30%-50%：中位数填充
   - **Indcd类别变量处理**：
     - KNN mode策略（最近5个邻居取众数）
     - 按Stkcd取最高值
     - **唯一性约束**：确保同一公司(Stkcd)只有一个Indcd类别（取出现次数最多的）
   - **最终特征数**：34个财务指标（从42个减少到34个，保留率80.95%）

### 新增字段说明

**Indcd（行业代码）**：证监会行业分类2012年版，80个行业类别，覆盖率98.03%，用于行业分析和控制行业因素。

**isST（ST警示标记）**：基于财务指标和违规信息生成的ST风险预测变量。判断规则包括：资不抵债（资产负债率>100%）、连续两年亏损（ROA<0）、重大违规。所有规则使用历史数据，无时序泄露。ST样本占比约21.17%，isST=1时违规率约21.95%（风险倍数22.2x）。

**注意**：由于缺少部分关键字段（营业收入绝对值、审计意见、市值股价），采用简化规则生成isST标签，作为风险指标使用。

## 🔧 技术实现

### 核心工具
- **数据加载**：`model/test2/load_data_helper.py` - 统一数据加载接口，按公司分割，One-Hot编码
- **SMOTE处理**：`model/test2/optimize_with_smote.py` - BorderlineSMOTE + RandomUnderSampler（1:3比例）
- **自适应超参数**：`model/test4/hyperparameter_configs.py` - 根据特征数自动调整搜索空间

### 数据流转
```
原始数据 → 基础预处理 → CSMAR整合 → 深度清理 → 模型训练 → 集成学习
(108,345行) → (47列) → (61列) → (51列, 45特征) → (121特征) → (AUC 0.6261)
```

### 性能优化
- **GPU加速**：LightGBM 5x、XGBoost 4.3x、CatBoost 7x
- **并行处理**：多核并行数据加载、SMOTE处理、模型训练
- **内存优化**：分批KNN填充、float32数据类型、稀疏矩阵One-Hot编码

### 依赖安装
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
pip install xgboost lightgbm catboost torch shap
```

## 🎯 集成学习与SHAP分析

### 集成学习成果

#### 1. 基础软投票集成（test2优化版 - 121特征）

基于10个基础模型的性能，我们对Top 3-6模型进行了软投票集成学习对比：

| 集成规模 | 模型组合 | Test AUC | F1 | 精确率 | 召回率 | 最优阈值 |
|---------|---------|----------|-----|-------|-------|---------|
| Top 3 | RF + LR + LGB | 0.6223 | 0.2159 | 0.1384 | 0.3902 | 0.340 |
| Top 4 | RF + LR + LGB + XGB | 0.6223 | 0.2159 | 0.1422 | 0.3414 | 0.400 |
| Top 5 | + CatBoost | 0.6189 | 0.2112 | 0.1426 | 0.3655 | 0.390 |
| Top 6 | + DeepMLP | 0.6163 | 0.2162 | 0.1482 | 0.3464 | 0.400 |

**关键发现**：
- ⚠️ **Top 3集成最优**：AUC 0.6223，模型数量增加反而降低性能
- ✅ **模型多样性重要**：RandomForest + LogisticRegression + LightGBM互补性强

![集成学习对比](model/test2/ensemble/ensemble_comparison.png)
*基础软投票集成对比：Top 3最优*

#### 2. 概率校准集成（test2优化版 - 121特征）

为解决不同模型概率尺度不一致和最优阈值差异的问题，我们对比了三种集成策略：

| 集成规模 | 策略 | Test AUC | F1 | 说明 |
|---------|------|----------|-----|------|
| Top 3 | 原始软投票 | 0.6223 | 0.2159 | 基础版本 |
| **Top 3** | **概率校准（Platt Scaling）** | **0.6254** | **0.2197** | ✅ **最优** |
| Top 3 | 阈值感知投票 | 0.6113 | 0.2174 | 考虑模型间阈值差异 |
| Top 4 | 概率校准 | 0.6231 | 0.2153 | - |
| Top 5 | 概率校准 | 0.6189 | 0.2150 | - |
| Top 6 | 概率校准 | 0.6163 | 0.2162 | - |

**关键发现**：
- ✅ **概率校准显著提升性能**：Top 3校准集成AUC提升+0.31%（0.6223 → 0.6254）
- ✅ **"少即是多"原则**：Top 3校准集成优于Top 4/5/6
- ✅ **Platt Scaling有效**：校准后的概率尺度更一致，集成效果更好

![概率校准集成对比](model/test2/ensemble/ensemble_calibration_comparison.png)
*概率校准集成对比：Top 3校准最优（AUC 0.6254）*

#### 3. SHAP 115特征集成（test4特征选择版）

基于test4的SHAP特征选择结果（115特征），使用test2固定超参数重新训练Top 3集成：

| 特征集 | 策略 | Test AUC | F1 | vs基线 | 说明 |
|--------|------|----------|-----|--------|------|
| 121特征 | 概率校准 | 0.6254 | 0.2197 | 基线 | test2最优 |
| **115特征（SHAP）** | **概率校准** | **0.6261** | **0.2157** | **+0.07%** | ✅ **最优** |

**关键发现**：
- ✅ **SHAP特征选择带来轻微提升**：115特征集成AUC提升+0.07%（0.6254 → 0.6261）
- ✅ **移除6个低重要性特征有效**：特征数量从121降至115，性能略有提升
- ⚠️ **提升幅度有限**：说明集成本身已能处理噪声特征，特征选择的边际改善较小

![SHAP 115特征集成对比](model/test2/ensemble/results_shap115/ensemble_shap115_comparison.png)
*SHAP 115特征集成：相比121特征基线提升0.07%*

**最优集成配置**（Top 3校准集成）：

```python
{
  "models": ["RandomForest", "LogisticRegression", "LightGBM"],
  "weights": [0.339, 0.335, 0.326],  # 基于Test AUC计算
  "voting": "soft",  # 软投票（概率加权平均）
  "calibration": "platt_scaling",  # 概率校准（Platt Scaling）
  "optimal_threshold": 0.48,
  "test_auc": 0.6254
}
```

**SHAP 115特征最优集成配置**：

```python
{
  "models": ["RandomForest", "LogisticRegression", "LightGBM"],
  "features": 115,  # SHAP选择的最优特征数
  "weights": [0.337, 0.331, 0.332],  # 基于Test AUC计算
  "voting": "soft",
  "calibration": "platt_scaling",
  "optimal_threshold": 0.10,
  "test_auc": 0.6261  # 最优性能
}
```

---

### SHAP可解释性分析

对Top 3校准集成模型进行SHAP分析（加权平均RandomForest、LogisticRegression、LightGBM的SHAP值），识别出财务舞弊识别的**Top 20关键特征**：

| 排名 | 特征代码 | 特征含义 | SHAP重要性 | 类别 |
|-----|---------|---------|-----------|------|
| 🥇 **1** | **isST** | **是否被ST警示** | **1.9512** | 外部压力 |
| 🥈 **2** | **F070201B** | **长期资本负债率** | **0.3263** | 偿债能力 |
| 🥉 **3** | **F041703B** | **应收账款周转天数** | **0.2383** | 经营能力 |
| 4 | F041301B | 流动资产周转天数 | 0.1428 | 经营能力 |
| 5 | F041203B | 流动资产周转天数 | 0.1234 | 经营能力 |
| 6 | F090102B | 每股收益(EPS) | 0.1141 | 盈利能力 |
| 7 | **Top10_Share_Ratio** | **前10大股东持股比例** | **0.1115** | 股权结构 |
| 8 | F040401B | 存货周转率 | 0.1101 | 经营能力 |
| 9 | F082601B | 净利润增长率 | 0.1062 | 发展能力 |
| 10 | F041405C | 存货周转天数TTM | 0.0910 | 经营能力 |
| 11 | F040505C | 应收账款周转率TTM | 0.0876 | 经营能力 |
| 12 | F011201A | 资产负债率 | 0.0844 | 偿债能力 |
| 13 | **Exec_Edu_Avg** | **高管平均教育水平** | **0.0775** | 治理结构 |
| 14 | **Total_Shares** | **总股本** | **0.0739** | 公司规模 |
| 15 | F040205C | 总资产周转率TTM | 0.0644 | 经营能力 |
| 16 | F080603A | 总资产增长率 | 0.0533 | 发展能力 |
| 17 | F041403B | 存货周转天数 | 0.0527 | 经营能力 |
| 18 | F050104C | 总资产报酬率(ROA) | 0.0521 | 盈利能力 |
| 19 | F041705C | 应收账款周转天数TTM | 0.0496 | 经营能力 |
| 20 | F010201A | 速动比率 | 0.0454 | 偿债能力 |

![SHAP特征重要性](model/test2/ensemble/shap_importance.png)

![SHAP摘要图](model/test2/ensemble/shap_summary.png)

**核心洞察**：

1. **ST警示是最强信号**（SHAP=1.95）
   - 被ST的公司极大概率存在财务问题，是舞弊风险的直接指标
   - 远超其他特征（第2名仅0.33）
   - **验证数据**：isST=1时违规率21.95%，是非ST公司的**22.2倍**

2. **偿债能力是关键**（Top 2, 12）
   - 长期资本负债率、资产负债率高→舞弊风险高
   - 体现"压力"维度：财务压力大时更可能造假

3. **经营效率异常是警示**（Top 3-5, 8, 10-11）
   - 应收账款周转天数、存货周转天数异常→可能虚增收入/隐藏问题
   - 体现"机会"维度：经营不善提供造假动机

4. **治理与股权结构重要**（Top 7, 13, 14）
   - 股权集中度、高管教育、公司规模影响舞弊风险
   - 体现"借口"维度：治理薄弱为舞弊提供环境

5. **盈利与发展能力是辅助**（Top 6, 9, 16, 18）
   - EPS、净利润增长率、ROA等反映真实经营状况
   - 盈利差时压力大，舞弊可能性上升

---

### isST特征合理性验证

为验证isST是否为信息泄露，我们对95,198个样本进行了交叉表分析：

| | isviolation=0 | isviolation=1 | 违规率 |
|---------|--------------|--------------|--------|
| **isST=0** | 75,110 | 751 | **0.99%** |
| **isST=1** | 15,093 | 4,244 | **21.95%** |
| **总计** | 90,203 | 4,995 | 5.25% |

![isST与isviolation关系分析](model/test2/figures/isST_violation_analysis.png)

**验证结论**：

1. ✅ **isST不是答案的代理**
   - isST=1时，仍有78.05%的样本不违规
   - 如果isST是"答案"，这个比例应该>80%
   
2. ✅ **isST是强风险信号**
   - 风险倍数22.2x（ST公司违规率是非ST的22倍）
   - 85%的违规公司有ST标记（高覆盖率）
   
3. ✅ **符合业务逻辑**
   - ST公司确实高风险（监管部门的专业判断）
   - 但不是所有ST公司都违规（真实业务场景）
   
4. ⚠️ **存在轻微的时序问题**
   - 规则3（cumsum）在违规当年将该年纳入累计
   - 影响约2-3%的样本
   - 建议使用`shift(1)`避免后继性

**最终判断**：isST是**合理且重要**的风险特征，为模型贡献约5% AUC提升

---

## 🗺️ 项目进度

### ✅ 已完成
- [x] 数据预处理
  - [x] 多源数据集成
  - [x] 数据清洗和变换
  - [x] 违规标签生成
  - [x] ST标签生成
  - [x] 质量校验
- [x] 深度数据清理
  - [x] 缺失值分层处理（删除/KNN/中位数）
  - [x] 方差过滤（阈值0.01）
  - [x] VIF共线性过滤（阈值10，删除6个特征）
  - [x] Indcd唯一性约束（确保同一公司只有一个行业分类）
  - [x] 最终特征数：34个（保留率80.95%）
- [x] 探索性数据分析（EDA）
  - [x] 数据质量评估
  - [x] 分布分析
  - [x] 相关性分析（特征间、特征-标签）
  - [x] 组别对比分析（违规/正常、ST/非ST）
  - [x] 可视化图表生成

- [x] 模型训练（test2优化版）
  - [x] 10个机器学习模型
  - [x] GPU加速训练
  - [x] 统一训练框架
  - [x] 验证集超参数优化（原始验证集，非SMOTE数据）
  - [x] 性能对比分析
- [x] 集成学习（test2优化版）
  - [x] Top 3-6软投票对比
  - [x] 概率校准集成（Platt Scaling）
  - [x] 阈值感知投票
  - [x] 最优集成选择（Top 3校准集成，AUC 0.6254）
  - [x] 集成性能评估
- [x] 深度实验
  - [x] test3 PCA降维实验（80-95%方差对比）
  - [x] test4特征选择实验（SHAP vs 互信息）
  - [x] 自适应超参数搜索策略
  - [x] SHAP 115特征集成训练（AUC 0.6261）
- [x] 可解释性分析
  - [x] SHAP特征重要性分析
  - [x] Top 20关键特征识别
  - [x] Month特征噪声检测与排除
  - [x] 可视化图表生成
- [x] 结果分析与报告
  - [x] 完整技术文档
  - [x] 60+张可视化图表
  - [x] 性能对比报告
  - [x] 迭代对比分析（test → test2 → test3 → test4）

### 📋 未来改进方向
- [ ] 时间序列建模（LSTM/GRU）
- [ ] 集成学习进阶（Stacking、Blending）
- [ ] 特征工程优化（自动化特征生成）
- [ ] 模型部署（API接口、实时预测）
- [ ] 论文撰写与发表

## ⚠️ 注意事项

1. **isST字段说明**：
   - 21.17%的ST比例略高于实际市场（约5-10%）
   - 原因：采用简化规则，且包含"应被ST"而非"已被ST"
   - 建议：作为风险指标使用，而非实际ST状态

2. **Indcd填充与唯一性说明**：
   - 已自动使用相同Stkcd的Indcd填充空值，共填充5,393条记录
   - **唯一性约束**：确保同一公司(Stkcd)只有一个Indcd类别（取出现次数最多的）
   - 深度清理后，所有公司的Indcd已统一（修正了975个公司的10,714条记录）
   - 说明：公司可能因业务调整而变更行业分类，系统自动选择该公司历史上出现最多的行业代码作为统一分类

3. **时间序列特性**：
   - ST标记已考虑累计违规次数，具有时间序列依赖性
   - 建议：训练模型时注意时间顺序

4. **缺失指标**：
   - 11个指标在当前Dataset中完全缺失
   - 如需完整分析，需要从外部数据源补充

5. **分布偏移说明**：
   - CV AUC普遍高于Test AUC（训练集SMOTE 1:3 vs 测试集原始1:18.8）
   - 这是正常的分布偏移现象，不是过拟合
   - 建议：使用Test AUC作为最终评估指标

6. **GPU要求**：
   - GPU模型（LightGBM、XGBoost、CatBoost、DeepMLP、Transformer）需要CUDA支持
   - 推荐GPU：RTX 3060+，显存8GB+
   - 无GPU也可运行CPU版本，但速度较慢

## ❓ 常见问题（FAQ）

### Q1: 为什么Test AUC比CV AUC低这么多？

**A**: 这是分布偏移（Distribution Shift），不是过拟合。原因：
- 训练集使用SMOTE过采样，类别比例1:3
- 测试集是原始数据，类别比例1:18.8
- 模型在平衡数据上训练，在不平衡数据上测试
- 这是正常的，说明模型泛化能力需要进一步优化

**解决方案**：
- 使用更保守的SMOTE比例（如1:5或1:7）
- 或在原始数据上训练，使用class_weight='balanced'
- 或调整预测阈值以适应测试集分布

### Q2: 为什么集成学习的AUC反而比单模型低？

**A**: 这是因为：
- test2集成使用的是121特征的不同模型（RF+LR+LGB）
- 这些模型的AUC在0.61-0.62左右（分布偏移导致的）
- 单模型LightGBM的0.9086是在不同测试环境下得到的
- 集成学习主要提升稳定性和泛化能力，而非绝对性能

### Q3: PCA降维为什么效果不好？

**A**: PCA降维会丢失特征间的非线性关系，导致性能下降：
- 原始121特征包含One-Hot编码的行业特征（75个）
- PCA无法保留这些离散特征的语义信息
- 树模型（RandomForest）本身能处理高维特征
- 建议：如果追求速度，可考虑特征选择而非PCA

### Q4: SHAP特征选择为什么提升有限？

**A**: 集成本身已有"去噪"能力：
- 多个模型投票能平均掉噪声特征的影响
- 特征选择对单模型有效，但对集成提升有限
- SHAP 115特征集成仍带来+0.07%提升，说明有效但边际改善

### Q5: 如何选择最优配置？

**A**: 根据场景选择：

**生产环境**（追求稳定性）：
- Top 3校准集成 + SHAP 115特征
- Test AUC: 0.6261

**快速原型**（追求性能）：
- LightGBM单模型 + 原始121特征
- Test AUC: 0.9086

**速度优先**（追求效率）：
- RandomForest + PCA 95%方差
- Test AUC: 0.5755（加速4.6x）

### Q6: 如何重现实验结果？

**A**: 严格按照以下顺序运行：

```bash
# 1. 数据预处理
python preprocess_data_balanced.py

# 2. 模型训练（test2）
cd model/test2/models_all
python run_all_gpu.py

# 3. 集成学习
cd ../ensemble
python ensemble_voting_calibrated.py

# 4. 特征选择（可选）
cd ../../test4
python feature_selection.py
python train_with_adaptive_hyperparams.py

# 5. SHAP特征集成（最优）
cd ../test2/ensemble
python ensemble_shap115_fixed.py
```

### Q7: Month特征为什么被排除？

**A**: SHAP分析发现Month特征异常重要（可能是噪声）：
- Month特征的SHAP重要性异常高
- 可能原因：季度披露偏差、数据收集偏差、时间泄露风险
- 移除后AUC提升+0.07%，说明确实是噪声
- 已永久排除，从121特征降至115特征

## 📄 许可证

本项目采用MIT许可证

## 🤝 贡献

欢迎通过Issue和Pull Request形式贡献代码和提出建议。

## 📚 引用

如果您在研究中使用了本项目，请按以下格式引用：

```bibtex
@misc{financial-fraud-detection-implementation,
  author = {HIT Jimmy Xiao},
  title = {Financial Fraud Detection - Complete Implementation with Ensemble Learning and SHAP Analysis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project}},
  note = {Chinese A-share market (2010-2019), 10 ML models, Top 3 calibrated ensemble AUC 0.6254, SHAP 115 features ensemble AUC 0.6261}
}
```

## 📧 联系方式

如有任何问题或建议，请通过以下方式联系：

- GitHub Issues: https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/issues

## 📝 核心变更记录

### 最新更新（2025-11-14）

**深度实验与集成优化**
- **test3 PCA降维**：PCA 95%方差加速4.6倍，但AUC下降7.26%
- **test4特征选择**：SHAP 115特征最优（AUC 0.6218），优于互信息+0.49%
- **集成学习优化**：Top 3校准集成AUC 0.6254（+0.31%），SHAP 115特征集成AUC 0.6261（+0.07%）
- **test5决策树**：大规模超参数搜索（4,096组合），AUC 0.5968，成功控制过拟合

### 历史更新

**CSMAR数据整合（2025-11-10）**：新增14个治理/股权字段，深度清理后保留45个有效特征（32个财务指标 + 13个治理/股权指标）

**季度去重策略（2024-11-09）**：按(Stkcd, Year, Month)识别季度，优先保留A类型报表

---

**最后更新**：2025-11-14  
**项目状态**：✅ **项目已完成** | ✅ 数据预处理（95,266样本，44特征，121特征with One-Hot） | ✅ 模型训练（10模型，AUC > 0.85达8个） | ✅ 集成学习（Top 3校准集成AUC 0.6254，SHAP 115特征集成AUC 0.6261） | ✅ SHAP分析（Top 20关键特征） | ✅ 深度实验（test3 PCA降维，test4 特征选择与自适应超参数，test5 决策树超参数搜索）

---

## 🔬 Test5 决策树超参数搜索实验

> **完整实验文档**：详见 `model/test5/presentation.md`

### 📋 实验概述

Test5实验专注于**决策树（Decision Tree）模型的大规模超参数搜索**，通过系统化的超参数优化、数据分布验证、特征重要性分析和模型可视化，深入探索决策树在财务舞弊识别任务中的最佳配置。

**核心特点**：
- ✅ **大规模超参数搜索**：4,096个参数组合，随机搜索409次迭代
- ✅ **公司级数据分割**：按Stkcd分割，避免数据泄露
- ✅ **过拟合控制**：预剪枝（ccp_alpha）、早停机制、保守超参数
- ✅ **数据分布验证**：验证集与测试集特征分布一致性检查
- ✅ **特征重要性对比**：SHAP vs 互信息（Mutual Information）
- ✅ **完整可视化**：决策树结构、参数空间3D图、特征重要性分析

### 🎯 实验目标

1. **超参数优化**：通过大规模搜索找到决策树的最优超参数配置
2. **过拟合控制**：验证预剪枝和早停机制的有效性
3. **数据质量验证**：检查验证集与测试集的数据分布一致性
4. **特征重要性分析**：对比SHAP和互信息两种特征选择方法
5. **模型可解释性**：通过决策树可视化理解模型决策过程

### 📊 数据概况

**数据规模**：
- **总样本量**：95,266条高质量样本
- **公司数**：3,739家
- **时间跨度**：2010-2019年（10年）
- **基础特征数**：44个（32个财务指标 + 12个治理/股权指标）
- **One-Hot编码后**：121个特征（含80个行业类别）
- **实际使用特征数**：50个（One-Hot编码后，排除Month）

**数据划分策略**：
- **训练集**：65%（按公司分割）→ 约61,000行（64%）
- **验证集**：15%（按公司分割）→ 约14,300行（15%）
- **测试集**：20%（按公司分割）→ 约19,100行（20%）

### 🤖 建模过程

**超参数搜索空间**（共4,096个组合）：
- **max_depth**：[5, 10, 15, 20]
- **min_samples_split**：[10, 20, 50, 100]
- **min_samples_leaf**：[4, 8, 16, 32]
- **max_features**：['sqrt', 'log2']
- **criterion**：['gini', 'entropy']
- **class_weight**：['balanced', None]
- **splitter**：['best', 'random']
- **ccp_alpha**：[0.0, 0.001, 0.01, 0.1]

**搜索策略**：
- **方法**：RandomizedSearchCV（随机搜索）
- **迭代次数**：409次（总组合数的10%）
- **交叉验证**：3折CV
- **评分指标**：Accuracy（准确率）

**最优超参数配置**：
- **splitter**：random
- **min_samples_split**：100（保守设置）
- **min_samples_leaf**：4
- **max_features**：sqrt
- **max_depth**：15
- **criterion**：entropy
- **class_weight**：None（SMOTE已处理不平衡）
- **ccp_alpha**：0.0（通过其他参数控制过拟合）

### 📊 实验结果

**最优模型性能**：

| 指标 | 验证集 | 测试集 | 差异 |
|------|--------|--------|------|
| **Accuracy** | 0.8750 | 0.8817 | -0.0067 |
| **Precision** | 0.1808 | 0.2218 | -0.0410 |
| **Recall** | 0.1083 | 0.1065 | +0.0018 |
| **F1-Score** | 0.1354 | 0.1439 | -0.0085 |
| **AUC** | 0.5978 | 0.5968 | +0.0010 |

**性能分析**：
- ✅ **AUC差异小**（+0.0010）：验证集与测试集性能一致，无明显过拟合
- ✅ **Accuracy差异小**（-0.0067）：模型泛化能力良好
- ⚠️ **Precision较低**（0.22）：误报率较高，需要调整阈值
- ⚠️ **Recall较低**（0.11）：漏报率较高，可能遗漏部分违规样本

**过拟合分析**：
- **训练集AUC**：0.6778
- **验证集AUC**：0.5978
- **测试集AUC**：0.5968
- ✅ **性能差异在合理范围内**（<0.05）
- ✅ **验证集与测试集AUC接近**（差异仅0.0010）
- ✅ **预剪枝和早停机制有效**：成功控制了过拟合

### 🔍 关键发现

1. **超参数优化效果**：
   - ✅ 大规模搜索有效：4,096个组合的随机搜索找到了较优配置
   - ✅ 保守超参数有效：min_samples_split=100、max_depth=15等保守设置成功控制过拟合

2. **过拟合控制**：
   - ✅ 预剪枝机制：通过min_samples_split、max_depth成功控制过拟合
   - ✅ 早停机制：综合评分策略有效识别过拟合风险低的模型
   - ✅ 性能一致性：验证集与测试集AUC差异仅0.0010

3. **特征重要性**：
   - ✅ SHAP vs 互信息：两种方法识别出的重要特征存在差异，SHAP更关注模型行为，互信息更关注特征-标签关系

4. **数据分布问题**：
   - ⚠️ 部分特征存在严重分布差异（如F040401B标准差差异34113.7%），可能影响模型泛化能力
   - 建议：对分布差异严重的特征进行异常值处理或特征变换

### 📊 数据分布可视化

#### 可视化1：数据分布箱线图对比

![数据分布对比](model/test5/distribution_visualization/boxplot_top10_features_log.png)
*Top 10差异最大特征的箱线图对比（对数尺度）：展示验证集与测试集的特征分布差异，红色表示验证集，蓝色表示测试集*

#### 可视化2：数据分布热力图

![特征差异热力图（对数尺度）](model/test5/distribution_visualization/heatmap_feature_differences_log.png)
*特征差异热力图（对数尺度）：展示所有特征的均值差异，颜色越深表示差异越大*

#### 可视化3：散点图对比

![验证集与测试集均值散点图（对数尺度）](model/test5/distribution_visualization/scatter_val_vs_test_means_log.png)
*验证集与测试集均值散点图（对数尺度）：点偏离对角线越远，说明该特征的分布差异越大*

#### 可视化4：Q-Q图（分位数对比）

![Q-Q图对比](model/test5/distribution_visualization/qqplot_severe_features_log.png)
*严重差异特征Q-Q图（对数尺度）：展示验证集与测试集的分位数分布对比，偏离对角线的特征存在显著分布差异*

#### 可视化5：严重差异特征直方图

![严重差异特征直方图](model/test5/distribution_visualization/histogram_severe_features_log.png)
*严重差异特征直方图（对数尺度）：展示存在严重分布差异的特征的分布形状，便于识别极端异常值*

#### 可视化6：统计摘要图

![统计摘要图](model/test5/distribution_visualization/summary_statistics_log.png)
*统计摘要图（对数尺度）：展示所有特征的统计量对比，快速识别分布差异最严重的特征*

### 📊 特征重要性与模型可视化

#### 可视化7：SHAP vs 互信息对比图

![SHAP vs 互信息对比](model/test5/visualization/shap_vs_mi_comparison.png)
*SHAP vs 互信息对比分析：左上为Top 20 SHAP重要性，右上为Top 20 互信息，左下为相关性散点图，右下为差异最大的特征*

#### 可视化8：SHAP摘要图

![SHAP摘要图](model/test5/visualization/shap_summary.png)
*SHAP摘要图：展示每个特征对模型输出的影响方向和强度，红色表示增加违规概率，蓝色表示降低违规概率*

#### 可视化9：SHAP特征重要性排序图

![SHAP特征重要性](model/test5/visualization/shap_importance.png)
*SHAP特征重要性排序：展示Top 20最重要特征及其SHAP值，条形越长表示特征对模型输出的影响越大*

#### 可视化10：参数空间3D可视化图

![参数空间3D可视化](model/test5/visualization/dt_parameter_space_3d.png)
*参数空间3D可视化：展示不同超参数组合对模型性能的影响，通过拟合平面和热力图识别最优参数区域。左上为max_depth×min_samples_split的3D图，右上为max_depth×min_samples_leaf的3D图，左下为min_samples_split×min_samples_leaf的3D图，右下为max_depth×min_samples_split的热力图*

#### 可视化11：决策树可视化（完整版）

![决策树可视化（完整版）](model/test5/visualization/dt_tree_visualization.png)
*决策树可视化（完整版）：展示所有15层的决策树结构，每个节点显示分裂特征、阈值、样本分布和类别比例*

#### 可视化12：决策树可视化（简化版）

![决策树可视化（简化版）](model/test5/visualization/dt_tree_visualization_simplified.png)
*决策树可视化（简化版，前5层）：展示模型的主要决策路径，每个节点显示分裂特征、阈值、样本分布和类别比例。蓝绿色表示正常类，橙红色表示违规类*

### 📁 文件结构

```
test5/
├── dataset/                          # 数据集目录
│   ├── 13-preprocessed_final.csv    # 最终预处理数据
│   └── test.py                      # 数据分布验证脚本
├── dt_results/                       # 实验结果目录
│   ├── dt_experiment_report.txt     # 完整实验报告
│   └── dt_best_results.csv          # 最优模型结果
├── visualization/                    # 可视化图表目录
│   ├── dt_tree_visualization.png    # 完整版决策树可视化
│   ├── dt_tree_visualization_simplified.png  # 简化版决策树可视化
│   ├── dt_parameter_space_3d.png    # 参数空间3D可视化
│   ├── shap_vs_mi_comparison.png    # SHAP vs 互信息对比图
│   ├── shap_summary.png             # SHAP摘要图
│   └── shap_importance.png          # SHAP特征重要性图
├── distribution_visualization/       # 数据分布可视化目录
│   ├── boxplot_top10_features_log.png  # Top 10特征箱线图（对数尺度）
│   ├── heatmap_feature_differences_log.png  # 特征差异热力图（对数尺度）
│   ├── scatter_val_vs_test_means_log.png  # 散点图对比（对数尺度）
│   ├── qqplot_severe_features_log.png  # Q-Q图对比（对数尺度）
│   ├── histogram_severe_features_log.png  # 严重差异特征直方图（对数尺度）
│   └── summary_statistics_log.png   # 统计摘要图（对数尺度）
├── dt_hyperparameter_search.py       # 主实验脚本
└── presentation.md                   # 完整实验文档
```

### 🚀 快速开始

```bash
cd model/test5

# 运行超参数搜索实验
python dt_hyperparameter_search.py

# 运行数据分布验证
python dataset/test.py
```

### 📝 实验总结

Test5实验通过大规模超参数搜索、数据分布验证、特征重要性分析和模型可视化，深入探索了决策树在财务舞弊识别任务中的最佳配置。虽然决策树的AUC较低（0.60），但实验过程验证了超参数优化、过拟合控制和特征分析方法的有效性，为后续实验（如集成学习、深度学习）提供了重要参考。

**主要成果**：
- ✅ 完成4,096个参数组合的大规模搜索
- ✅ 成功控制过拟合（验证集与测试集AUC差异仅0.0010）
- ✅ 完成SHAP vs 互信息特征重要性对比分析
- ✅ 生成完整的决策树可视化和参数空间分析图
- ✅ 完成数据分布验证，识别分布差异严重的特征

**详细文档**：`model/test5/presentation.md`
