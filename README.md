<p align="center">
  <img src="https://img.shields.io/badge/财务舞弊识别-数据预处理系统-blue" alt="财务舞弊识别-数据预处理系统" width="600"/>
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

本项目是针对中国A股市场（2010-2019年）的财务舞弊识别研究项目。基于舞弊三角理论（压力、机会、借口），通过数据预处理、特征工程、模型训练等步骤，构建财务舞弊识别系统。

## 🎯 项目目标

- **数据预处理**：对多源财务数据进行集成、清洗、变换和标注
- **特征工程**：基于舞弊三角理论构建特征体系
- **模型训练**：训练机器学习模型识别财务舞弊
- **结果分析**：分析模型性能和改进方向

## ✨ 项目特点

- **理论基础**：基于舞弊三角理论（压力、机会、借口）构建特征体系
- **多源数据集成**：整合8个财务主题表（偿债能力、经营能力、盈利能力等）和违规信息表
- **完整流程**：从数据预处理到模型训练的完整pipeline
- **质量保证**：完整的数据质量校验和模型评估流程

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
│   ├── 1-preprocessed.csv        # ✅ 最终预处理数据（57,621行×42列）
│   ├── preprocess_log_balanced.txt  # 处理日志
│   ├── 质量报告_最终版.md        # 数据质量报告
│   └── 完成总结_最终版.md        # 任务完成总结
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
pip install pandas numpy openpyxl scikit-learn matplotlib seaborn
```

### 阶段1：数据预处理

```bash
# 推荐使用balanced版本（三键策略）
python preprocess_data_balanced.py

# 或使用简化版本（二键策略）
python preprocess_data_tiny_version.py
```

### 使用Jupyter Notebook逐步执行

```bash
# 打开Jupyter Notebook
jupyter notebook 数据预处理步骤指南.ipynb
```

### 输出文件位置

运行成功后，预处理数据将保存在：
- **主文件**：`Insight_output/1-preprocessed.csv`（包含Indcd和isST字段）
- **日志**：`Insight_output/preprocess_log_balanced.txt`
- **质量报告**：`Insight_output/质量报告_最终版.md`
- **完成总结**：`Insight_output/完成总结_最终版.md`

### 后续阶段

- **阶段2：特征工程**（待实现）
- **阶段3：模型训练**（待实现）
- **阶段4：结果分析**（待实现）

## 📊 数据统计

### 预处理后数据

| 指标 | 数值 |
|------|-----|
| **样本量** | 57,621 条 |
| **公司数** | 3,757 家 |
| **时间跨度** | 2010-2019 (10年) |
| **特征数** | 42 列（6个主键/标签 + 36个财务指标） |
| **违规样本** | 2,989 (5.19%) |
| **ST样本** | 11,301 (19.61%) |
| **文件大小** | 16.87 MB |

### 报表类型分布

- **A类型（年报）**：29,025 条 (50.37%)
- **B类型（半年报）**：28,596 条 (49.63%)

## 📖 数据预处理详情

### 核心策略

1. **三键主键策略**：使用 (Stkcd, Year, Typrep) 作为主键
   - Stkcd：标准化为6位字符串
   - Year：从Accper日期字段提取
   - Typrep：保留原值，优先级 K > C > A > B

2. **智能去重**：
   - 按Typrep优先级排序
   - 每个 (Stkcd, Year, Typrep) 组合只保留一条记录
   - 避免数据爆炸（从200k+行稳定到57k行）

3. **中文列名处理**：
   - 自动识别偿债能力表的中文列名
   - 动态映射到标准英文列名

4. **数据集成**：
   - 使用outer join保留所有有效数据
   - 各表先独立去重再合并
   - 处理无Typrep的表（如披露财务指标表）

### 新增字段说明

### 1. Indcd（行业代码）

**字段含义：** 证监会行业分类2012年版

**数据来源：** 偿债能力表（FI_T1.xlsx）

**覆盖率：** 98.03%（56,486/57,621条记录）

**唯一值：** 80个不同行业代码

**示例值：** `K70`（房地产业）、`C27`（医药制造业）、`S90`（综合）等

**用途：** 
- 按行业分类进行财务舞弊特征分析
- 控制行业因素的影响
- 行业对比分析

**使用示例：**

```python
import pandas as pd
df = pd.read_csv('Insight_output/1-preprocessed.csv')

# 按行业分组统计ST比例
st_by_industry = df.groupby('Indcd').agg({
    'isST': 'mean',
    'Stkcd': 'count'
}).round(4)
print("各行业ST比例:")
print(st_by_industry)
```

### 2. isST（ST警示标记）

**字段含义：** 标记公司是否应被ST（特别处理）

**取值：** 
- `0`：正常，未满足ST条件
- `1`：应被ST，满足ST判断规则

**判断规则（基于2024年修订规则）：**

#### 规则1：资不抵债
- **条件：** 资产负债率 > 100%（即净资产为负）
- **对应指标：** `F011201A` > 1.0
- **依据：** 主板公司最近一年期末净资产为负值触发*ST

#### 规则2：净利润为负 + 违规记录
- **条件：** ROA < 0 且 isviolation = 1
- **对应指标：** `F050204C` < 0 且有违规记录
- **依据：** 财务表现不佳叠加违规行为，增加ST风险

#### 规则3：累计违规次数
- **条件：** 同一公司累计违规 ≥ 2次
- **计算方法：** 按公司分组，累加违规次数
- **依据：** 连续违规公司更可能被ST

**统计结果：**
- ST样本数：11,301（19.61%）
- 非ST样本：46,320（80.39%）

**ST样本构成：**
- ST且违规：2,144（3.72%）- 同时满足财务异常和违规
- ST但未违规：9,157（15.89%）- 仅因财务异常（如资不抵债）

**局限性说明：**

由于数据集中缺少以下关键字段，无法实现完整ST判断规则：
1. ❌ 营业收入绝对值（无法判断"净利润为负且营收<3亿"）
2. ❌ 审计意见类型（无法判断"无法表示意见"或"否定意见"）
3. ❌ 市值和股价数据（无法判断"连续20日市值<5亿或股价<1元"）
4. ❌ 资金占用、违规担保等规范类指标
5. ❌ 现金分红数据（无法判断"分红不达标"）

因此，本脚本采用**简化规则**，基于现有财务指标和违规信息生成isST标签，作为ST风险的**代理变量**使用。

## 📋 指标完整性检查报告

### 检查的12个指标

1. 董事长与总经理兼任情况
2. 四委设立个数
3. 监管层持股比例
4. 国有股比例
5. 是否被ST
6. 管理层持股比例
7. 高管持股比例
8. 高管平均教育背景
9. 审计意见类型
10. 股权集中指标%
11. Herfindahl_5指数
12. Z指数

### 最终检查结果

#### ✅ 已存在并已集成（1个）

**行业大类（Indcd）**
- 位置：偿债能力表（FI_T1.xlsx）
- 字段名：`Indcd [行业代码]`（证监会行业分类2012年版）
- 状态：✅ 已成功提取并集成到最终输出
- 覆盖率：98.03%

#### ❌ 完全缺失（11个）

经过深度检查（包括列名、数据内容、关键词匹配），以下11个指标在Dataset的所有9个Excel文件中**完全不存在**：

**机会指标 - 治理结构类（4个）**
1. ❌ **董事长与总经理兼任情况** - 不存在
2. ❌ **四委设立个数** - 不存在
3. ❌ **监管层持股比例** - 不存在
4. ❌ **国有股比例** - 不存在

**机会指标 - 股权结构类（3个）**
5. ❌ **股权集中指标%** - 不存在
6. ❌ **Herfindahl_5指数** - 不存在
7. ❌ **Z指数** - 不存在

**借口指标（2个）**
8. ❌ **高管平均教育背景** - 不存在
9. ❌ **审计意见类型** - 不存在

**管理层财务状况（2个）**
10. ❌ **管理层持股比例** - 不存在
11. ❌ **高管持股比例** - 不存在

**外部压力（1个）**
12. ❌ **是否被ST** - 不存在（已通过计算生成isST作为代理变量）

### 检查覆盖范围

✅ 检查了所有9个Excel文件：
- 偿债能力表（FI_T1.xlsx）
- 披露财务指标表（FI_T2.xlsx）
- 经营能力表（FI_T4.xlsx）
- 盈利能力表（FI_T5.xlsx）
- 风险水平表（FI_T7.xlsx）
- 发展能力表（FI_T8.xlsx）
- 每股指标表（FI_T9.xlsx）
- 股利分配表（FI_T11.xlsx）
- 违规信息总表（STK_Violation_Main.xlsx）

✅ 检查方法：
- DES描述文件
- Excel列名
- 数据内容（前100行）
- 关键词匹配（中英文、缩写、同义词）

### 结论

**总计12个指标：**
- ✅ 已存在并已集成：1个（行业大类 Indcd）
- ❌ 完全缺失：11个

**缺失率：91.7%**（11/12）

这11个缺失的指标需要从外部数据源获取，或通过计算得出：
- 治理结构、股权结构类指标：需要公司治理数据、股东持股明细数据
- 管理层持股、高管持股：需要高管持股明细数据
- 审计意见：需要年报审计报告数据
- ST标记：需要交易所公告或公司基本信息数据（已通过财务指标计算生成isST）
- 高管教育背景：需要高管个人信息数据

## 🔧 技术实现

### 数据预处理代码结构

```python
class FinancialDataPreprocessor:
    def __init__(self):          # 初始化配置
    def read_excel_safe(self):   # 安全读取Excel
    def standardize_stock_code(self):  # 标准化股票代码
    def extract_year(self):      # 提取年份
    def load_and_prepare_table(self):  # 读取并规范化单个表
    def merge_financial_tables(self):  # 数据集成
    def add_violation_label(self):     # 生成违规标签
    def clean_data(self):        # 数据清洗
    def generate_st_label(self): # 生成ST标签
    def prepare_final_output(self):    # 准备最终输出
    def quality_check(self):     # 质量校验
    def save_output(self):       # 导出文件
```

## 📈 数据质量指标

### 完整性检查
- 主键完整率：100%
- 数据量：57,621 条
- 公司数：3,757 家
- 年份范围：2010-2019

### 一致性检查
- 报表类型分布合理（A:50.37%, B:49.63%）
- 违规比例：5.19%（合理范围：3-10%）
- ST比例：19.61%（基于简化规则）

### 合理性检查
- 缺失值统计：大部分列缺失率 < 15%
- 高缺失列：10个列缺失率 > 10%，主要为业务合理缺失
- 数据类型：所有指标列已转换为数值类型

## 🗺️ 项目进度

### ✅ 已完成
- [x] 数据预处理
  - [x] 多源数据集成
  - [x] 数据清洗和变换
  - [x] 违规标签生成
  - [x] ST标签生成
  - [x] 质量校验

### 🚧 进行中
- [ ] 特征工程
- [ ] 模型训练
- [ ] 结果分析

### 📋 待规划
- [ ] 模型优化
- [ ] 结果可视化
- [ ] 论文撰写

## ⚠️ 注意事项

1. **isST字段说明**：
   - 19.61%的ST比例略高于实际市场（约5-10%）
   - 原因：采用简化规则，且包含"应被ST"而非"已被ST"
   - 建议：作为风险指标使用，而非实际ST状态

2. **Indcd缺失处理**：
   - 1.97%的样本Indcd为空
   - 建议：使用时可填充为"未知行业"或根据公司历史行业代码补充

3. **时间序列特性**：
   - ST标记已考虑累计违规次数，具有时间序列依赖性
   - 建议：训练模型时注意时间顺序

4. **缺失指标**：
   - 11个指标在当前Dataset中完全缺失
   - 如需完整分析，需要从外部数据源补充

## 📄 许可证

本项目采用MIT许可证

## 🤝 贡献

欢迎通过Issue和Pull Request形式贡献代码和提出建议。

## 📚 引用

如果您在研究中使用了本项目，请按以下格式引用：

```bibtex
@misc{financial-fraud-detection-preprocessing,
  author = {HIT Jimmy Xiao},
  title = {Financial Fraud Detection Implementation Project - Data Preprocessing},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project}
}
```

## 📧 联系方式

如有任何问题或建议，请通过以下方式联系：

- GitHub Issues: https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/issues

---

**最后更新**：2025-11-08  
**项目状态**：✅ 数据预处理已完成 | 🚧 特征工程和模型训练进行中
