<p align="center">
  <img src="https://img.shields.io/badge/Financial-Fraud-Detection-Data-Preprocessing-blue" alt="Financial Fraud Detection - Data Preprocessing System" width="600"/>
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
        <a href="https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/blob/main/README.md">简体中文</a> |
        <b>English</b>
    </p>
</h4>

## 📋 Project Overview

This project is a data preprocessing system for financial fraud detection in the Chinese A-share market (2010-2019). Based on the Fraud Triangle Theory (Pressure, Opportunity, Rationalization), it integrates, cleans, transforms, and labels multi-source financial data to provide high-quality datasets for subsequent machine learning model training.

## ✨ System Features

- **Multi-source Data Integration**: Integrates 8 financial theme tables (solvency, operating capability, profitability, etc.) and violation information tables
- **Three-key Strategy**: Uses (Stkcd, Year, Typrep) as primary keys to ensure data integrity
- **Intelligent Deduplication**: Smart deduplication based on Typrep priority to avoid data explosion
- **Chinese Column Name Handling**: Automatically identifies and converts Chinese column names in solvency tables
- **New Fields**: Integrates industry code (Indcd) and ST warning marker (isST)
- **Quality Assurance**: Complete data quality validation process

## 🔍 Project Structure

```
Financial_Fraud_Detection_Implementation_Project/
├── Dataset/                      # Raw dataset directory
│   ├── 偿债能力/                 # FI_T1.xlsx - Solvency indicators (with Indcd)
│   ├── 经营能力/                 # FI_T4.xlsx - Operating capability indicators
│   ├── 盈利能力/                 # FI_T5.xlsx - Profitability indicators
│   ├── 发展能力/                 # FI_T8.xlsx - Growth capability indicators
│   ├── 风险水平/                 # FI_T7.xlsx - Risk level indicators
│   ├── 披露财务指标/             # FI_T2.xlsx - Disclosed financial indicators
│   ├── 每股指标/                 # FI_T9.xlsx - Per-share indicators
│   ├── 股利分配/                 # FI_T11.xlsx - Dividend distribution indicators
│   ├── 违规信息总表/             # STK_Violation_Main.xlsx - Violation labeling data source
│   └── 集成数据示例.csv          # Reference example
│
├── Insight_output/               # Output directory
│   ├── 1-preprocessed.csv        # ✅ Final preprocessed data (57,621 rows × 42 columns)
│   ├── preprocess_log_balanced.txt  # Processing log
│   ├── 质量报告_最终版.md        # Data quality report
│   └── 完成总结_最终版.md        # Task completion summary
│
├── preprocess_data_balanced.py   # ✅ Recommended preprocessing script (three-key strategy)
├── preprocess_data_tiny_version.py  # Simplified preprocessing script (two-key strategy)
├── 数据预处理步骤指南.ipynb      # Jupyter Notebook step-by-step guide
├── taskmap.md                    # Task guidance checklist
├── README.md                     # This file (Chinese version)
└── README_en.md                   # English version README
```

## 🚀 Quick Start

### Install Dependencies

```bash
pip install pandas numpy openpyxl
```

### Run Data Preprocessing

```bash
# Recommended: balanced version (three-key strategy)
python preprocess_data_balanced.py

# Or use simplified version (two-key strategy)
python preprocess_data_tiny_version.py
```

### Use Jupyter Notebook for Step-by-Step Execution

```bash
# Open Jupyter Notebook
jupyter notebook 数据预处理步骤指南.ipynb
```

### Output File Locations

After successful execution, preprocessed data will be saved in:
- **Main file**: `Insight_output/1-preprocessed.csv` (contains Indcd and isST fields)
- **Log**: `Insight_output/preprocess_log_balanced.txt`
- **Quality report**: `Insight_output/质量报告_最终版.md`
- **Completion summary**: `Insight_output/完成总结_最终版.md`

## 📊 Data Statistics

### Final Output Data

| Metric | Value |
|--------|-------|
| **Sample Size** | 57,621 records |
| **Number of Companies** | 3,757 companies |
| **Time Span** | 2010-2019 (10 years) |
| **Number of Features** | 42 columns (6 keys/labels + 36 financial indicators) |
| **Violation Samples** | 2,989 (5.19%) |
| **ST Samples** | 11,301 (19.61%) |
| **File Size** | 16.87 MB |

### Report Type Distribution

- **Type A (Annual Report)**: 29,025 records (50.37%)
- **Type B (Semi-annual Report)**: 28,596 records (49.63%)

## 📖 New Fields Description

### 1. Indcd (Industry Code)

**Field Meaning**: CSRC Industry Classification 2012 Edition

**Data Source**: Solvency table (FI_T1.xlsx)

**Coverage**: 98.03% (56,486/57,621 records)

**Unique Values**: 80 different industry codes

**Example Values**: `K70` (Real Estate), `C27` (Pharmaceutical Manufacturing), `S90` (Comprehensive), etc.

**Usage**: 
- Industry-based financial fraud feature analysis
- Control for industry effects
- Industry comparison analysis

**Usage Example**:
```python
import pandas as pd
df = pd.read_csv('Insight_output/1-preprocessed.csv')

# Group by industry and calculate ST ratio
st_by_industry = df.groupby('Indcd').agg({
    'isST': 'mean',
    'Stkcd': 'count'
}).round(4)
print("ST ratio by industry:")
print(st_by_industry)
```

### 2. isST (ST Warning Marker)

**Field Meaning**: Marks whether a company should be ST (Special Treatment)

**Values**: 
- `0`: Normal, does not meet ST conditions
- `1`: Should be ST, meets ST judgment rules

**Judgment Rules (Based on 2024 Revised Rules)**:

#### Rule 1: Insolvency
- **Condition**: Asset-liability ratio > 100% (i.e., negative net assets)
- **Corresponding Indicator**: `F011201A` > 1.0
- **Basis**: Main board companies with negative net assets at the end of the most recent year trigger *ST

#### Rule 2: Negative Net Profit + Violation Record
- **Condition**: ROA < 0 and isviolation = 1
- **Corresponding Indicator**: `F050204C` < 0 with violation record
- **Basis**: Poor financial performance combined with violation behavior increases ST risk

#### Rule 3: Cumulative Violation Count
- **Condition**: Same company with cumulative violations ≥ 2
- **Calculation Method**: Group by company, accumulate violation counts
- **Basis**: Companies with consecutive violations are more likely to be ST

**Statistical Results**:
- ST samples: 11,301 (19.61%)
- Non-ST samples: 46,320 (80.39%)

**ST Sample Composition**:
- ST and violation: 2,144 (3.72%) - Both financial anomalies and violations
- ST but no violation: 9,157 (15.89%) - Only financial anomalies (e.g., insolvency)

**Limitations**:

Due to missing key fields in the dataset, complete ST judgment rules cannot be implemented:
1. ❌ Absolute revenue (cannot judge "negative net profit and revenue < 300 million")
2. ❌ Audit opinion type (cannot judge "unable to express opinion" or "adverse opinion")
3. ❌ Market value and stock price data (cannot judge "market value < 500 million or stock price < 1 yuan for 20 consecutive days")
4. ❌ Capital occupation, illegal guarantee and other regulatory indicators
5. ❌ Cash dividend details (cannot judge "dividend non-compliance")

Therefore, this script uses **simplified rules** to generate isST labels based on existing financial indicators and violation information, serving as a **proxy variable** for ST risk.

## 📋 Indicator Completeness Check Report

### 12 Indicators Checked

1. Chairman and General Manager Concurrent Position
2. Number of Four Committees
3. Supervisor Shareholding Ratio
4. State-owned Share Ratio
5. Whether ST
6. Management Shareholding Ratio
7. Executive Shareholding Ratio
8. Executive Average Education Background
9. Audit Opinion Type
10. Equity Concentration Index%
11. Herfindahl_5 Index
12. Z Index

### Final Check Results

#### ✅ Exists and Integrated (1)

**Industry Category (Indcd)**
- Location: Solvency table (FI_T1.xlsx)
- Field Name: `Indcd [Industry Code]` (CSRC Industry Classification 2012 Edition)
- Status: ✅ Successfully extracted and integrated into final output
- Coverage: 98.03%

#### ❌ Completely Missing (11)

After deep inspection (including column names, data content, keyword matching), the following 11 indicators **completely do not exist** in all 9 Excel files in the Dataset:

**Opportunity Indicators - Governance Structure (4)**
1. ❌ **Chairman and General Manager Concurrent Position** - Not found
2. ❌ **Number of Four Committees** - Not found
3. ❌ **Supervisor Shareholding Ratio** - Not found
4. ❌ **State-owned Share Ratio** - Not found

**Opportunity Indicators - Equity Structure (3)**
5. ❌ **Equity Concentration Index%** - Not found
6. ❌ **Herfindahl_5 Index** - Not found
7. ❌ **Z Index** - Not found

**Rationalization Indicators (2)**
8. ❌ **Executive Average Education Background** - Not found
9. ❌ **Audit Opinion Type** - Not found

**Management Financial Status (2)**
10. ❌ **Management Shareholding Ratio** - Not found
11. ❌ **Executive Shareholding Ratio** - Not found

**External Pressure (1)**
12. ❌ **Whether ST** - Not found (isST generated through calculation as proxy variable)

### Check Coverage

✅ Checked all 9 Excel files:
- Solvency table (FI_T1.xlsx)
- Disclosed financial indicators table (FI_T2.xlsx)
- Operating capability table (FI_T4.xlsx)
- Profitability table (FI_T5.xlsx)
- Risk level table (FI_T7.xlsx)
- Growth capability table (FI_T8.xlsx)
- Per-share indicators table (FI_T9.xlsx)
- Dividend distribution table (FI_T11.xlsx)
- Violation information table (STK_Violation_Main.xlsx)

✅ Check Methods:
- DES description files
- Excel column names
- Data content (first 100 rows)
- Keyword matching (Chinese, English, abbreviations, synonyms)

### Conclusion

**Total 12 indicators:**
- ✅ Exists and integrated: 1 (Industry category Indcd)
- ❌ Completely missing: 11

**Missing Rate: 91.7%** (11/12)

These 11 missing indicators need to be obtained from external data sources or calculated:
- Governance structure and equity structure indicators: Require corporate governance data and shareholder holding details
- Management and executive shareholding: Require executive holding details
- Audit opinion: Require annual report audit data
- ST marker: Require exchange announcements or company basic information (isST generated through financial indicators)
- Executive education background: Require executive personal information data

## 🔧 Technical Implementation

### Core Strategy

1. **Three-key Primary Key Strategy**: Uses (Stkcd, Year, Typrep) as primary keys
   - Stkcd: Standardized to 6-digit string
   - Year: Extracted from Accper date field
   - Typrep: Original value retained, priority K > C > A > B

2. **Intelligent Deduplication**:
   - Sort by Typrep priority
   - Keep only one record per (Stkcd, Year, Typrep) combination
   - Avoid data explosion (stabilized from 200k+ rows to 57k rows)

3. **Chinese Column Name Handling**:
   - Automatically identify Chinese column names in solvency tables
   - Dynamically map to standard English column names

4. **Data Integration**:
   - Use outer join to retain all valid data
   - Deduplicate each table individually before merging
   - Handle tables without Typrep (e.g., disclosed financial indicators table)

### Code Structure

```python
class FinancialDataPreprocessor:
    def __init__(self):          # Initialize configuration
    def read_excel_safe(self):   # Safely read Excel
    def standardize_stock_code(self):  # Standardize stock code
    def extract_year(self):      # Extract year
    def load_and_prepare_table(self):  # Load and normalize single table
    def merge_financial_tables(self):  # Data integration
    def add_violation_label(self):     # Generate violation label
    def clean_data(self):        # Data cleaning
    def generate_st_label(self): # Generate ST label
    def prepare_final_output(self):    # Prepare final output
    def quality_check(self):     # Quality validation
    def save_output(self):       # Export file
```

## 📈 Data Quality Metrics

### Completeness Check
- Primary key completeness: 100%
- Data volume: 57,621 records
- Number of companies: 3,757 companies
- Year range: 2010-2019

### Consistency Check
- Reasonable report type distribution (A: 50.37%, B: 49.63%)
- Violation ratio: 5.19% (reasonable range: 3-10%)
- ST ratio: 19.61% (based on simplified rules)

### Reasonableness Check
- Missing value statistics: Most columns have missing rate < 15%
- High missing columns: 10 columns with missing rate > 10%, mainly business-reasonable missing
- Data types: All indicator columns converted to numeric types

## ⚠️ Important Notes

1. **isST Field Description**:
   - 19.61% ST ratio is slightly higher than actual market (approximately 5-10%)
   - Reason: Uses simplified rules and includes "should be ST" rather than "already ST"
   - Recommendation: Use as risk indicator, not actual ST status

2. **Indcd Missing Value Handling**:
   - 1.97% of samples have empty Indcd
   - Recommendation: Fill with "Unknown Industry" or supplement based on company's historical industry codes when using

3. **Time Series Characteristics**:
   - ST markers consider cumulative violation counts, have time series dependency
   - Recommendation: Pay attention to time order when training models

4. **Missing Indicators**:
   - 11 indicators are completely missing in current Dataset
   - Complete analysis requires supplementing from external data sources

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributions

Contributions through Issues and Pull Requests are welcome.

## 📚 Citation

If you use this project in your research, please cite it as follows:

```bibtex
@misc{financial-fraud-detection-preprocessing,
  author = {HIT Jimmy Xiao},
  title = {Financial Fraud Detection Implementation Project - Data Preprocessing},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project}
}
```

## 📧 Contact Information

For any questions or suggestions, please contact through:

- GitHub Issues: https://github.com/HIT-JimmyXiao/Financial_Fraud_Detection_Implementation_Project/issues

---

**Last Updated**: 2025-11-08  
**Project Status**: ✅ Data preprocessing completed

