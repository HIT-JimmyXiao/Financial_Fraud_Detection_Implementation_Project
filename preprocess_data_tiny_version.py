#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务舞弊识别实验1 - 数据预处理脚本 Tiny版（极简策略）
策略：只使用 (Stkcd, Year) 作为主键，每个公司每年只保留一条记录
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Insight_output/preprocess_log_tiny_version.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FinancialDataPreprocessor:
    """财务数据预处理器 - Tiny版"""
    
    def __init__(self, data_dir='Dataset', output_dir='Insight_output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.group_id = '13'
        
        self.data_files = {
            'solvency': os.path.join(data_dir, '偿债能力', 'FI_T1.xlsx'),
            'disclosure': os.path.join(data_dir, '披露财务指标', 'FI_T2.xlsx'),
            'operation': os.path.join(data_dir, '经营能力', 'FI_T4.xlsx'),
            'profit': os.path.join(data_dir, '盈利能力', 'FI_T5.xlsx'),
            'risk': os.path.join(data_dir, '风险水平', 'FI_T7.xlsx'),
            'growth': os.path.join(data_dir, '发展能力', 'FI_T8.xlsx'),
            'pershare': os.path.join(data_dir, '每股指标', 'FI_T9.xlsx'),
            'dividend': os.path.join(data_dir, '股利分配', 'FI_T11.xlsx'),
            'violation': os.path.join(data_dir, '违规信息总表', 'STK_Violation_Main.xlsx')
        }
        
        self.column_order = [
            'Stkcd', 'Accper', 'Typrep', 'Indcd', 'isviolation', 'isST',
            # 偿债能力（FI_T1.xlsx）
            'F010101A', 'F010201A', 'F010702B', 'F010801B', 'F011201A',
            # 经营能力（FI_T4.xlsx）
            'F040101B', 'F040202B', 'F040203B', 'F040205C', 'F040401B', 'F040503B', 'F040505C',
            'F040803B', 'F040805C', 'F041203B', 'F041205C', 'F041301B', 'F041403B', 'F041405C',
            'F041703B', 'F041705C', 'F041803B', 'F041805C',
            # 盈利能力（FI_T5.xlsx）
            'F050104C', 'F050204C', 'F053201B', 'F053301C', 'F052401B', 'F053202B',
            # 发展能力（FI_T8.xlsx）
            'F080102A', 'F081002B', 'F082601B', 'F080603A',
            # 风险水平（FI_T7.xlsx）
            'F070101B', 'F070201B', 'F070301B',
            # 每股指标（FI_T9.xlsx）
            'F090102B',
            # 披露财务（FI_T2.xlsx）
            'F020108',
            # 股利分配（FI_T11.xlsx）
            'F110101B', 'F110301B', 'F110801B'
        ]
        
        logger.info("=" * 80)
        logger.info("财务舞弊识别实验1 - 数据预处理 Tiny版")
        logger.info("策略：只使用 (Stkcd, Year) 作为主键，优先选择合并报表")
        logger.info("=" * 80)
    
    def read_excel_safe(self, filepath):
        try:
            logger.info(f"正在读取: {os.path.basename(filepath)}")
            df = pd.read_excel(filepath)
            logger.info(f"  成功读取 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"  读取失败: {str(e)}")
            return None
    
    def standardize_stock_code(self, code):
        if pd.isna(code):
            return None
        try:
            return str(int(code)).zfill(6)
        except:
            return str(code).zfill(6)
    
    def extract_year(self, date_str):
        if pd.isna(date_str):
            return None
        try:
            if isinstance(date_str, str):
                return int(date_str.split('-')[0])
            elif isinstance(date_str, (pd.Timestamp, datetime)):
                return date_str.year
            else:
                return int(date_str)
        except:
            return None
    
    def load_table(self, name, filepath):
        """加载单个表并进行去重"""
        if not os.path.exists(filepath):
            logger.warning(f"{name} 文件不存在")
            return None
        
        df = self.read_excel_safe(filepath)
        if df is None:
            return None
        
        # 处理偿债能力表的中文列名
        if name == 'solvency':
            df.columns = [col.strip().strip("'") for col in df.columns]
            column_mapping = {}
            for col in df.columns:
                if '股票代码' in col:
                    column_mapping[col] = 'Stkcd'
                elif '截止日期' in col:
                    column_mapping[col] = 'Accper'
                elif '报表类型编码' in col:
                    column_mapping[col] = 'Typrep'
                elif '行业代码' in col or col == 'Indcd':
                    column_mapping[col] = 'Indcd'
                elif '流动比率' in col:
                    column_mapping[col] = 'F010101A'
                elif '速动比率' in col:
                    column_mapping[col] = 'F010201A'
                elif '利息保障倍数B' in col:
                    column_mapping[col] = 'F010702B'
                elif '现金流量' in col and '流动负债' in col:
                    column_mapping[col] = 'F010801B'
                elif '资产负债率' in col:
                    column_mapping[col] = 'F011201A'
            df = df.rename(columns=column_mapping)
        
        # 标准化主键
        if 'Stkcd' in df.columns:
            df['Stkcd_std'] = df['Stkcd'].apply(self.standardize_stock_code)
        if 'Accper' in df.columns:
            df['Year'] = df['Accper'].apply(self.extract_year)
        
        # Typrep 优先级（基于CSMAR标准）：A(合并报表期末) > B(母公司报表期末) > C(合并报表期初) > D(母公司报表期初) > 其他
        if 'Typrep' in df.columns:
            typrep_priority = {
                'A': 1,  # 合并报表期末（最常用）
                'B': 2,  # 母公司报表期末
                'C': 3,  # 合并报表期初
                'D': 4,  # 母公司报表期初
                'K': 5, 'S': 6, 'H': 7, 'F': 8, 'E': 9, 'N': 10
            }
            df['typrep_priority'] = df['Typrep'].map(typrep_priority).fillna(99)
        else:
            df['typrep_priority'] = 99
        
        # 按 (Stkcd_std, Year) 分组，选择优先级最高的记录
        df = df.sort_values(['Stkcd_std', 'Year', 'typrep_priority'])
        df = df.groupby(['Stkcd_std', 'Year'], as_index=False).first()
        
        logger.info(f"  去重后: {len(df)} 行")
        return df
    
    def merge_all_data(self):
        """合并所有数据"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 1-3: 数据集成")
        logger.info("=" * 80)
        
        # 加载所有表
        dfs = {}
        for name, filepath in self.data_files.items():
            if name != 'violation':
                df = self.load_table(name, filepath)
                if df is not None:
                    dfs[name] = df
        
        # 以经营能力表为基准
        if 'operation' not in dfs:
            logger.error("缺少经营能力表！")
            return None
        
        # 选择列时包含Indcd（如果存在）
        keep_cols = ['Stkcd_std', 'Year', 'Typrep']
        if 'Indcd' in dfs['operation'].columns:
            keep_cols.append('Indcd')
        keep_cols += [c for c in dfs['operation'].columns if c.startswith('F')]
        base_df = dfs['operation'][keep_cols].copy()
        logger.info(f"\n基准表: {len(base_df)} 行")
        
        # 依次合并其他表
        for name in ['profit', 'growth', 'solvency', 'risk', 'pershare', 'dividend', 'disclosure']:
            if name not in dfs:
                continue
            
            df = dfs[name]
            indicator_cols = [c for c in df.columns if c.startswith('F')]
            other_cols = [c for c in df.columns if c in ['Indcd']]
            keep_cols = ['Stkcd_std', 'Year'] + other_cols + indicator_cols
            keep_cols = [c for c in keep_cols if c in df.columns]
            
            df_subset = df[keep_cols].copy()
            
            logger.info(f"\n合并 {name}...")
            logger.info(f"  合并前: {len(base_df)} 行")
            base_df = base_df.merge(df_subset, on=['Stkcd_std', 'Year'], how='left')
            logger.info(f"  合并后: {len(base_df)} 行")
        
        return base_df
    
    def add_violation(self, df):
        """添加违规标签"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: 添加违规标签")
        logger.info("=" * 80)
        
        filepath = self.data_files['violation']
        if not os.path.exists(filepath):
            df['isviolation'] = 0
            return df
        
        violation_df = self.read_excel_safe(filepath)
        if violation_df is None:
            df['isviolation'] = 0
            return df
        
        violation_df['Stkcd_std'] = violation_df['Symbol'].apply(self.standardize_stock_code)
        # 尝试转换年份，忽略无法转换的
        violation_df['Year'] = pd.to_numeric(violation_df['ViolationYear'], errors='coerce')
        # 删除无法转换的行
        violation_df = violation_df[violation_df['Year'].notna()].copy()
        violation_df['Year'] = violation_df['Year'].astype(int)
        
        # 确保df的Year也是int类型
        df['Year'] = df['Year'].astype(int)
        
        violation_set = violation_df[violation_df['IsViolated'] == 'Y'].groupby(
            ['Stkcd_std', 'Year']).size().reset_index(name='count')
        violation_set['isviolation'] = 1
        
        df = df.merge(violation_set[['Stkcd_std', 'Year', 'isviolation']], 
                     on=['Stkcd_std', 'Year'], how='left')
        df['isviolation'] = df['isviolation'].fillna(0).astype(int)
        
        logger.info(f"违规样本: {df['isviolation'].sum()} ({df['isviolation'].mean():.2%})")
        return df
    
    def generate_st_label(self, df):
        """生成ST标签（基于财务异常指标）"""
        logger.info("\n生成ST标签...")
        
        # 初始化isST为0
        df['isST'] = 0
        
        # 规则1: 资不抵债（资产负债率>100%，即净资产为负）
        if 'F011201A' in df.columns:
            condition_insolvency = df['F011201A'] > 1.0  # 资产负债率>100%
        else:
            condition_insolvency = False
        
        # 规则2: 净利润为负（ROA为负）
        if 'F050204C' in df.columns:
            condition_negative_profit = df['F050204C'] < 0  # ROA<0表示净利润为负
        else:
            condition_negative_profit = False
        
        # 规则3: 有违规记录
        condition_violation = df['isviolation'] == 1
        
        # 综合判断：满足以下任一条件即标记为ST
        # 1. 资不抵债（净资产为负）
        # 2. 净利润为负 且 有违规记录
        # 3. 有重大违规（isviolation=1可视为重大违规）
        df.loc[condition_insolvency, 'isST'] = 1
        df.loc[condition_negative_profit & condition_violation, 'isST'] = 1
        
        # 特别处理：连续违规的公司更可能被ST
        # 按公司分组，计算累计违规次数
        if 'Stkcd_std' in df.columns and 'Year' in df.columns:
            df = df.sort_values(['Stkcd_std', 'Year'])
            df['cumulative_violations'] = df.groupby('Stkcd_std')['isviolation'].cumsum()
            # 累计违规>=2次的标记为ST
            df.loc[df['cumulative_violations'] >= 2, 'isST'] = 1
            df = df.drop(columns=['cumulative_violations'])
        
        st_count = df['isST'].sum()
        st_rate = df['isST'].mean()
        logger.info(f"  ST样本: {st_count} ({st_rate:.2%})")
        logger.info(f"  ST判断依据:")
        logger.info(f"    - 资不抵债（资产负债率>100%）")
        logger.info(f"    - 净利润为负 + 有违规记录")
        logger.info(f"    - 累计违规>=2次")
        
        return df
    
    def finalize(self, df):
        """最终清洗和输出"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 5-6: 最终处理")
        logger.info("=" * 80)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        df['Stkcd'] = df['Stkcd_std'].apply(lambda x: int(x) if pd.notna(x) else np.nan)
        df['Accper'] = df['Year']
        
        # 确保 Indcd 存在
        if 'Indcd' not in df.columns:
            df['Indcd'] = ''
        
        # 生成isST标签
        df = self.generate_st_label(df)
        
        # 确保所有列存在
        for col in self.column_order:
            if col not in df.columns:
                if col in ['isviolation', 'isST']:
                    df[col] = 0
                elif col not in ['Stkcd', 'Accper', 'Typrep', 'Indcd']:
                    df[col] = np.nan
        
        output_df = df[self.column_order].copy()
        output_df = output_df.drop_duplicates().reset_index(drop=True)
        output_df = output_df.sort_values(['Stkcd', 'Accper']).reset_index(drop=True)
        
        logger.info(f"最终数据形状: {output_df.shape}")
        logger.info(f"样本量: {len(output_df)}")
        logger.info(f"公司数: {output_df['Stkcd'].nunique()}")
        logger.info(f"年份范围: {output_df['Accper'].min()}-{output_df['Accper'].max()}")
        
        return output_df
    
    def save(self, df):
        """保存"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: 保存输出")
        logger.info("=" * 80)
        
        output_file = os.path.join(self.output_dir, f'{self.group_id}-preprocessed.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        logger.info(f"成功保存: {output_file}")
        logger.info(f"文件大小: {size_mb:.2f} MB")
        logger.info(f"\n前5行:\n{df.head()}")
        
        return output_file
    
    def run(self):
        """执行"""
        start = datetime.now()
        logger.info(f"开始时间: {start}")
        
        try:
            merged_df = self.merge_all_data()
            if merged_df is None:
                return None
            
            merged_df = self.add_violation(merged_df)
            final_df = self.finalize(merged_df)
            output_file = self.save(final_df)
            
            elapsed = (datetime.now() - start).total_seconds()
            logger.info("\n" + "=" * 80)
            logger.info(f"完成！耗时: {elapsed:.2f} 秒")
            logger.info("=" * 80)
            
            return output_file
        except Exception as e:
            logger.error(f"错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def main():
    preprocessor = FinancialDataPreprocessor(data_dir='Dataset', output_dir='Insight_output')
    result = preprocessor.run()
    
    if result:
        print(f"\n[SUCCESS] 完成！文件: {result}")
    else:
        print(f"\n[FAILED] 失败！")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())

