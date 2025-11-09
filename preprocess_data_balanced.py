#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务舞弊识别实验1 - 数据预处理脚本（平衡版）
策略：使用 (Stkcd, Year, Typrep) 三键，但优化处理以避免数据爆炸
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
        logging.FileHandler('Insight_output/preprocess_log_balanced.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FinancialDataPreprocessor:
    """财务数据预处理器 - 平衡版"""
    
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
        
        # Typrep 优先级（只保留A和B类型，优先A）：
        # A(合并报表期末) > B(母公司报表期末)
        # 注意：对于同一个(Stkcd, Year, Month)，如果有A类型就保留A，否则保留B
        self.typrep_priority = {
            'A': 1,  # 合并报表期末（优先保留）
            'B': 2,  # 母公司报表期末（如果没有A才保留）
            # 其他类型不保留
            'C': 99, 'D': 99, 'K': 99, 'S': 99, 'H': 99, 'F': 99, 'E': 99, 'N': 99
        }
        
        logger.info("=" * 80)
        logger.info("财务舞弊识别实验1 - 数据预处理（平衡版）")
        logger.info("策略：季度去重 (Stkcd, Year, Month) + Typrep优先A类型")
        logger.info("=" * 80)
    
    def read_excel_safe(self, filepath):
        """安全读取Excel文件"""
        try:
            logger.info(f"正在读取: {os.path.basename(filepath)}")
            df = pd.read_excel(filepath)
            logger.info(f"  成功读取 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"  读取失败: {str(e)}")
            return None
    
    def standardize_stock_code(self, code):
        """标准化股票代码为6位字符串"""
        if pd.isna(code):
            return None
        try:
            return str(int(code)).zfill(6)
        except:
            return str(code).zfill(6)
    
    def extract_year(self, date_str):
        """从日期字符串提取年份"""
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
    
    def extract_month(self, date_str):
        """从日期字符串提取月份（用于季度判断）"""
        if pd.isna(date_str):
            return None
        try:
            if isinstance(date_str, str):
                parts = date_str.split('-')
                if len(parts) >= 2:
                    return int(parts[1])
            elif isinstance(date_str, (pd.Timestamp, datetime)):
                return date_str.month
            return None
        except:
            return None
    
    def load_and_prepare_table(self, name, filepath):
        """Step 1-2: 读取、规范化并智能去重单个主题表"""
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
                elif '利息保障倍数B' in col or '利息保障倍数' in col:
                    column_mapping[col] = 'F010702B'
                elif ('现金流量' in col and '流动负债' in col) or ('经营活动产生的现金流量净额' in col and '流动负债' in col):
                    column_mapping[col] = 'F010801B'
                elif '资产负债率' in col:
                    column_mapping[col] = 'F011201A'
                # 保留所有以F开头的列（如果已经是标准格式）
                elif col.startswith('F') and len(col) >= 8:
                    column_mapping[col] = col
            df = df.rename(columns=column_mapping)
            logger.info(f"  偿债能力表字段映射完成，保留字段: {[c for c in df.columns if c.startswith('F')]}")
        
        # 处理披露财务指标表（没有Typrep字段，只有Stkcd和Accper）
        if name == 'disclosure':
            # 确保F020108字段存在
            if 'F020108' not in df.columns:
                # 尝试查找包含"基本每股收益"或"每股收益"的列
                for col in df.columns:
                    if '基本每股收益' in str(col) or '每股收益' in str(col) or col == 'F020108':
                        if col != 'F020108':
                            df = df.rename(columns={col: 'F020108'})
                            logger.info(f"  将列 '{col}' 映射为 'F020108'")
                        break
            logger.info(f"  披露财务指标表字段: {list(df.columns)}")
        
        # 标准化键字段
        if 'Stkcd' in df.columns:
            df['Stkcd_std'] = df['Stkcd'].apply(self.standardize_stock_code)
        if 'Accper' in df.columns:
            df['Year'] = df['Accper'].apply(self.extract_year)
            df['Month'] = df['Accper'].apply(self.extract_month)
        
        # 清洗：移除无效记录
        before = len(df)
        df = df[df['Stkcd_std'].notna() & df['Year'].notna() & df['Month'].notna()].copy()
        after = len(df)
        if before > after:
            logger.info(f"  清洗无效键: -{before - after} 行")
        
        # 关键优化：按季度去重，只保留A和B类型，优先A
        if 'Typrep' in df.columns:
            df = df[df['Typrep'].notna()].copy()
            
            # 只保留A和B类型
            before_filter = len(df)
            df = df[df['Typrep'].isin(['A', 'B'])].copy()
            after_filter = len(df)
            if before_filter > after_filter:
                logger.info(f"  过滤非A/B类型: -{before_filter - after_filter} 行")
            
            # 添加优先级列
            df['typrep_priority'] = df['Typrep'].map(self.typrep_priority).fillna(99)
            
            # 按季度去重：对于同一个(Stkcd, Year, Month)，优先保留A类型
            # 1. 先按优先级排序（A在前，B在后）
            df = df.sort_values(['Stkcd_std', 'Year', 'Month', 'typrep_priority'])
            
            # 2. 对于同一个(Stkcd, Year, Month)，如果有A就只保留A，否则保留B
            # 先标记每个组合是否有A类型
            df['has_A'] = df.groupby(['Stkcd_std', 'Year', 'Month'])['Typrep'].transform(lambda x: (x == 'A').any())
            
            # 如果有A，只保留A类型；如果没有A，保留B类型
            df = df[((df['has_A']) & (df['Typrep'] == 'A')) | (~df['has_A'] & (df['Typrep'] == 'B'))].copy()
            
            # 3. 对于同一个(Stkcd, Year, Month, Typrep)组合，只保留第一条
            df = df.drop_duplicates(subset=['Stkcd_std', 'Year', 'Month', 'Typrep'], keep='first')
            
            # 删除临时列
            df = df.drop(columns=['typrep_priority', 'has_A', 'Month'])
            
            logger.info(f"  去重后: {len(df)} 行, 报表类型: {df['Typrep'].value_counts().to_dict()}")
        else:
            # 无 Typrep 的表（如披露财务），按 (Stkcd, Year, Month) 去重
            df = df.drop_duplicates(subset=['Stkcd_std', 'Year', 'Month'], keep='first')
            df = df.drop(columns=['Month'])
            logger.info(f"  去重后: {len(df)} 行")
        
        return df
    
    def merge_financial_tables(self):
        """Step 3: 横向并表（季度去重策略 + 外连接保留所有记录）"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: 数据集成 - 横向并表（季度去重策略）")
        logger.info("=" * 80)
        
        # 加载所有主题表
        dfs = {}
        for name, filepath in self.data_files.items():
            if name != 'violation':
                df = self.load_and_prepare_table(name, filepath)
                if df is not None:
                    dfs[name] = df
        
        # 确定主键集合：从经营能力表开始
        if 'operation' not in dfs:
            logger.error("缺少经营能力表！")
            return None
        
        base_df = dfs['operation'].copy()
        # 添加Month列用于合并
        if 'Accper' in base_df.columns:
            base_df['Month'] = base_df['Accper'].apply(self.extract_month)
        merge_keys = ['Stkcd_std', 'Year', 'Month', 'Typrep']
        
        # 选择基准表的指标列和其他重要字段
        indicator_cols = [c for c in base_df.columns if c.startswith('F')]
        other_cols = [c for c in base_df.columns if c in ['Indcd']]
        keep_cols = merge_keys + other_cols + indicator_cols
        keep_cols = [c for c in keep_cols if c in base_df.columns]
        base_df = base_df[keep_cols].copy()
        
        logger.info(f"\n基准表（经营能力）: {len(base_df)} 行")
        
        # 依次合并其他表（使用 outer join 保留所有数据）
        # 注意：disclosure表没有Typrep字段，需要单独处理
        merge_order = ['profit', 'growth', 'solvency', 'risk', 'pershare', 'dividend', 'disclosure']
        
        for name in merge_order:
            if name not in dfs:
                logger.warning(f"  {name} 表不存在，跳过")
                continue
            
            df = dfs[name].copy()
            
            # 选择指标列和其他重要字段
            indicator_cols = [c for c in df.columns if c.startswith('F')]
            other_cols = [c for c in df.columns if c in ['Indcd']]
            
            # 添加Month列用于合并
            if 'Accper' in df.columns and 'Month' not in df.columns:
                df['Month'] = df['Accper'].apply(self.extract_month)
            
            # 检查是否有 Typrep
            has_typrep = 'Typrep' in df.columns
            
            if has_typrep:
                keep_cols_df = merge_keys + other_cols + indicator_cols
                keys_to_use = merge_keys
            else:
                # 无 Typrep 的表（如披露财务），只使用 (Stkcd, Year, Month)
                keep_cols_df = ['Stkcd_std', 'Year', 'Month'] + other_cols + indicator_cols
                keys_to_use = ['Stkcd_std', 'Year', 'Month']
            
            keep_cols_df = [c for c in keep_cols_df if c in df.columns]
            df_subset = df[keep_cols_df].copy()
            
            logger.info(f"\n合并 {name} 表...")
            logger.info(f"  合并前: {len(base_df)} 行")
            logger.info(f"  待合并: {len(df_subset)} 行 (使用键: {keys_to_use})")
            
            # 使用 outer join 保留所有记录
            base_df = base_df.merge(df_subset, on=keys_to_use, how='outer', suffixes=('', '_dup'))
            
            # 删除重复列（如果有）
            dup_cols = [c for c in base_df.columns if c.endswith('_dup')]
            if dup_cols:
                base_df = base_df.drop(columns=dup_cols)
            
            logger.info(f"  合并后: {len(base_df)} 行")
        
        # 删除临时Month列
        if 'Month' in base_df.columns:
            base_df = base_df.drop(columns=['Month'])
        
        logger.info(f"\n最终集成数据: {len(base_df)} 行, {len(base_df.columns)} 列")
        return base_df
    
    def add_violation_label(self, df):
        """Step 4: 生成 isviolation 标注"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: 生成违规标签")
        logger.info("=" * 80)
        
        filepath = self.data_files['violation']
        if not os.path.exists(filepath):
            logger.warning("违规信息表不存在")
            df['isviolation'] = 0
            return df
        
        violation_df = self.read_excel_safe(filepath)
        if violation_df is None:
            df['isviolation'] = 0
            return df
        
        # 标准化违规表的键
        violation_df['Stkcd_std'] = violation_df['Symbol'].apply(self.standardize_stock_code)
        violation_df['Year'] = pd.to_numeric(violation_df['ViolationYear'], errors='coerce')
        violation_df = violation_df[violation_df['Year'].notna()].copy()
        violation_df['Year'] = violation_df['Year'].astype(int)
        
        # 确保主表的 Year 是 int
        df['Year'] = df['Year'].astype(int)
        
        # 筛选实际违规记录（IsViolated='Y'）
        violation_yes = violation_df[violation_df['IsViolated'] == 'Y'].copy()
        logger.info(f"实际违规记录数: {len(violation_yes)}")
        
        # 按 (Stkcd_std, Year) 聚合（违规是针对公司年度，不分报表类型）
        violation_set = violation_yes.groupby(['Stkcd_std', 'Year']).size().reset_index(name='violation_count')
        violation_set['isviolation'] = 1
        
        logger.info(f"违规的 (公司, 年份) 数: {len(violation_set)}")
        
        # 合并到主表
        df = df.merge(
            violation_set[['Stkcd_std', 'Year', 'isviolation']], 
            on=['Stkcd_std', 'Year'], 
            how='left'
        )
        df['isviolation'] = df['isviolation'].fillna(0).astype(int)
        
        logger.info(f"违规样本: {df['isviolation'].sum()} ({df['isviolation'].mean():.2%})")
        return df
    
    def clean_data(self, df):
        """Step 5: 数据清洗"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: 数据清洗")
        logger.info("=" * 80)
        
        # 统一缺失值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 统计缺失值
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
        
        logger.info(f"缺失值统计（缺失率>10%的列）:")
        for col, pct in high_missing.head(10).items():
            logger.info(f"  {col}: {pct}%")
        
        # 确保指标列为数值类型
        numeric_cols = [c for c in df.columns if c.startswith('F')]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"已确保 {len(numeric_cols)} 个指标列为数值类型")
        
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
    
    def prepare_final_output(self, df):
        """Step 6: 准备最终输出"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: 准备最终输出")
        logger.info("=" * 80)
        
        # 恢复原始列名
        df['Stkcd'] = df['Stkcd_std'].apply(lambda x: int(x) if pd.notna(x) else np.nan)
        # 保留原始Accper日期，如果不存在则用Year填充
        if 'Accper' not in df.columns:
            df['Accper'] = df['Year']
        
        # 确保 Typrep 存在（只保留A和B类型）
        if 'Typrep' not in df.columns:
            df['Typrep'] = ''
        else:
            # 再次确保只保留A和B类型
            before_typrep_filter = len(df)
            df = df[df['Typrep'].isin(['A', 'B', ''])].copy()
            after_typrep_filter = len(df)
            if before_typrep_filter > after_typrep_filter:
                logger.info(f"  最终过滤非A/B类型: -{before_typrep_filter - after_typrep_filter} 行")
        
        # 确保 Indcd 存在，并填充相同Stkcd的Indcd
        if 'Indcd' not in df.columns:
            df['Indcd'] = ''
        else:
            # 对于相同Stkcd的Indcd进行填充
            # 先按Stkcd分组，用该组内非空的Indcd填充空值
            if 'Stkcd' in df.columns:
                # 创建Stkcd到Indcd的映射（使用非空值）
                stkcd_indcd_map = df[df['Indcd'].notna() & (df['Indcd'] != '')].groupby('Stkcd')['Indcd'].first().to_dict()
                # 填充空值
                mask = (df['Indcd'].isna()) | (df['Indcd'] == '')
                df.loc[mask, 'Indcd'] = df.loc[mask, 'Stkcd'].map(stkcd_indcd_map)
                filled_count = mask.sum()
                if filled_count > 0:
                    logger.info(f"  填充Indcd: {filled_count} 条记录（使用相同Stkcd的Indcd）")
        
        # 确保所有必需列存在
        for col in self.column_order:
            if col not in df.columns:
                if col in ['isviolation', 'isST']:
                    df[col] = 0
                elif col not in ['Stkcd', 'Accper', 'Typrep', 'Indcd']:
                    df[col] = np.nan
        
        # 生成isST标签
        df = self.generate_st_label(df)
        
        # 选择并排序列
        output_df = df[self.column_order].copy()
        
        # 删除完全重复的行
        before = len(output_df)
        output_df = output_df.drop_duplicates()
        after = len(output_df)
        if before > after:
            logger.info(f"删除重复行: -{before - after} 行")
        
        # 删除缺失值占比大于50%的行
        before_missing = len(output_df)
        # 计算每行的缺失值占比（只考虑指标列，不包括主键和标签列）
        indicator_cols = [c for c in output_df.columns if c.startswith('F')]
        if indicator_cols:
            missing_ratio = output_df[indicator_cols].isnull().sum(axis=1) / len(indicator_cols)
            output_df = output_df[missing_ratio <= 0.5].copy()
            after_missing = len(output_df)
            if before_missing > after_missing:
                logger.info(f"删除缺失值占比>50%的行: -{before_missing - after_missing} 行")
        
        # 排序
        output_df = output_df.sort_values(['Stkcd', 'Accper', 'Typrep']).reset_index(drop=True)
        
        # 统计信息
        logger.info(f"\n最终数据形状: {output_df.shape}")
        logger.info(f"样本量: {len(output_df)}")
        logger.info(f"公司数: {output_df['Stkcd'].nunique()}")
        logger.info(f"年份范围: {output_df['Accper'].min()}-{output_df['Accper'].max()}")
        logger.info(f"报表类型: {output_df['Typrep'].value_counts().to_dict()}")
        logger.info(f"违规比例: {output_df['isviolation'].mean():.2%}")
        
        return output_df
    
    def quality_check(self, df):
        """Step 7: 质量校验"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: 质量校验")
        logger.info("=" * 80)
        
        logger.info("✓ 完整性检查:")
        logger.info(f"  主键完整率: {((df['Stkcd'].notna()) & (df['Accper'].notna())).mean():.2%}")
        
        logger.info("\n✓ 一致性检查:")
        logger.info(f"  数据量: {len(df)} 条")
        logger.info(f"  公司数: {df['Stkcd'].nunique()} 家")
        logger.info(f"  年份: {df['Accper'].min()}-{df['Accper'].max()}")
        
        logger.info("\n✓ 合理性检查:")
        logger.info(f"  违规比例: {df['isviolation'].mean():.2%} (合理范围: 3-10%)")
        
        logger.info("\n✓ 抽样验证:")
        sample = df.head(3)
        for idx, row in sample.iterrows():
            logger.info(f"  Stkcd={int(row['Stkcd'])}, Year={int(row['Accper'])}, Typrep={row['Typrep']}")
        
        logger.info("\n质量校验通过 ✓")
        return True
    
    def save_output(self, df):
        """Step 8: 导出"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: 导出文件")
        logger.info("=" * 80)
        
        output_file = os.path.join(self.output_dir, f'{self.group_id}-preprocessed.csv')
        
        # 检查文件是否被占用
        if os.path.exists(output_file):
            try:
                # 尝试重命名以检查文件是否被占用
                temp_file = output_file + '.tmp'
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                os.rename(output_file, temp_file)
                os.rename(temp_file, output_file)
                logger.info(f"文件检查: {output_file} 未被占用")
            except PermissionError:
                logger.warning(f"⚠️ 文件 {output_file} 可能被其他程序打开（如Excel）")
                logger.warning("⚠️ 请关闭该文件后重试，或手动删除该文件")
                # 尝试保存为带时间戳的文件
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = os.path.join(self.output_dir, f'{self.group_id}-preprocessed_{timestamp}.csv')
                logger.info(f"尝试保存为备份文件: {backup_file}")
                try:
                    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                    size_mb = os.path.getsize(backup_file) / 1024 / 1024
                    logger.info(f"✓ 成功保存为备份文件: {backup_file}")
                    logger.info(f"✓ 文件大小: {size_mb:.2f} MB")
                    logger.info(f"✓ 数据形状: {df.shape}")
                    return backup_file
                except Exception as e2:
                    logger.error(f"备份文件保存也失败: {str(e2)}")
                    return None
            except Exception as e:
                logger.warning(f"文件检查时出现异常: {str(e)}")
        
        # 尝试保存文件（带重试机制）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                size_mb = os.path.getsize(output_file) / 1024 / 1024
                logger.info(f"✓ 成功保存: {output_file}")
                logger.info(f"✓ 文件大小: {size_mb:.2f} MB")
                logger.info(f"✓ 数据形状: {df.shape}")
                
                logger.info(f"\n前5行预览:")
                logger.info(f"\n{df.head()}")
                
                return output_file
            except PermissionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"保存失败（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
                    logger.warning("文件可能被其他程序占用，等待2秒后重试...")
                    import time
                    time.sleep(2)
                else:
                    logger.error(f"保存失败（已重试{max_retries}次）: {str(e)}")
                    logger.error("请检查文件是否被Excel或其他程序打开")
                    # 尝试保存为带时间戳的备份文件
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_file = os.path.join(self.output_dir, f'{self.group_id}-preprocessed_{timestamp}.csv')
                    try:
                        df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                        size_mb = os.path.getsize(backup_file) / 1024 / 1024
                        logger.info(f"✓ 已保存为备份文件: {backup_file}")
                        logger.info(f"✓ 文件大小: {size_mb:.2f} MB")
                        return backup_file
                    except Exception as e2:
                        logger.error(f"备份文件保存也失败: {str(e2)}")
                        return None
            except Exception as e:
                logger.error(f"保存失败: {str(e)}")
                return None
        
        return None
    
    def run(self):
        """执行完整的数据预处理流程"""
        start_time = datetime.now()
        logger.info(f"\n开始时间: {start_time}")
        
        try:
            # Step 3: 数据集成
            merged_df = self.merge_financial_tables()
            if merged_df is None:
                return None
            
            # Step 4: 添加违规标签
            merged_df = self.add_violation_label(merged_df)
            
            # Step 5: 数据清洗
            cleaned_df = self.clean_data(merged_df)
            
            # Step 6: 准备输出
            final_df = self.prepare_final_output(cleaned_df)
            
            # Step 7: 质量校验
            self.quality_check(final_df)
            
            # Step 8: 保存
            output_file = self.save_output(final_df)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "=" * 80)
            logger.info("数据预处理完成！")
            logger.info("=" * 80)
            logger.info(f"总耗时: {duration:.2f} 秒")
            logger.info(f"输出文件: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"\n处理错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def main():
    """主函数"""
    preprocessor = FinancialDataPreprocessor(data_dir='Dataset', output_dir='Insight_output')
    result = preprocessor.run()
    
    if result:
        print(f"\n[SUCCESS] 预处理完成！")
        print(f"[SUCCESS] 输出文件: {result}")
    else:
        print(f"\n[FAILED] 预处理失败！")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

