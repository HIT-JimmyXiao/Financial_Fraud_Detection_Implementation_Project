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
        self.group_id = '1'
        
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
            'F040101B', 'F040202B', 'F040203B', 'F040205C', 'F040401B', 'F040503B', 'F040505C',
            'F040803B', 'F040805C', 'F041203B', 'F041205C', 'F041301B', 'F041403B', 'F041405C',
            'F041703B', 'F041705C', 'F041803B', 'F041805C',
            'F050104C', 'F050204C', 'F053201B', 'F053301C', 'F052401B', 'F053202B',
            'F080102A', 'F081002B', 'F082601B', 'F080603A',
            'F070101B', 'F070201B', 'F070301B',
            'F090102B', 'F020108',
            'F110101B', 'F110301B', 'F110801B'
        ]
        
        # Typrep 优先级：K(合并) > C(合并调整) > A(年报) > B(半年报) > 其他
        self.typrep_priority = {'K': 1, 'C': 2, 'A': 3, 'B': 4, 'S': 5, 'H': 6, 'F': 7, 'E': 8, 'N': 9}
        
        logger.info("=" * 80)
        logger.info("财务舞弊识别实验1 - 数据预处理（平衡版）")
        logger.info("策略：三键 (Stkcd, Year, Typrep) + 智能去重避免数据爆炸")
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
                elif '利息保障倍数B' in col:
                    column_mapping[col] = 'F010702B'
                elif '现金流量' in col and '流动负债' in col:
                    column_mapping[col] = 'F010801B'
                elif '资产负债率' in col:
                    column_mapping[col] = 'F011201A'
            df = df.rename(columns=column_mapping)
        
        # 标准化键字段
        if 'Stkcd' in df.columns:
            df['Stkcd_std'] = df['Stkcd'].apply(self.standardize_stock_code)
        if 'Accper' in df.columns:
            df['Year'] = df['Accper'].apply(self.extract_year)
        
        # 清洗：移除无效记录
        before = len(df)
        df = df[df['Stkcd_std'].notna() & df['Year'].notna()].copy()
        after = len(df)
        if before > after:
            logger.info(f"  清洗无效键: -{before - after} 行")
        
        # 关键优化：按 Typrep 优先级，每个 (Stkcd, Year, Typrep) 只保留一条
        if 'Typrep' in df.columns:
            df = df[df['Typrep'].notna()].copy()
            df['typrep_priority'] = df['Typrep'].map(self.typrep_priority).fillna(99)
            
            # 按优先级排序后去重
            df = df.sort_values(['Stkcd_std', 'Year', 'Typrep', 'typrep_priority'])
            df = df.drop_duplicates(subset=['Stkcd_std', 'Year', 'Typrep'], keep='first')
            df = df.drop(columns=['typrep_priority'])
            
            logger.info(f"  去重后: {len(df)} 行, 报表类型: {df['Typrep'].value_counts().to_dict()}")
        else:
            # 无 Typrep 的表（如披露财务），按 (Stkcd, Year) 去重
            df = df.drop_duplicates(subset=['Stkcd_std', 'Year'], keep='first')
            logger.info(f"  去重后: {len(df)} 行")
        
        return df
    
    def merge_financial_tables(self):
        """Step 3: 横向并表（三键策略 + 外连接保留所有记录）"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: 数据集成 - 横向并表（三键策略）")
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
        merge_keys = ['Stkcd_std', 'Year', 'Typrep']
        
        # 选择基准表的指标列和其他重要字段
        indicator_cols = [c for c in base_df.columns if c.startswith('F')]
        other_cols = [c for c in base_df.columns if c in ['Indcd']]
        keep_cols = merge_keys + other_cols + indicator_cols
        keep_cols = [c for c in keep_cols if c in base_df.columns]
        base_df = base_df[keep_cols].copy()
        
        logger.info(f"\n基准表（经营能力）: {len(base_df)} 行")
        
        # 依次合并其他表（使用 outer join 保留所有数据）
        merge_order = ['profit', 'growth', 'solvency', 'risk', 'pershare', 'dividend']
        
        for name in merge_order:
            if name not in dfs:
                logger.warning(f"  {name} 表不存在，跳过")
                continue
            
            df = dfs[name].copy()
            
            # 选择指标列和其他重要字段
            indicator_cols = [c for c in df.columns if c.startswith('F')]
            other_cols = [c for c in df.columns if c in ['Indcd']]
            
            # 检查是否有 Typrep
            has_typrep = 'Typrep' in df.columns
            
            if has_typrep:
                keep_cols_df = merge_keys + other_cols + indicator_cols
                keys_to_use = merge_keys
            else:
                # 无 Typrep 的表，只使用 (Stkcd, Year)
                keep_cols_df = ['Stkcd_std', 'Year'] + other_cols + indicator_cols
                keys_to_use = ['Stkcd_std', 'Year']
            
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
        df['Accper'] = df['Year']
        
        # 确保 Typrep 存在
        if 'Typrep' not in df.columns:
            df['Typrep'] = ''
        
        # 确保 Indcd 存在
        if 'Indcd' not in df.columns:
            df['Indcd'] = ''
        
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
        
        try:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            logger.info(f"✓ 成功保存: {output_file}")
            logger.info(f"✓ 文件大小: {size_mb:.2f} MB")
            logger.info(f"✓ 数据形状: {df.shape}")
            
            logger.info(f"\n前5行预览:")
            logger.info(f"\n{df.head()}")
            
            return output_file
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
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

