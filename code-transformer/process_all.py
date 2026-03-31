import os
import pandas as pd
from utils import read_data
from parse import parse_args
import numpy as np

# 1. 初始化设置
args = parse_args()
output_dir = 'dataset/UrbanEV'
os.makedirs(output_dir, exist_ok=True)

# 强制设置参数以生成“全特征”文件
args.feat = 'occ'       # 主特征：占用率
args.add_feat = 'all'   # 外部特征：全部引入
args.pred_type = 'region'

print(f"开始预处理数据，目标：生成 {args.feat}-{args.add_feat}.csv ...")

# 2. 读取数据 (read_data 会根据 args.add_feat='all' 自动加载电价和天气)
feat, adj, extra_feat, time = read_data(args)

# 3. 转换为主数据框 (275个区域的占用率)
df_feat = pd.DataFrame(feat)
# 规范化列名：第一列为 OT，后续为 1-274
feat_columns = ['OT'] + [str(i) for i in range(1, df_feat.shape[1])]
df_feat.columns = feat_columns

# 4. 缝合外部特征 (关键步骤！)
# 4. 缝合外部特征 (解决 3D 维度报错)
if extra_feat is not None:
    # 核心修复：将 3D 数据 (4344, 275, 8) 展平为 2D (4344, 2200)
    # 意思是：把 275 个区域的 8 个特征，全部平铺成列
    extra_feat_2d = extra_feat.reshape(extra_feat.shape[0], -1)

    # 将展平后的 2D 数据转为 DataFrame
    df_extra = pd.DataFrame(extra_feat_2d)

    # 为这 2200 个外部特征列命名，防止列名冲突
    df_extra.columns = [f'ext_feat_{i}' for i in range(df_extra.shape[1])]

    # 横向合并：275列占用率 + 2200列外部特征
    data = pd.concat([df_feat, df_extra], axis=1)
else:
    data = df_feat

# 5. 插入时间轴
data.insert(0, 'date', time)

# 6. 保存文件
output_path = os.path.join(output_dir, f'{args.feat}-{args.add_feat}.csv')
data.to_csv(output_path, index=False, header=True)

print(f"✅ 处理完成！文件已保存至: {output_path}")
print(f"数据维度: {data.shape} (包含时间列、275个区域列和外部特征列)")