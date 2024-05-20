# 导入必要的库
import streamlit as st           # 导入Streamlit库，用于创建和运行Web应用
import pandas as pd              # 导入Pandas库，用于数据处理和分析
from io import StringIO          # 从io库导入StringIO，用于处理内存中的字符串流
import csv                       # 导入csv库，用于读写csv文件
import numpy as np               # 导入NumPy库，用于数值计算
from funs import predicting, get_model_list  # 从funs模块导入predicting和get_model_list函数

# 数据定义
datas = []                       # 初始化数据列表，用于存储电池数据
count = 0                        # 初始化计数器，用于统计数据点的数量
ys, preds, mses, rmses, maes, mapes = [], [], [], [], [], []  # 初始化存储各种评估指标和预测结果的列表
models = get_model_list()        # 调用get_model_list获取模型列表

title = "锂电池剩余使用寿命预测"      # 设置页面标题
# 设置Streamlit页面的配置
st.set_page_config(
    page_title=title,
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(title)                  # 显示页面标题

# 博客介绍部分
st.markdown("""
## 背景介绍
锂电池因其高能量密度和长寿命而被广泛应用于多种设备中，如手机、笔记本电脑和电动车等。然而，随着使用时间的增长，锂电池的性能会逐渐衰退。准确预测电池的剩余使用寿命（RUL）对于设备维护和安全运行至关重要。

## 常用预测方法
- **统计方法**：使用历史数据来建模电池衰退过程。
- **机器学习方法**：通过训练模型来预测电池寿命，例如本应用所使用的深度学习模型。

## 相关研究
- [论文1](http://example.com)
- [论文2](http://example.com)

""")

# 创建一个选择框，让用户选择模型权重文件
option = st.selectbox(
        '请选择模型权重文件',
        models,
        index= models.index(st.session_state.get('model_name')) if st.session_state.get('model_name') in models else 0
    )

# 创建一个文件上传器，用户可以上传CSV格式的电池数据
uploaded_file = st.file_uploader("请上传csv格式电池数据", type=['csv'])

# 处理上传的文件
if uploaded_file is not None:
    datas = []  # 清空数据列表
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))  # 读取文件内容到内存中的字符串流
    csv_reader = csv.reader(stringio, delimiter=',')  # 创建CSV阅读器
    for row in csv_reader:
        data = [float(i) for i in row]  # 将每一行数据转换为浮点数列表
        datas.append(data)  # 将转换后的数据添加到列表中
    ys, preds, mses, rmses, maes, mapes = predicting(datas, option)  # 调用predicting函数进行预测并获取结果
    count = len(datas)  # 更新数据点计数器

# 展示预测结果和评估指标
for i, data in enumerate(datas):
    with st.container():  # 创建一个新的容器
        st.write(f'## 数据{i + 1}')  # 显示数据序号
        ys_i_reshaped = np.array(ys[i]).reshape(1, -1)  # 将实际数据转换为二维数组
        preds_i_reshaped = np.array(preds[i]).reshape(1, -1)  # 将预测数据转换为二维数组

        chart, params = st.columns([0.7, 0.3])  # 创建两列，一列用于显示图表，一列用于显示指标

        with chart:
            # 创建并显示图表
            chart_data = pd.DataFrame(np.concatenate([ys_i_reshaped, preds_i_reshaped], axis=0).T, columns=["真实电池数据", "预测电池数据"])
            st.line_chart(chart_data)

        with params:
            # 显示评估指标
            dt = {
                'mse': mses[i],
                'rmse': rmses[i],
                'mae': maes[i],
                # 'mape': mapes[i],
            }
            dict_list = [{'指标': key, '数值': value} for key, value in dt.items()]
            st.table(dict_list)
        
        st.markdown("---")  # 在部分之间添加分割线

