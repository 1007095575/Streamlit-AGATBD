import streamlit as st
from io import StringIO
import csv
from funs import training, get_model_data
from args import AGATBD_args_parser

# 数据定义
datas = []
count = 0

# 初始化 session state
st.session_state.setdefault('model_name', '')
st.session_state.setdefault('training', False)

# 页面设置
title = "模型训练阶段"
st.set_page_config(
    page_title=title,
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(title)

st.write("""
         在本页面您可以按照上面超参数的设置自由地训练您的模型，模型训练完毕后可以点击下载到您自己的本地电脑，模型命名为“当前时间戳.pkl”。如当您训练模型时为2024年5月23日上午9时30分，则模型命名为2024_05_22_09_30_00.pkl，模型训练好后在predict页面能够选中到您的模型。
         """)

# 设置参数
param1, param2, param3 = st.columns(3)
with param1:
    batch_size = st.number_input('batch_size', value=64)
with param2:
    optimizer = st.selectbox('优化器', ('adam',), index=0)
with param3:
    step_size = st.slider('step_size', 0, 200, 50)

param4, param5, param6 = st.columns(3)
with param4:
    epoch = st.number_input('epoch (自然数)', min_value=1, value=400, step=1)
with param5:
    input_size = st.number_input('input_size (电池个数)', min_value=1, value=4, step=1)
with param6:
    lr = st.number_input('lr (学习率)', min_value=0.0001, value=0.01, step=0.0001, format="%.4f")

param7, param8, param9 = st.columns(3)
with param7:
    num_layers = st.number_input('num_layers (GRU层数)', min_value=1, value=2, step=1)
with param8:
    seq_len = st.number_input('seq_len (滑动窗口长度)', min_value=1, value=8, step=1)
with param9:
    heads = st.number_input('heads (GAT多头注意个数)', min_value=1, value=6, step=1)


uploaded_file = st.file_uploader("请上传csv格式电池数据", type=['csv'])

if uploaded_file is not None:
    datas = []
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    csv_reader = csv.reader(stringio, delimiter=',')
    for row in csv_reader:
        data = [float(i) for i in row]
        datas.append(data)
    if datas:
        data_len = len(datas[0])
        args = AGATBD_args_parser(data_len=data_len)  # 传递数据长度
    else:
        args = AGATBD_args_parser()  # 使用默认数据长度

if st.button("开始训练"):
    st.session_state['model_name'] = ''
    if not uploaded_file:
        st.write('## 请先上传数据')
    if uploaded_file is not None and st.session_state['model_name'] == '':
        params = {
            'batch_size': batch_size,
            'optimizer': optimizer,
            'step_size': step_size,
            'epoch': epoch,
            'input_size': input_size,
            'lr': lr,
            'num_layers': num_layers,
            'seq_len': seq_len,
            'heads': heads,
        }
        st.warning('训练中，请耐心等待')
        st.session_state['model_name'] = training(args, datas, params)
        st.warning('训练完成，模型名已保存')

if uploaded_file and st.session_state.get('model_name'):
    st.warning('可以下载了')
    
    model_data = get_model_data(st.session_state['model_name'])
    if model_data:
        st.download_button(
            label="点击下载训练权重",
            data=model_data,
            file_name=f'{st.session_state["model_name"]}.pkl',
        )

