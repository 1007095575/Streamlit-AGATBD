import streamlit as st
from io import StringIO
import csv
from funs import training, get_model_data
from args import AGATBD_args_parser
# 数据定义
datas = []
count = 0
st.session_state.setdefault('model_name', '')
st.session_state.setdefault('training', False)

title = "电池预测 | 训练阶段"
st.set_page_config(
    page_title=title,
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",)

st.title(title)


# 设置参数，如果需要其他输入组件，可以参考https://docs.streamlit.io/library/api-reference/widgets，设置一行三列
param1, param2, param3 = st.columns(3)

with param1:
   batch_size = st.number_input('batch_size', value=64)
with param2:
   optimizer = st.selectbox(
        '优化器',
        ('adam',),
        index=0
    )
with param3:
   step_size = st.slider('step_size', 0, 200, 50)

uploaded_file = st.file_uploader("请上传cvs格式电池数据", type=['csv'])

if uploaded_file is not None:
    datas = []
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    csv_reader = csv.reader(stringio, delimiter=',')
    for row in csv_reader:
        # 打印每一行数据，也可以进行其他操作
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
            'step_size': step_size
        }
        st.warning('训练中，请耐心等待')
        st.session_state['model_name'] = training(args, datas, params)

if uploaded_file and st.session_state.get('model_name'):
    st.warning('可以下载了')
    mode_data = get_model_data(st.session_state['model_name'])
    if mode_data:
        st.download_button(
            label="点击下载训练权重",
            data=mode_data,
            file_name='model.pkl',
        )
    