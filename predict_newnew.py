# 导入必要的库
import streamlit as st           # 导入Streamlit库，用于创建和运行Web应用
import pandas as pd              # 导入Pandas库，用于数据处理和分析
from io import StringIO          # 从io库导入StringIO，用于处理内存中的字符串流
import csv                       # 导入csv库，用于读写csv文件
import numpy as np               # 导入NumPy库，用于数值计算
from funs import predicting, get_model_list  # 从funs模块导入predicting和get_model_list函数
import pyperclip
from args import AGATBD_args_parser
# 数据定义
datas = []                       # 初始化数据列表，用于存储电池数据
count = 0                        # 初始化计数器，用于统计数据点的数量
ys, preds, mses, rmses, maes, mapes = [], [], [], [], [], []  # 初始化存储各种评估指标和预测结果的列表
models = get_model_list()        # 调用get_model_list获取模型列表


title = "锂电池组健康状态预测 | 基于自适应图学习的锂电池组健康状态预测算法-AGATBD"      # 设置页面标题

# 设置页面
st.set_page_config(
    page_title=title,
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* 修改标题的字体大小和颜色 */
h1 {
    font-size: 26px;
    color: #0e1117;
    font-family: 'Helvetica';
}

/* 修改目录和小标题的字体大小、样式和颜色 */
h2 {
    font-size: 22px;
    color: #0e1117;
    font-family: 'Verdana';
}

/* 修改正文的字体大小、样式 */
body {
    font-size: 18px;
    font-family: 'Arial';
    color: #4a4a4a;
}
</style>
""", unsafe_allow_html=True)

def load_css():
    with open("style.css", "r", encoding="utf-8") as f:  # 指定文件以UTF-8编码打开
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# 页面标题和介绍
st.title("锂电池组健康状态预测 | 基于自适应图学习的锂电池组健康状态预测算法-AGATBD")

# 目录部分
st.markdown("""
## 目录
1. [研究背景](#background-introduction)
2. [研究现状](#methodology)
3. [创新点](#case-study)
4. [基于自适应图学习的锂电池组健康状态预测算法-AGATBD](#results-analysis)
5. [关键代码](#conclusion)
6. [上传数据观察结果](#test)
""", unsafe_allow_html=True)

# 各章节内容与锚点
st.markdown("<a id='background-introduction'></a>", unsafe_allow_html=True)
st.header("1. 研究背景")
st.write("""
         目前，可持续能源技术的发展已成为全球关注的焦点。在这个大背景下，作为新能源产业的关键技术之一，锂离子电池凭借其环保、高效及
         长寿命等特性被公认为是储能的首选，已经在全球能源市场上占据了重要地位。如图1所示，2014年至2023年间，全球锂离子电池的出货量呈现逐年增长的趋势。
         但是，随着锂电池充放电次数的增加，其内部电化学成分将产生负面变化，导致电池容量产生不可逆的衰减，进而影响电池性能和使用寿命，增加漏电和短路的风险，导致安全隐患。因锂电池老化故障造成人员生命财产安全受到严重威胁的例子屡见不鲜。
         锂离子电池的可靠性和安全性评估已成为电池制造商非常关注的问题，尤其是对电池未来性能的预测能力。
         
         电池管理系统（Battery Management System，BMS）中的电池健康状态（State of Health，SOH）是一个表征电池当前容量与额定容量比值的关键指标，它不仅能够反映电池的容量和寿命状况，还是评估电池老化程度的重要指标，
         通过分析锂电池的SOH能够推断其RUL，对于确保电池性能、延长使用寿命以及提高工业场景能源系统的安全性和可靠性具有至关重要的作用。
         """)
col1, col2, col3 = st.columns([1,2,1])  # st.columns([1,2,1]) 创建了三列，中间列是两侧列的两倍宽。将图片放在中间列可以达到居中的效果。
with col2:  # 使用中间列来显示图片，会使图片居中
    st.image("C:/Users/86180/Desktop/IJCNN/展示网页的材料/图1.jpg", caption="图1 2014-2023年全球锂离子电池出货量示意图")

st.markdown("<a id='methodology'></a>", unsafe_allow_html=True)
st.header("2. 研究现状")
st.write("""
         对于锂电池的SOH预测，通常可以分为三类：传统的基于模型的方法、数据驱动方法以及融合方法，图2展示了该领域整体的研究现状。
         
         - **基于模型的方法**：基于模型的方法早期研究主要是基于实验条件下直接测量出来的能够表征电池衰退状态的循环寿命、阻抗、内阻等特征参数来预测SOH值，主要包括电化学模型和等效电路模型。
         这种方法深入研究锂电池退化和失效的原理，依托电池复杂的物理行为或者化学行为对能够表征锂电池退化机制的关键参数进行数学建模，进而预测未来的健康状态，对于特定电池建立的模型性能很强，但与之对应的是高度依赖专家知识和较低的泛用性。
         
         - **基于数据驱动的方法**：基于数据驱动的方法不需要知道锂电池内部复杂的老化机理，只根据传感器设备监测到的历史参数数据预测电池SOH，随着数据挖掘和大数据分析技术的发展，数据驱动的方法已经成为现阶段的研究热点。以方法领域细分，
         包括有统计学中的回归分析方法如ARIMA、自适应维纳过程回归、高斯过程回归等等，机器学习方法如支持向量机SVR、决策树DT、随机森林RF算法等等，深度学习方法如卷积神经网络CNN、循环神经网络RNN及其变体LSTM、GRU等等，近年来出现的一些
         深度学习框架如transformer和图神经网络GNN等也被广泛用于锂电池的SOH预测。
         
         - **融合方法**：融合方法的目的就是通过精细地设计，将上述算法针对特定任务、特定场景和特定数据进行有机结合，共同发挥优势和克服局限性，使算法更适合实际的运行条件。现阶段融合方法的种类非常之多，
         不仅包括基于模型的方法和基于数据驱动的方法之间的结合，也有数据驱动算法的相互结合，融合的算法个数也可以是多个。
         
         尽管上述SOH预测方法已展现出卓越的性能，其应用于实际工程项目时仍面临一系列挑战。
         
         （1）当前SOH预测方法大多仅关注单一电池的SOH预测，忽略了电池组作为一个整体的健康状况。
         现有能源设备对能源需求持续增长，单个电池往往无法满足需求，而是依赖多电池组合的电池组提供必要的能源。因此，仅通过对单个电池SOH的预测来推断整个电池组的健康状态无法全面反映电池组的整体健康状况，进而可能影响系统的总体性能与安全性。
         
         （2）电池组内部电池之间存在的复杂相互作用及其动态变化，为SOH的准确预测带来了额外挑战。目前的预测方法依赖于单一电池的预定义特征分析或人工提取特征，这种做法无法准确地理解这些复杂内部动态对于特定任务场景的底层机制，
         导致无法有效管理电池组的健康状态，进而影响整个系统的性能。如何以最有利的方式提取这些电池之间的空间互连方式信息以准确预测SOH，这仍然是该领域一个亟待解决的问题。
         
         """)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:  # 使用中间列来显示图片，会使图片居中
    st.image("C:/Users/86180/Desktop/IJCNN/展示网页的材料/锂电池SOH预测现状.jpg", caption="图2 锂电池SOH预测研究现状")

st.markdown("<a id='case-study'></a>", unsafe_allow_html=True)
st.header("3. 创新点")
st.write("""
         - **锂电池组建模**：将锂电池组电池容量数据以图形的方式建模，利用GNN强大的关系推理能力捕捉电池依赖关系；
         
         - **最优特征提取**：通过所设计的最优图提取器自适应提取电池之间的最优空间互连方式，在此基础上利用GAT和GRU分别提取空间域和时间域的特征，从而有效提高算法预测精度和可解释性。 
         
         - **特征融合**：为了防止连续时空特征的提取而导致关键特征被稀释和淡化的问题，本文引入了一个时空特征融合策略，以最大化利用所提取的有效信息，进一步增强算法的性能。
         
         - **通用框架**：AGATBD是一个通用的训练框架，无需事先了解锂电池组的具体先验知识。该框架具有即插即用的特点，可适用于不同连接方式如串联或并联的电池组，从而提高算法在现实场景下对各种能源设备的电池组的鲁棒性。
         """)

st.markdown("<a id='results-analysis'></a>", unsafe_allow_html=True)
st.header("4. 基于自适应图学习的锂电池组健康状态预测算法-AGATBD")
st.write("""
         本文所提算法AGATBD(Adaptive Graph Attention Networks for Battery Dependencies)旨在分析单个电池在同一电池组中的复杂相互作用和影响，以理解电池组整体健康状况并预测锂电池组的剩余使用寿命。整体框架如图3所示，
         主要包括最优图提取器、空间特征提取器、时间特征提取器、时空特征融合器和预测器五个部分。
         """)

# 插入图片
col1, col2, col3 = st.columns([1, 8, 1])
with col2:  # 使用中间列来显示图片，会使图片居中
    st.image("C:/Users/86180/Desktop/IJCNN/展示网页的材料/整体框架.jpg", caption="图3 算法整体框架")


# 在 Markdown 中为特定标题添加类名
st.markdown('<h2 class="custom-subheader">4.1 数据预处理与图建模</h2>', unsafe_allow_html=True)
# 定义该类的 CSS
st.markdown("""
<style>
.custom-subheader {
    font-size: 20px;  /* 特定标题的字体大小 */
    color: #0e1117;  /* 特定标题的字体颜色 */
    font-family: 'Verdana';  /* 特定标题的字体 */
}
</style>
""", unsafe_allow_html=True)

st.write("""
         为了给后续的锂电池组SOH预测任务提供可靠的数据基础，需要先对输入数据进行预处理。为了保证模型对不同电池数据的统一处理，算法采用基于最大最小值的数据归一化处理方法MinMaxScaler()，将原始数据线性地缩放到一个指定的区间范围[0,1]内。
         将归一化后的电池组容量数据以滑动窗口的方式建模成图表示，该过程如图4所示。具体而言，将每个电池都视为图中的一个节点，以节点间的边表示电池存在相互作用和反馈关系。
         """)
# 插入图片
col1, col2, col3 = st.columns([1, 2, 1])
with col2:  # 使用中间列来显示图片，会使图片居中
    st.image("C:/Users/86180/Desktop/IJCNN/展示网页的材料/图建模.jpg", caption="图4 锂电池组建模图表示")


st.markdown('<h2 class="custom-subheader">4.2 最优图提取器</h2>', unsafe_allow_html=True)
st.markdown("""
最优图提取器是该算法的核心。具体来说，将邻接矩阵$A \in R^{n×n}$的每一个元素都视为可学习的参数，并且用伯努利分布对这些元素进行参数化，得到对应的参数化矩阵$P_{ij} \in [0,1]$，其中${A}_{ij}\sim Ber(p_{ij})$，其中$p_{ij} \in P$表示邻接矩阵的元素$A_{ij}$取值为1的概率。
同时，算法使用$\\theta$表示为下游模型的参数集，${P}(\\theta) \in {[0,1]}^{n×n}$表示由元素$p_{ij}(\\theta)$组成的矩阵，该矩阵的值受下游模型的参数$\\theta$的影响。在整个算法训练过程中，下游模型的参数θ随着反向传播算法而变化，从而调整参数化矩阵$P$中每个概率${p}_{ij}$的大小，本质上是调整了邻接矩阵的元素${A}_{ij}$取值为1的概率，
进而通过伯努利采样塑造图结构，这种动态生成的图结构与下游任务的特定需求将保持一致。利用这一机制，最优图提取器旨在识别最有效的电池间连接来准确预测电池组的SOH。最终，最优图提取器被设计为通过最大化整个概率图的平均任务性能即最小化训练的平均损失值来找出最优图结构，通过优化如下公式来实现：
""", unsafe_allow_html=True)

st.latex(r"\min \limits_{\theta }E_{{A}\sim Ber({P}(\theta ))}[L({A},\theta ,{X}_{train})]")
st.markdown("""
其中，${X}_{train}$代表用于训练的电池容量数据，$E[]$表示伯努利分布的统计期望，$L$是整个算法在训练阶段所使用的损失函数即均方误差损失函数MSE。由于在采样过程中需要计算统计期望，可能会导致概率${P}_{ij}$的梯度无法计算，即导致${P}_{ij}$不可微。因为算法需要对该期望值进行离散采样，
由于离散变量的取值不是连续可微的，因而无法对梯度进行计算。为了解决这个问题，算法引入Jang等人引入的Gumbel-Softmax重新参数化方法确保${P}_{ij}$的梯度可以计算。Gumbel-Softmax重新参数化方法请见如下链接：
- [Gumbel-Softmax重新参数化](https://blog.csdn.net/weixin_42468475/article/details/123014858)

最优图提取器总体框架如图6所示，它包括两个主要组成部分，特征提取器和边预测器。特征提取器对输入的全部训练数据进行非线性特征提取，映射到对应的一维特征向量${c}_{i},i=(1, 2, 3,..., n)$，紧接着边预测器接收一对特征向量${c}_{i}, {c}_{j}$，输出两个节点的链接概率$p_{ij}(\\theta)\in\{0,1\}$。
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:  # 使用中间列来显示图片，会使图片居中
    st.image("C:/Users/86180/Desktop/IJCNN/展示网页的材料/最优图提取器.jpg", caption="图5 最优图提取器整体框架，Conv代表一维卷积，BN是批归一化操作，FC是全连接层")

st.markdown('<h2 class="custom-subheader">4.3 空、时间特征提取器</h2>', unsafe_allow_html=True)
st.markdown("""
空间特征提取器利用图注意机制GAT基于所学习的最优图结构来提取最适合下游SOH预测任务的电池之间的空间互连特征。这里，“空间互连”并不是狭义上的电池空间位置信息，而是广义上包含了电池组工作时不同电池之间的相互影响、相互反馈和联系的特定信息在内的特征。
AGATBD在考虑电池组内单体电池相互作用的基础上，实现准确预测电池组SOH的目标，显著提高预测的准确性和可解释性。而时间特征提取器采用门控循环单元GRU作为提取时间特征的主要结构，构建输入特征和输出特征之间复杂的时间依赖关系。GAT和GRU的详细介绍见如下链接：
- [GAT](https://blog.csdn.net/xiao_muyu/article/details/121762806)
- [GRU](https://blog.csdn.net/Michale_L/article/details/122778270)
""", unsafe_allow_html=True)

st.markdown('<h2 class="custom-subheader">4.4 时空特征融合器</h2>', unsafe_allow_html=True)
st.markdown("""
在针对锂电池组电池容量数据的处理过程中，算法首先从空间维度对数据进行特征提取，以获取电池节点间的相互作用和关系模式。然而，若在此基础上直接进入时间维度的特征提取，不加以区分地将更新后的节点特征矩阵应用于时间特征的挖掘会导致原始空间特征会随着神经网络参数的迭代更新被逐渐被稀释和淡化，从而造成导致关键特征信息的丢失。
这种信息的缺失会直接影响到算法的性能，造成不准确的电池容量退化模式预测，降低整个算法的预测准确度。为了解决这个问题，算法设计了一个时空特征融合器，旨在对提取的时空特征进行有效整合，以提升算法的预测性能，所设计时空特征融合器的示意图如图6所示：
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:  # 使用中间列来显示图片，会使图片居中
    st.image("C:/Users/86180/Desktop/IJCNN/展示网页的材料/时空特征融合器.jpg", caption="图6 时空特征融合过程")


st.markdown("<a id='conclusion'></a>", unsafe_allow_html=True)
st.header("5. 关键代码")
st.markdown('<h2 class="custom-subheader">5.1 数据集构建</h2>', unsafe_allow_html=True)
st.write("算法采用滑动窗口的方式对归一化后的电池组容量数据进行处理，滑动窗口大小即为seq_len，训练集和测试集的比例是8:2，代码返回打包后的训练数据、测试数据和全部数据。")

# 定义要显示和复制的代码
code_data = """
def process(args, dataset, batch_size, step_size, shuffle):
    seq = []
    graphs = []
    for i in tqdm(range(0, len(dataset) - args.seq_len - 1, step_size)):
        train_seq = []
        for j in range(i, i + args.seq_len):
            x = []
            for c in range(len(dataset[0])):  # 前seq_len个时刻的所有变量
                x.append(dataset[j][c])
            train_seq.append(x)
        # 下1个时刻的所有变量
        train_labels = []
        for j in range(len(dataset[0])):
            train_label = []
            for k in range(i + args.seq_len, i + args.seq_len + 1):
                train_label.append(dataset[k][j])
            train_labels.append(train_label)
        # tensor
        train_seq = torch.FloatTensor(train_seq)
        train_labels = torch.FloatTensor(train_labels)

        temp = Data(x=train_seq.T, y=train_labels, edge_index=edge_index)
        graphs.append(temp)
    train = graphs[:int(len(graphs) * 0.8)]
    test = graphs[int(len(graphs) * 0.8):len(graphs)]
    train_data = torch_geometric.loader.DataLoader(train, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    test_data = torch_geometric.loader.DataLoader(test, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    total_data = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    return train_data, test_data, graphs, total_data
"""

# 显示代码
st.code(code_data, language='python')

# 添加一个按钮用于复制代码
if st.button('复制代码到剪贴板', key='copy_code_1'):
    pyperclip.copy(code_data)  # 复制代码到剪贴板
    st.success('代码已成功复制到剪贴板！')  # 显示成功消息

##########################################################################

st.markdown('<h2 class="custom-subheader">5.2 模型构建</h2>', unsafe_allow_html=True)
st.write("算法共包括最优图提取器、GAT、GRU、特征融合和全连接层预测五部分，具体定义代码如下：")

# 定义要显示和复制的代码
code_data = """
class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=args.heads, concat=False, dropout=args.dropout)  #in_feats是输入特征数，h_feats是输出特征数，heads是多头注意机制，negative_slope是leakyRELU的参数，默认0.2       
        self.conv2 = GATConv(h_feats, out_feats, heads=args.heads, concat=False, dropout=args.dropout)   #    concat如果是 False，多头注意机制就是平均而不是拼接
    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class AGATBD(nn.Module):
    def __init__(self, args):
        super(AGATBD, self).__init__()
        self.args = args
        self.num_nodes = args.input_size
        self.out_feats = args.out_feats
        self.embedding_dim = 100
        self.hidden_size = args.hidden_size
        self.dim_fc = (args.data_len - 18) * 16
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        
        self.gat = GAT(in_feats=args.seq_len, h_feats=args.h_feats, out_feats=self.out_feats)
        #self.gat = GAT_Encoder(input_dim=args.seq_len, hid_dim=32, gnn_embed_dim=128, dropout=0.5, heads=4)
        self.lstm = nn.GRU(input_size=args.input_size, hidden_size=self.hidden_size,
                            num_layers=args.num_layers, batch_first=True, dropout=args.dropout)
        #self.conv = torch.nn.Conv1d(1, 4, 10, stride=1)  # .to(device)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):    # 原模型的全连接层
            self.fcs.append(nn.Sequential(
                nn.Linear(self.out_feats + self.hidden_size, (self.out_feats + self.hidden_size) // 2),
                nn.ReLU(),
                nn.Linear((self.out_feats + self.hidden_size) // 2, args.output_size)
            ))
    
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([4, 4])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(args.device)
        self.rel_send = torch.FloatTensor(rel_send).to(args.device)

    def forward(self, data, node_feas):
        # 最优图提取器
        # node_feas是原始训练数据，如果共n个电池，每个电池容量长度为l，则node_feas维度为(n, int(0.8*l))
        node_feas = node_feas.to(self.args.device)             
        x = node_feas.view(self.num_nodes, 1, -1)   
        x = self.conv1(x)                       
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)                       
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)         
        x = self.fc(x)                         
        x = F.relu(x)
        x = self.bn3(x)
        
        receivers = torch.matmul(self.rel_rec, x)  
        senders = torch.matmul(self.rel_send, x)      
        x = torch.cat([senders, receivers], dim=1)    
        x = torch.relu(self.fc_out(x))                
        x = self.fc_cat(x)                            

        adj = gumbel_softmax(x, temperature=0.5, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(self.args.device)
        adj.masked_fill_(mask, 0)                 
        adj = sp.coo_matrix(adj.cpu().detach())  
        indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
        edge_index = torch.LongTensor(indices).to(self.args.device)  # PyG框架需要的coo形式
        
        # 空间、时间特征提取器和时空特征融合器
        # x是处理好的训练数据
        x, batch = data.x, data.batch      # x (batch_size*num_nodes , seq_len)

        batch_size = torch.max(batch).item() + 1
        x = self.gat(x, edge_index)   
       # print(x.shape)   # # x (batch_size*num_nodes , out_feas)
        batch_list = batch.cpu().numpy()
        # split
        xs = [[] for k in range(batch_size)]
        ys = [[] for k in range(batch_size)]
        for k in range(x.shape[0]):
            xs[batch_list[k]].append(x[k, :])
            ys[batch_list[k]].append(data.y[k, :])

        xs = [torch.stack(x, dim=0) for x in xs]
        ys = [torch.stack(x, dim=0) for x in ys]
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        # print(x.shape, y.shape)  #    (batchsize, 4, out_feas)  / (batchsize, 4, 1)
        gat_output = x
        x = x.permute(0, 2, 1)   #     (batchsize, out_feas, 4)
        x, _ = self.lstm(x)
        # print(x.shape)      #    (batchsize, out_feas, hiddensize)
        x = x[:, -1, :]
        # print(x.shape)      # 4, 128                    (batchsize, hiddensize)
        
        # 特征融合模块
        fuse = [[] for i in range(self.num_nodes)]
        gat_output = gat_output.permute(1, 0, 2)       # (4， batchsize, out_feas)

        for i in range(args.input_size):
            fuse[i] = torch.cat((gat_output[i], x), dim=1)
            fuse.append(torch.cat((gat_output[i][0], x[0]), dim=0))
        # print(fuse[0].shape)     # 一个列表，四个元素，每个元素是(batchsize, (out_feas + hidden_size)), 第N个元素是第N个变量和全部经过GRU的输出相拼接而成
        preds = []
        for i, fc in enumerate(self.fcs):    
            preds.append(fc(fuse[i]))
        pred = torch.stack(preds, dim=0)    
        pred = pred.permute(1, 0, 2)
        return pred, y
"""

# 显示代码
st.code(code_data, language='python')

# 添加一个按钮用于复制代码
if st.button('复制代码到剪贴板', key='copy_code_2'):
    pyperclip.copy(code_data)  # 复制代码到剪贴板
    st.success('代码已成功复制到剪贴板！')  # 显示成功消息
##########################################################################

st.markdown('<h2 class="custom-subheader">5.3 模型训练</h2>', unsafe_allow_html=True)
st.write("模型训练代码如下，训练各参数可更改，具体在train页面可上传您自己的数据集训练一个模型并测试。")

# 定义要显示和复制的代码
code_data = """
def train(args, train_data, node_feas):
    
    model = AGATBD(args).to(args.device)
    model.train()
    loss_function_1 = nn.MSELoss().to(args.device)
    loss_function_2 = nn.L1Loss().to(args.device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    loss = []
    for epoch in tqdm(range(args.epochs), desc='Training'):
        train_loss = []
        epoch_loss = 0
        for graph in train_data:
            graph = graph.to(args.device)
            preds, labels = model(graph, node_feas)       
            total_loss = loss_function_1(preds, labels) + loss_function_2(preds, labels)
            epoch_loss = epoch_loss + total_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())
        loss.append(epoch_loss / len(train_data))
        scheduler.step()
        tqdm.write("epoch {:03d} train_loss {:.8f}".format(epoch, np.mean(train_loss)))

        
    state = {'model': model.state_dict()}
    name = f'{int(time.time())}' + '.pkl'
    path = 'models/' + name
    torch.save(state, path)
    return name
"""

# 显示代码
st.code(code_data, language='python')

# 添加一个按钮用于复制代码
if st.button('复制代码到剪贴板', key='copy_code_3'):
    pyperclip.copy(code_data)  # 复制代码到剪贴板
    st.success('代码已成功复制到剪贴板！')  # 显示成功消息

####################################################################

st.markdown('<h2 class="custom-subheader">5.4 模型测试</h2>', unsafe_allow_html=True)
st.write("模型测试代码如下，具体在下方可上传您自己的数据集进行测试，注意下方只提供我们使用NASA和CALCE数据集训练好的模型，如果要用您自己的数据集进行训练，建议在train页面重新训练一个模型。")

# 定义要显示和复制的代码
code_data = """
@torch.no_grad()
def test(args, test_data, node_feas, pt_path):
    #graph_struct = []
    print('loading models...')
    model = AGATBD(args).to(args.device)
    print(pt_path)
    if os.path.exists(pt_path):
        print(11)
        model.load_state_dict(torch.load(pt_path)['model'])
    else:
        model.load_state_dict(torch.load('model_version_100.pkl')['model'])

    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for graph in tqdm(test_data):
        graph = graph.to(args.device)
        _pred, targets= model(graph, node_feas)     #
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)  

        for i in range(args.input_size):
            target = targets[:, i, :]    
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        for i in range(args.input_size):
            pred = _pred[:, i, :]
            pred = list(chain.from_iterable(pred.data.tolist()))
            preds[i].extend(pred)
    ys, preds = np.array(ys), np.array(preds)
    mses, rmses, maes, mapes = [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个变量:')
        print(len(y))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        mapes.append(get_mape(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
    return ys, preds, mses, rmses, maes, mapes
"""

# 显示代码
st.code(code_data, language='python')

# 添加一个按钮用于复制代码
if st.button('复制代码到剪贴板', key='copy_code_4'):
    pyperclip.copy(code_data)  # 复制代码到剪贴板
    st.success('代码已成功复制到剪贴板！')  # 显示成功消息
#############################################################################

# 各章节内容与锚点
st.markdown("<a id='test'></a>", unsafe_allow_html=True)
st.header("6. 上传数据观察结果")
st.write("""
         您可以在下方选择模型并上传csv文件进行测试，csv文件请按电池的个数按行处理，即如果有4个电池，每个电池的电池容量序列长度为168，则csv文件的前四行应为各个电池的电池容量数据，形成（4,168）矩阵的形式。
         注意下方只提供我们使用NASA和CALCE数据集训练好的模型，如果要用您自己的数据集进行训练，建议在train页面重新训练一个模型，之后再回到本页面选中您的模型再上传全部数据，本页面会显示包括训练集和测试集全部数据的预测结果方便您观察。
         """)
# 创建一个选择框，让用户选择模型权重文件
option = st.selectbox(
        '请选择模型权重文件',
        models,
        index= models.index(st.session_state.get('model_name')) if st.session_state.get('model_name') in models else 0
    )

# 创建一个文件上传器，用户可以上传CSV格式的电池数据
uploaded_file = st.file_uploader("请上传cvs格式电池数据", type=['csv'])

# 处理上传的文件
if uploaded_file is not None:
    datas = []  # 清空数据列表
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))  # 读取文件内容到内存中的字符串流
    csv_reader = csv.reader(stringio, delimiter=',')  # 创建CSV阅读器
    for row in csv_reader:
        data = [float(i) for i in row]  # 将每一行数据转换为浮点数列表
        datas.append(data)  # 将转换后的数据添加到列表中
    # print("data是")
    # print(datas[0])  # 长度为168的列表
    if datas:
        data_len = len(datas[0])
        args = AGATBD_args_parser(data_len=data_len)  # 传递数据长度
    else:
        args = AGATBD_args_parser()  # 使用默认数据长度
    ys, preds, mses, rmses, maes, mapes = predicting(args, datas, option)  # 调用predicting函数进行预测并获取结果
    count = len(datas)  # 更新数据点计数器

# 展示预测结果和评估指标
for i, data in enumerate(datas):
    with st.container():  # 创建一个新的容器
        st.write(f'## 第{i + 1}个电池')  # 显示数据序号
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

# # 插入表格
# df = pd.DataFrame({
#     '第一列': [1, 2, 3, 4],
#     '第二列': [10, 20, 30, 40]
# })
# st.table(df)

# # 展示Markdown文本，展示自定义HTML内容
# st.markdown("""
# <style>
# .big-font {
#     font-size:30px !important;
#     font-weight: bold;
# }
# </style>
# <p class='big-font'>这是大号字体的文本</p>
# """, unsafe_allow_html=True)

# 整个页面的背景设置
st.markdown("""
<style>
body {
    background-color: #E6E6FA;
    color: #333333;
}
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

