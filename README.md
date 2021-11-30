# Transformer用于文本情感分析
## 介绍
随着计算机技术的飞速发展，数据的规模与形式呈“爆炸式”的增长，数据挖掘技术在这样一个数值化、信息化的时代则显得尤为重要。随之而来的是对各种模态中提取出精确特征技术的更高要求，如在2016年的美国大选中，传统媒体都推测希拉里将获得大选胜利，而人工智能结合Twitter、Facebook等大数据却分析得出特朗普有可能赢得最后选举，事实证明，人工智能的结论是正确的。在现实生活中， 观点的呈现多以文本的形式，文本数据挖掘技术至2000年开始迅猛发展，现已成为自然语言处理和数据挖掘交叉领域的热点方向，而中情感分析（Sentiment analysis）为text mining的基本问题。

本文旨在使用Transformer的encoder结构实现文本情感分析任务。Transformer是Google于2017年在Attention is all you need一文中提出的基于encoder-decoder结构的深度学习框架。如今，Transform不仅被广泛应用在nlp的各类任务中，甚至在cv领域的一些问题上也展现了出色的性能。

## 数据
### 数据集
数据集源自微博中对话数据集，数据量约8万句，每句话有一个情绪标签，情绪标签有六种，
分别是(Happiness, Love, Sorrow, Fear, Disgust, None)，被标识为(1, 2, 3, 4, 5, 6)

数据集部分展示如下：
| id | sentence | label |
| :--: | :--- | :---:|
| 1 | 我就奇怪了，为啥你能拍的这么美呢？| 2 |
| 2 | 是这是人家自己的事就算我能见到她也不会说你们分手吧什么的可是我真心不喜欢冯绍峰这个理由够吗| 5 |
| 3 |偶看报纸上的评论，好像也不看好。 | 3 |
| ... |... | ... |

## 数据预处理
该数据集比较简单，我对其的预处理就是简单的去除id和无意义的符号，将每句话与标签一一对应。
## 数据集划分
为了防止过拟合，我将原始数据集按照7:3的比例划分为训练集和交叉验证集。
## 评价指标
本文采用的评价指标是基于混淆矩阵计算的F1-score。

## 模型
本次任务是一个分类问题，因此我们只需要Transformer中的encoder部分。
### 数据处理
数据处理比较常规就不附代码了。
### transformer的实现
transformer的encoder结构：
+ 嵌入层
+ 前馈层
+ 前馈网络
+ 多头注意力机制

#### 嵌入层
根据源论文，transformer架构的embedding层由一个线性转换器和softmax函数组成。在处理嵌入层时，我们顺便把句向量拉长到$max\_len$

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim, pad):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad)
    def forward(self, X):
        for i in range(len(X)):
            if len(X[i]) < max_len:
                X[i].extend([0] * (max_len - len(X[i])))
        X = torch.LongTensor(X)
        return self.embedding(X)
```
#### 前馈层
前馈层主要由两个线性层和一个ReLU组成。
```python
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.Layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, X):
        return self.Layer(X)
```
#### 多头注意力机制
##### self-attention机制
```python
class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_k)
        self.V = nn.Linear(input_dim, dim_v)
        self.dk = 1 / math.sqrt(dim_k)
    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        out = nn.Softmax(dim=-1)(torch.matmul(q, k.permute(0, 2, 1)))
        out = torch.matmul(out, v)
        return out
```
##### multi-heads attention机制
所谓的多头注意力机制其实就是在自注意力机制的基础上，对张量q的每一个维度都单独进行计算。实现起来也并不困难，可以给进行线性变换后的`q、k、v`矩阵的最后一个维度进行切片，提升`q、k、v`矩阵的维度，然后对其单独计算，计算后恢复维度。
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_k, dim_v, heads):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_k)
        self.V = nn.Linear(input_dim, dim_v)
        self.dk = 1 / math.sqrt(dim_k)
        self.heads = heads
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
    def forward(self, x):
        q = self.Q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_q // self.heads)
        k = self.K(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.heads)
        v = self.V(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.heads)
        out = nn.Softmax(dim=-1)(torch.matmul(q, k.permute(0, 1, 3, 2)))
        out = torch.matmul(out, v)
        out.reshape(x.shape[0], x.shape[1], -1)
        return out
```
#### 位置编码
根据位置编码公式
$$
PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_model}) \\
PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_model})
$$

可以比较容易写出位置编码器的代码：
```Python
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim

    def forward(self):
        pe = np.zeros((max_len, self.input_dim))
        for i in range(max_len):
            for j in range(self.input_dim):
                if j % 2 == 0:
                    pe[i][j] = math.sin(j / pow(10000, 2 * j / self.input_dim))
                else:
                    pe[i][j] = math.cos(j / pow(10000, 2 * j / self.input_dim))
        return torch.from_numpy(pe)
```

### 主模型
由于本次任务是一个6分类任务，output_dim设置为6.
```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = Embedding(vocab_size, input_dim, pad)
        self.positional = PositionalEncoding(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(input_dim, heads_num, hidden_dim, p_drop)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(max_len, 6)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.embedding(x) + self.positional().to(device)
        x = x.to(torch.float32)
        x = self.encoder(x)
        x = torch.mean(x, 1)
        x = self.fc(x)
        return self.softmax(x)
```
## 实验
### 参数调整
在此问题中，我们一般可调整参数的有，$max\_len, input\_dim(embedding\_dim), hidden\_dim, p\_drop, batch\_size$

### 实验结果
在调整参数后，最终实验精度可以达到精确度75%，F1-score 80%左右。
### 心得体会
来到东北大学读研究生后，选择了一系列跟未来研究方向相关的专业课程。其中，《语言分析与机器翻译》一课是令我印象最深刻的。肖老师讲课幽默风趣，深入浅出。在讲解理论的同时注重实践。

感谢肖桐老师和博士师兄对我们的教导，希望将来还能有机会跟随肖老师学习。
