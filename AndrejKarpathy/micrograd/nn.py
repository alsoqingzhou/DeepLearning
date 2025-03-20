import random
from micrograd.engine import Value

## 基准类，单个神经元类，单层神经元类，多层感知

## 基准类
class Module:
    def zero_grad(self):
        ## 所有权重和偏置的value类，其梯度初始化为0
        for p in self.parameters():
            p.grad = 0

    def parameters():
        return []

## 单个神经元
    ##方法零：init,接受哪些参数？初始化哪些属性？
    ##方法一：__call__如何计算加权和输出？
    ##方法二：parameter，汇总单个神经元的所有权重和偏置参数成列表
    ##方法三：__repr__打印单个神经元的哪些信息？

class Neuron(Module):
    #初始化神经元，传入该神经元的输入，是否应用非线性激活函数
    def _init__(self, nin, nonlin=True):
        #初始化权重列表，列表的每一项为一个输入
        #类型为value,值为-1到1之间的随机数
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]

        #初始化偏置，单个神经元只有一个偏置
        #类型为value，值为0
        self.b = Value(0)

        #确认该神经元是否应用非线性激活函数
        self.nonlin = nonlin

    #定义调用该神经元作为函数时，执行什么操作
    #需要传入上一层神经元的输出值向量，用于本神经元计算输出
    def __call__(self, x):
        #调用该神经元，响应为先计算加权和
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        
        #输出激活值
        return act.relu() if self.nonlin else act

    #定义单个神经元的参数列表
    def parameters(self):   
        #返回权重和偏置拼接而成的参数列表
        return self.w + [self.b]
    
    def __repr__(self):
        #打印单个神经元，返回字符串，字符串由两部分构成
        #前面显示神经元是否使用非线性激活函数
        #后面显示神经元的输入数量，该数量从权重获取
        return f"{'ReLu' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

## 单层神经元
    ##方法零：Init，接受哪些参数？初始化哪些属性？
    ##方法一：call 如何计算该层输出？
    ##方法二：如何遍历该层每一个神经元的每一个参数，汇总成一维列表？
    ##方法三：打印该层哪些信息？如何显示？

class Layer(Module):

    #初始化一层神经元，传入参数为前一层神经元数，本层神经元数
    #**kwargs为可变关键字参数字典，可以传递额外的参数给神经元，例如是否应用非线性激活函数
    def __init__(self, nin, nout, **kwargs):
        #一层神经元存储为一个列表，其中的每一项由neuron类生成
        #生成数量，遍历本层神经元数得到
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    #调用该层神经元，需传入上一层的全部输出值
    def __call__(self, x):
        #输出值为一个列表，列表的每一项为该层某一个神经元的输出
        #每一项的具体内容，将上一层的输出值传递给当前遍历到的神经元，该神经元运算得到
        out = [n(x) for n in self.neurons]

        #如果该层只有一个神经元，就直接返回列表的第一项
        #为什么添加这个判断？因为默认返回的是一个列表
        #而只有一项，不必返回列表，直接返回那一项的value类实例
        return out[0] if len(out) ==1 else out
    
    def parameters(self):
        #返回以为列表，存储该层的全部神经元的权重和偏置
        #两层遍历，首先遍历该层的每一个神经元，再遍历每一个神经元的参数列表
        return [p for n in self.neurons for p in n.parameters]
    
    def __repr__(self):
        #打印该层神经元的方法，首先遍历每一个神经元
        #str(n),调用每一个神经元的repr方法，返回字符串
        #使用， 将所有神经元的字符串打印连接起来
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

## 多层感知
    ##方法零：Init，接受哪些参数？怎样初始化各层？
    ##方法一：call 如何前馈运算每一层得到输出层？
    ##方法二：如何遍历每一层每一个神经元，汇总列表？列表是多少维？
    ##方法四：如何打印神经网络？

class MLP(Module):
    #初始化神经网络，传入参数为输入层神经元数量，其他层的神经元数量的列表
    def __init__(self, nin, nouts):
        #定义神经网络的尺寸，类型是列表，每一项表示各层的神经元数量
        #包含从输入层一直到输出层所有层
        sz = [nin] + nouts

        #接下来初始化每一层,i表示输出索引，从输入层到最后一个中间层
        #j表示输出层索引，从第一个中间层到输出层
        #除了输出层，其他层都应用非线性激活函数
        self.layers = [Layer(sz[i], sz[j], nonlin= j!= len(sz)-1) for i, j in zip(sz[:-1],sz[1:])]

    def __call__(self, x):
        #调用mlp，从输出层的数据，前向传播到输出层
        for layer in self.layers:
            #x追踪从中间层开始的每一层的输出，最终x记录为输出层的值
            x = layer(x)
        return x
    
    def parameters(self):
        #返回神经网络的一维参数列表
        #循环遍历每一层，获取每一层的参数列表
        return [p for layer in self.layers for p in layer.parameters]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
        