import random

import numpy as np

def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
    
class Network(object):

    ## 初始化神经网络，确定初始权重和偏置
    def _init_(self,sizes): ## sizes列表，每一项存储神经网络各层的神经元数量
        self.num_layers = len(sizes) ## sizes数组的长度，即为网络的层数
        self.sizes = sizes

        ## 网络偏置，存储为列表，列表长度为网络除输入层之外的层数
        ## 列表中的每一项，为y行，1列的二维数组，y为相应层的神经元数量
        ## 之所以存为y行，1列的二维数组，而非一维的列向量，是方便后续的点积运算
        ## 网络偏置的初始值，为标准正态分布随机取值
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        ## 网络权重，存储为列表，列表长度为除了输出神经元之外的层数
        ## 列表中的每一项，为y行，x列的二维数组。
        ## 将权重形象化为从前一层神经元指向下一层神经元的箭头。
        ## y为被指向的该层的神经元数，x为指出层的神经元数。
        ## 之所以将被指向层的单个神经元，关联的权重放在一行，因为这一层的每一个神经元，都接受上一层的所有神经元的激活值。
        ## 这种多对一，可以类比成多元函数接受多个输入，返回一个输出。
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1],sizes[1:])]
    
    ## 前馈函数，输出整个网络各层的加权
    def feedforward(self, a):

        ## 遍历偏置和权重列表中的每一项，即各层神经元的权重和偏置
        for b, w in zip(self.biases, self.weights):
            
            ## 权重矩阵与上一层神经元的激活值点积，随后加上偏置，得到这一层神经元激活值
            ## 上面的计算理解，是矩阵的每一行，和偏置数组的一列对应相乘相加

            ## 换一种理解方式，将权重矩阵视作线性变换的实体，虽然该矩阵不是方阵，但可以粗略类比。
            ## 此时不再一行一行关注该矩阵，而是一列一列关注，每一列为变换后的基向量在新坐标体系的坐标
            ## 如此来看，该权重矩阵的作用，就是将最初输入的向量，进行升维（中间神经元数量比输入神经元多）或者降维操作
            ## 形象理解，对一句话，升维可能意味着从不同的角度去拷问它，是否是疑问句，情绪积极还是消极等等。
            a = sigmoid(np.dot(w, a) + b)
        return a

    
    def SGD(self, training_data, epochs, eta, mini_batch_size,
            test_data = None):
        ## 应用随机梯度下降法，训练多轮，每一轮中，先打散全部训练数据，随后将训练数据拆散成小块
        ## 针对每一小块， 微调权重和偏置
        ## 测试数据默认无，如果有，利用测试数据进行评估，打印结果

        training_data = list(training_data) ##无论训练数据类型，都先获得其列表形式的拷贝
        n = len(training_data) ##获取训练数据大小，用于切片成小块

        if test_data: ##如果有测试数据，对测试数据进行一样的处理，为之后的评估做准备
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs): ## 每一轮都使用所有训练数据
            random.shuffle(training_data) ## 首先打散数据，确保本轮拆分的组块和之前不同

            ## 将训练数据，按照组块大小，进行均匀切片
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            ## 针对每一个组块，更新组块，在损失函数的空间，朝最小值走一步
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data: ##打印第几轮，测试数据正确比例
                print("Epochs {} : {} / {}".format(j, self.evaluate(test_data),n_test))
            else:
                print("Epochs {} complete".format(j))        
    
    def update_mini_batch(self, mini_batch, eta):
        ## 传入单个组块，应用反向传播算法，完成组块整体的权重和偏置更新

        ## 遍历偏置列表，对列表中每一项二维数组，获取其行列数，即形状
        ## 然后新建同形状的二维数组，作为初始的偏置梯度列表的一项
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch: ##针对单个小块中的每一个样本，求出偏置和权重的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) ## bp算法求出单个样本的权重，偏置梯度

            ## 不断累加每个样本生成的梯度，更新初始化的偏置，权重列表
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        ## 完成一个小块内所有的样本的梯度求解，汇总在两个列表中，求得平均值用于更新网络的权重，偏置列表
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        
    def backprop(self, x, y):
        
        ## 初始化偏置和权重的梯度列表
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ## 前馈运算，根据输入的单样本，计算网络的各层激活值

        activation = x ## activation记录当前层的激活值，当前为输入层

        activations = [x] ## 该列表纳入每一次计算出的新一层的激活值，存储整个网络的激活值,开始时只有输入层
        zs = [] ##初始化空列表，纳入各层的加权和，存储整个网络的加权和

        for b, w in zip(self.biases, self.weights): ##开始前馈运算，使用当前网络的权重和偏置列表计算
            ## 计算顺序按照网络中偏置和权重列表存储的顺序，从计算第二层的激活值开始
            ## b和w 记录当前计算的层，所需的偏置和权重

            z = np.dot(w, activation) + b ## 计算当前层的加权和
            zs.append(z) ## 计算完立即纳入加权和列表

            activation = sigmoid(z) ##计算当前层的激活值根据当前 
            activations.append(activation)

        ## 开始反向传播

        ## 第一步，计算输出层的误差,利用损失函数对输出层偏导，和激活函数导数
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        ## 第二步，更新输出层的偏置，权重
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        ## 第三步,对中间层，反向计算各层误差，由计算出的误差求解相应层的偏置，权重偏导
        for l in range(2, self.num_layers): ##遍历网络倒数第二层直至第二层
            z = zs[-l] ## 取当前层的加权和
            sp = sigmoid_prime(z) ## 计算当前层的激活函数对加权和导数

            ## 计算第 l 层的误差，l 取值范围[2，L-1]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            ## 根据计算出的 l 层误差，求得 l 层的偏置和权重
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    ## 如果由测试数据，进行评估




    ## 损失函数对输出层激活值的偏导计算
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

## 激活函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

## 激活函数对加权和的偏导
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))