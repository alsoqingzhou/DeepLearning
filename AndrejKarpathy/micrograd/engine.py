## value类
    ## Init，传入哪些参数？初始化哪些属性？
        # 传入参数
            # data，int型，具体数值
            # children，集合 该数值由哪些运算得到
            # op, 字符串，运算符号
        # 初始化属性
            # self.data = data
            # self.children = child
            # self.op = ''
            # self.grad = 0
    ## 加法，传入哪些参数？如何运算？其中的反向传播如何定义？
        # 传入参数
            # self，other
        # out.data = self.data + 
    ## 乘法，同上
    ## 乘方，同上
    ## 反向传播
        ## 首先深度优先偏离，用列表存储神经网络的拓扑排序
        ## 对拓扑排序的数组中的每一项，针对产生其的运算符号，应用反向传播

class Value:

    # 创建一个新的value类，必须输入该实例的值，它的children和操作符是可选项，不输入默认为空
    # 设置默认值，可以方便创建新的value实例
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        ## 新创建的类初始梯度为空
        self.grad = 0

        #实例的children接受类的元组，将其创建为一个集合。
        #使用集合，可以确保其中的children不重复
        self._prev = set(_children) 
        self._op = _op

        #backward方法初始化为匿名函数，每个类实例定义各自的方法
        self._backward = lambda: None

    def __add__(self, other):
        # 确保other也是value类，以免出现a+1，而1不是value类，报错
        other = other if isinstance(other, Value) else Value(other)

        # 运算出add方法的返回value类实例
        out = Value(self.data+other.data, (self,other), '+')

        #定义参与add方法的类实例的梯度
        def _backward():
            #使用+= 因为每个类实例不止输出到一个类，需要累加梯度
            self.grad += out.grad
            other.grad += out.grad
        #将定义好的backward赋予add方法的结果
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data*other.data, (self,other), '*')

        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad

        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        #幂运算的指数只支撑整型或浮点型
        assert isinstance(other, (int, float))
        #结果的children只有self, 运算符是指数为other的幂运算
        out = Value(self.data**other.data,(self,), f'**{other}')

        def _backward():
            self.grad += out.grad*(other * self.data**(other-1))

        out._backward = _backward

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'Relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

    def backward(self):
        #初始化拓扑排序列表和访问集合
        #访问集合采用集合，避免重复
        visited = set()
        topo = []

        #深度优先搜索，完成拓扑排序
        def build_topo(v):
            #首先检查节点是否访问过，如果已访问过，则出栈，访问上一层的其他前驱
            if v not in visited:
                #没有访问过，则访问，并加入访问集合
                visited.add(v)
                #遍历v的每一个前驱节点
                for child in v._prev:
                    #针对v的前驱，深度优先搜索，建立拓扑排序
                    build_topo(child)
                #将v的所有前驱节点添加到拓扑列表中，再尾插入v
                topo.append(v)
        #从self开始构建拓扑排序
        build_topo(self)

        #开始反向传播
        self.grad = 1 #从self开始，self关于自身的导数为1

        #首先逆转拓扑序列，从尾部开始反向传播计算梯度
        #首先更新输出层的直接前驱的梯度，然后逐次向前
        for v in reversed(topo): 
            v._backward()
