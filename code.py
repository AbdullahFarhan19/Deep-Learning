import math
import random
import graphviz as gv

class Value:
    def __init__  (self, data, _children = (), _op = "", label = "", grad = 0.0):
        self._backward = lambda: None
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = grad
    
    def __repr__ (self):
        return f"Value(data = {self.data})"
    
    def __add__ (self, other):
        other = other if isinstance(other, Value) else Value(other)
        outer = Value(self.data + other.data, (self, other), _op = "+")
 
        def _backward():
            self.grad += outer.grad
            other.grad += outer.grad

        outer._backward = _backward
        return outer
    
    def __radd__ (self, other):
        return self + other
    
    def __mul__ (self, other):
        other = other if isinstance(other, Value) else Value(other)
        outer = Value(self.data * other.data, (self, other), _op = "*")

        def _backward():
            self.grad += other.data * outer.grad
            other.grad += self.data * outer.grad

        outer._backward = _backward
        return outer

    def __rmul__ (self, other): 
        return self * other
    
    def __neg__ (self):
        return self * (-1)
    
    def __sub__ (self, other):
        return self + (-other)
    
    def exp(self):
        x = self
        o = Value(math.exp(x.data), (x,), _op = "exp")

        def _backward():
            x.grad += o.data * o.grad
        o._backward = _backward
        
        return o
    
    def __pow__ (self, y):
        assert isinstance (y, (int, float)), "int or float only"
        x = self
        o = Value(math.pow(x.data, y), (x, ), _op = f"**{y}")

        def _backward():
            x.grad += y * x.data**(y - 1) * o.grad

        o._backward = _backward

        return o

    def __truediv__ (self, x):
        return self * x**(-1)
    
    def tanh (self):
        x = self
        t = (math.exp(2*x.data) - 1)/(math.exp(2*x.data) + 1)

        outer = Value(t, (x,), "tanh")

        def _backward():
            x.grad += (1 - t*t) * outer.grad

        outer._backward = _backward
        return outer
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
            
                for child in node._prev:
                    build_topo(child)

                topo.append(node)
        
        self.grad = 1.0
        build_topo(self)

        for node in reversed(topo):
            node._backward()
    
class Neuron:
    def __init__ (self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__ (self, x):
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return out.tanh()
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__ (self, nin, nn):
        self.neurons = [Neuron(nin) for _ in range(nn)]
    
    def __call__ (self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__ (self, nin, nns):
        dim = [nin] + nns
        self.layers = [Layer(dim[i], dim[i+1]) for i in range(len(nns))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

def trace(root):
    nodes, edges = set(), set()

    def build(node):
        if node not in nodes:
            nodes.add(node)
        for child in node._prev:
            edges.add((child, node))
            build(child)
    
    build(root)

    return nodes, edges

def buildDot(root):
    dot = gv.Digraph(format = "svg", graph_attr = {"rankdir" : "LR"})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{%s | data %.4f | grad %.2f }" % (n.label, n.data, n.grad), shape = "record")

        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1,n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# x = [4, 1, 3]
# m = MLP(3, [4,4,1])
# mx = m(x)
# mx.backward()

# print(mx.parameters())

# graph = buildDot(mx)
# graph.render()

# # inputs x1,x2
# x1 = Value(2.0, label='x1')
# x2 = Value(0.0, label='x2')
# # weights w1,w2
# w1 = Value(-3.0, label='w1')
# w2 = Value(1.0, label='w2')
# # bias of the neuron
# b = Value(6.8813735870195432, label='b')
# # x1*w1 + x2*w2 + b
# x1w1 = x1*w1; x1w1.label = 'x1*w1'
# x2w2 = x2*w2; x2w2.label = 'x2*w2'
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# n = x1w1x2w2 + b; n.label = 'n'
# # o = n.tanh(); o.label = 'o'

# e = (2*n).exp()
# o = (e - 1)/(e + 1)

# o.backward()
# graph = buildDot(o)
# graph.render()

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

m = MLP(3, [4,4,1])

for epoch in range(20):
    y_hat = [m(x) for x in xs]  
    
    loss = sum((y_hat - ys)**2 for y_hat, ys in zip(y_hat, ys))

    for p in m.parameters():
        p.grad = 0
    
    loss.backward()

    for p in m.parameters():
        p.data += -p.grad * 0.01

    if epoch % 5 == 0:
        print(f"epoch: {epoch} loss {loss} y: {y_hat}")