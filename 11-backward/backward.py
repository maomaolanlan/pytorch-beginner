import torch as t
from torch.autograd import Variable as v

# simple gradient
a = v(t.FloatTensor([2, 3]), requires_grad=True)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward()
print('*'*10)
print('=====simple gradient======')
print('input')
print(a.data)
print('compute result is')
print(out.item())
print('input gradients are')
print(a.grad.data)

# backward on non-scalar output
m = v(t.FloatTensor([[2, 3]]), requires_grad=True)
n = v(t.zeros(1, 2))
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
n.backward(t.FloatTensor([[1, 1]]))
print('*'*10)
print('=====non scalar output======')
print('input')
print(m.data)
print('input gradients are')
print(m.grad.data)

# jacobian
j = t.zeros(2 ,2)
k = v(t.zeros(1, 2))
m.grad.data.zero_()
k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
#retain_graph=True，这个参数默认是False，
#也就是反向传播之后这个计算图的内存会被释放，
#这样就没办法进行第二次反向传播了，所以我们需要设置为True，
#因为这里我们需要进行两次反向传播求得jacobian矩阵
k.backward(t.FloatTensor([[1, 0]]), retain_graph=True)
j[:, 0] = m.grad.data
m.grad.data.zero_()
k.backward(t.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('jacobian matrix is')
print(j)

# compute jacobian matrix
x = t.FloatTensor([2, 1]).view(1, 2)
x = v(x, requires_grad=True)
y = v(t.FloatTensor([[1, 2], [3, 4]]))

z = t.mm(x, y)
jacobian = t.zeros((2, 2))
z.backward(t.FloatTensor([[1, 0]]), retain_graph=True)  # dz1/dx1, dz1/dx2
jacobian[:, 0] = x.grad.data
x.grad.data.zero_()
z.backward(t.FloatTensor([[0, 1]]))  # dz2/dx1, dz2/dx2
jacobian[:, 1] = x.grad.data
print('=========jacobian========')
print('x')
print(x.data)
print('y')
print(y.data)
print('compute result')
print(z.data)
print('jacobian matrix is')
print(jacobian)
