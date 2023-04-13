import torch
import matplotlib.pyplot as plt
t1 = torch.tensor(4.0)
t1
t2 = torch.tensor([1.0, 2, 3, 4])
t2
t3 = torch.tensor([[1.0, 2], [3, 4], [5, 6]])
t3
t4 = torch.tensor([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9.0]]])
t4
t4.shape
u = torch.tensor(3.0, requires_grad=True)
v = torch.tensor(4.0, requires_grad=True)
(u, v)
f = u ** 3 + v ** 2
print(f)
f.backward()
print('\ndf/du :', u.grad)
print('df/dv :', v.grad)

def function(a, b, x):
    return a * x ** 3 + b * x ** 2
x = torch.linspace(-2.1, 2.1, 20)[:, None]
y = function(10, 3, x)
plt.scatter(x, y)
plt.title('Plot for $a=10$ and $b=3$')

def mae(truth, preds):
    return torch.abs(preds - truth).mean()
y2 = function(1, 1, x)
y3 = function(2, 1, x)
plt.scatter(x, y)
plt.plot(x, y2, c='orange')
plt.plot(x, y3, c='green')
plt.legend(['Data', 'a = 1, b = 1, MAE = {:.3f}'.format(mae(y, y2).item()), 'a = 4, b = 1, MAE = {:.3f}'.format(mae(y, y3).item())])

def calc_mae(args):
    y_new = function(*args, x)
    return mae(y, y_new)
calc_mae([5, 1])
ab = torch.tensor([1.1, 1], requires_grad=True)
ab
loss = calc_mae(ab)
loss
loss.backward()
ab.grad
with torch.no_grad():
    ab -= ab.grad * 0.02
    loss = calc_mae(ab)
print(f'loss={loss:.2f}')
for i in range(15):
    loss = calc_mae(ab)
    loss.backward()
    with torch.no_grad():
        ab -= ab.grad * 0.02
    print(f'step={i}; loss={loss:.2f}')
ab
(a, b) = ab.detach()
y2 = function(1, 1, x)
y3 = function(2, 1, x)
y4 = function(a, b, x)
plt.scatter(x, y)
plt.plot(x, y2, c='orange')
plt.plot(x, y3, c='green')
plt.plot(x, y4, c='red')
plt.legend(['Data', 'a = 1, b = 1, MAE = {:.3f}'.format(mae(y, y2).item()), 'a = 2, b = 1, MAE = {:.3f}'.format(mae(y, y3).item()), 'a = {:.1f}, b = {:.1f}, MAE = {:.3f}'.format(a, b, mae(y, y3).item())])