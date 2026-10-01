import numpy as np
def f(x):
    return x**2 - 4*x + 5
def df(x):
    return 2*x - 4
x = 5.0
eta = 0.2
print(f"Khoi tao: x(0) = {x:.4f}, f(x(0)) = {f(x):.4f}\n")
for t in range(1, 5):
    grad = df(x)
    x = x - eta * grad
    print(f"Buoc {t}: x({t}) = {x:.4f}, f(x({t})) = {f(x):.6f}, grad = {grad:.4f}")