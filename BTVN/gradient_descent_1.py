import numpy as np

def grad(x):
    return 2*x

def cost(x):
    return x**2 - 2

def gradient_descent(x0, eta):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, i)

(x1, i1) = gradient_descent(10, .1)
(x2, i2) = gradient_descent(-10, .1)
print('Solution x1 = %f, cost = %f, after %d iterations' % (x1[-1], cost(x1[-1]), i1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2[-1], cost(x2[-1]), i2))