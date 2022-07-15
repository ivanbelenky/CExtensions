import lreg
import numpy as np
import time


def naive_pure_python(x,y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    ones = np.ones(x.shape)
    A = np.hstack([x, ones])
    At = A.T
    AtA = At@A
    AtA_1 = np.linalg.inv(AtA)
    b = AtA_1@(At.dot(y))
    return b

def test_linear_reg(fun, x, y):
    t=time.time()
    b=fun(x,y)
    took=time.time()-t
    print(f"{fun.__name__} took {took}")
    b_str = np.array2string(b.reshape(-1), precision=5)
    print(f"{fun.__name__} fit result: {b_str}")
    return b, took

def main():
    naive = lreg.naive

    a=3
    b=1.2
    NUM_POINTS = int(1e6)

    x = np.linspace(0,1,NUM_POINTS)
    y = a*x+ b + np.random.normal(size=NUM_POINTS)
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)

    lreg_b, lreg_dt = test_linear_reg(naive, x.reshape(-1), y.reshape(-1))
    pp_b, pp_dt = test_linear_reg(naive_pure_python, x, y)


if __name__ == "__main__":
    main()
    