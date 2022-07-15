import lreg
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use("dark_background")

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
    t = time.time()
    b = fun(x,y)
    took = time.time()-t
    print(f"{fun.__name__} took {took}")
    b_str = np.array2string(b.reshape(-1), precision=5)
    print(f"{fun.__name__} fit result: {b_str}")
    return took

def test_linalg_solv(x,y):
    t = time.time()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    ones = np.ones(x.shape)
    A = np.hstack([x, ones])
    np.linalg.solve(A.T@A,A.T@y)
    took = time.time()-t
    print(f"np.linalg.solve took {took}")
    return took

def main():
    naive = lreg.naive

    a=3
    b=1.2
    NUM_POINTS = int(1e7)

    x = np.linspace(0,1,NUM_POINTS)
    y = a*x+ b + np.random.normal(size=NUM_POINTS)
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)

    lreg_dt = test_linear_reg(naive, x.reshape(-1), y.reshape(-1))
    pp_dt = test_linear_reg(naive_pure_python, x, y)
    test_linalg_solv(x,y)
        
    results = []
    t = np.linspace(10,int(1e6), 100)
    for NUMPOINTS in t :
        x = np.linspace(0,1,int(NUMPOINTS))
        y = a*x+ b + np.random.normal(size=int(NUMPOINTS))
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)

        lreg_dt = test_linear_reg(naive, x.reshape(-1), y.reshape(-1))
        pp_dt = test_linear_reg(naive_pure_python, x, y)
        np_ls = test_linalg_solv(x,y)
        results.append([lreg_dt, pp_dt, np_ls])

    results = np.array(results)
    plt.plot(t, results[:,0], label="naive")
    plt.plot(t, results[:,1], label="naive pure python")
    plt.plot(t, results[:,2], label="np.linalg.solve")
    plt.legend()
    plt.show()

        
if __name__ == "__main__":
    main()
    
