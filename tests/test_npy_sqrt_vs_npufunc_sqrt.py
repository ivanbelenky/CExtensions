import numpy as np
import npufunc
import time
import matplotlib.pyplot as plt
plt.style.use("dark_background")

def test_sqrt(fun, arr):
    t = time.time()    
    sr = fun(arr)
    took = time.time() - t
    print(f"{fun.__name__} took:", took)
    return sr, took

arr = np.linspace(0.95, 1, int(1e5))

sr_np, t_np = test_sqrt(np.sqrt, arr)
sr_npu, t_npu = test_sqrt(npufunc.sqrt, arr)
sr_npu2, t_npu2 = test_sqrt(npufunc.sqrt2, arr)

""" print('Taking the GT to be numpy lets mean difference')
print(np.linalg.norm(sr_np-sr_npu)/np.linalg.norm(sr_np))
print(np.linalg.norm(sr_np-sr_npu2)/np.linalg.norm(sr_np))
 """
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(20,20))

fig.suptitle('Numpy vs. npufunc')

ax[0][0].plot(arr, sr_np, label=f'numpy T: {t_np:0.3e}')
ax[0][1].plot(arr, sr_np, label=f'numpy T: {t_np:0.3e}')

ax[0][0].plot(arr, sr_npu, label=f'npufunc.sqrt T: {t_npu:0.3e}')
ax[0][1].plot(arr, sr_npu2, label=f'npufunc.sqrt2 T: {t_npu2:0.3e}')

ax[1][0].plot(arr, np.abs(sr_np-sr_npu), label=f'ΔT sqrt: {(t_np-t_npu):0.3e}')
ax[1][1].plot(arr, np.abs(sr_np-sr_npu2), label=f'ΔT sqrt2: {(t_np-t_npu2):0.3e}')

ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()


ax[1][1].set_yscale('log')
ax[1][0].set_yscale('log')


plt.show()
