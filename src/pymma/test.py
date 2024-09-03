import numpy as np
from mma import mmasub

n = 125000
m = 1
iter = 8
xval = np.fromfile("src/pymma/data/xval.bin", dtype=">f8")
xold1 = np.fromfile("src/pymma/data/xold1.bin", dtype=">f8")
xold2 = np.fromfile("src/pymma/data/xold2.bin", dtype=">f8")
upp = np.fromfile("src/pymma/data/upp.bin", dtype=">f8")
low = np.fromfile("src/pymma/data/low.bin", dtype=">f8")
df0dx = np.fromfile("src/pymma/data/dc.bin", dtype=">f8")
x = np.fromfile("src/pymma/data/x.bin", dtype=">f8")
x = x[1:]
x = x.reshape((n, 1))
xval = xval[1:]
xval = xval.reshape((n, 1))
xold1 = xold1[1:]
xold1 = xold1.reshape((n, 1))
xold2 = xold2[1:]
xold2 = xold2.reshape((n, 1))
upp = upp[1:]
upp = upp.reshape((n, 1))
low = low[1:]
low = low.reshape((n, 1))
df0dx = df0dx[1:]
df0dx = df0dx.reshape((n, 1))
f0val = 0
fval = np.sum(xval) - n * 0.1
a0 = 1
a = np.zeros((m, 1))
c = np.full((m, 1), 1000)
d = np.ones((m, 1))
move = 0.5
xmin = np.zeros((n, 1))
xmax = np.ones((n, 1))
dfdx = np.ones((m, n))
xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
    m,
    n,
    iter,
    xval,
    xmin,
    xmax,
    xold1,
    xold2,
    f0val,
    df0dx,
    fval,
    dfdx,
    low,
    upp,
    a0,
    a,
    c,
    d,
    move,
)
print((np.max(np.abs(xmma - x))))
