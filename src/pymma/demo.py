from mma import mmasub
import numpy as np

m = 1
n = 8
iter = 3
xval = np.array(
    [
        [
            0.175301,
            0.175301,
            0.175301,
            0.175301,
            0.0240835,
            0.0240835,
            0.0240835,
            0.0240835,
        ]
    ]
)
xval = xval.T
xmin = np.full((n, m), 0.0)
xmax = np.full((n, m), 1.0)
xold2 = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
xold2 = xold2.T
xold1 = np.array(
    [
        [
0.166553, 0.166553, 0.166553, 0.166553, 0.00922103, 0.00922103, 0.00922103, 0.00922103
        ]
    ]
)
xold1 = xold1.T
f0val = 0
df0dx = np.array(
    [
        [
            -0.00979544,
            -0.00979544,
            -0.00979544,
            -0.00979544,
            -0.0142712,
            -0.0142712,
            -0.0142712,
            -0.0142712,
        ]
    ]
)
df0dx = df0dx.T
fval = -0.002462
dfdx = np.ones((m, n))
low = np.array(
    [
        [
            -0.333447,
            -0.333447,
            -0.333447,
            -0.333447,
            -0.490779,
            -0.490779,
            -0.490779,
            -0.490779,
        ]
    ]
)
low = low.T
upp = np.array(
    [[0.666553, 0.666553, 0.666553, 0.666553, 0.509221, 0.509221, 0.509221, 0.509221]]
)
upp = upp.T
a0 = 1
a = np.zeros((m, 1))
c = np.full((m, 1), 1000)
d = np.ones((m, 1))
move = 0.5
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
print(xmma)