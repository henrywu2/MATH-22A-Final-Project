dpa = 4
dpb = 1

# custom inner product :)
def dp(v1, v2):
    out = 0
    for i in range(len(v1)):
        out += v1[i] * v2[i] / (((v1[i] - dpa)**2 + (v2[i] - dpb)**2)**0.5 + 1)
    return out

# idk why numpy doesn't work, so here are operations on lists/column vectors
def minus(v1, v2):
    out = []
    for i in range(len(v1)):
        out.append(v1[i] - v2[i])
    return out

def plus(v1, v2):
    out = []
    for i in range(len(v1)):
        out.append(v1[i] + v2[i])
    return out

def times(c, v):
    out = []
    for i in range(len(v)):
        out.append(c * v[i])
    return out

# projection of y onto x
def proj(v1, v2):
    return times(dp(v2, v1) / dp(v1, v1), v1)

if False:
    # Example 1: Fitting an equation of the form c0 + c1 * x + c2 * y = z to five data points in three-dimensional space
    A = [[1, 1, 1, 1, 1], [-2, 0, -6, 0, 3], [0, 2, -1, 1, 3]]
    b = [6, 5, 10, 2, -6]

    # Find an orthogonal basis for Col(A)
    u = []
    u.append(A[0])
    u.append(minus(A[1], proj(u[0], A[1])))
    u.append(minus(minus(A[2], proj(u[0], A[2])), proj(u[1], A[2])))
    print(u)

    # Find the projection of b onto Col(A)
    bhat = plus(plus(proj(u[0], b), proj(u[1], b)), proj(u[2], b))
    print(bhat)

if True:
    # Example 2: Fitting an equation of the form c0 + c1 * x = y to seven data points in two-dimensional space
    A = [[1, 1, 1, 1, 1, 1, 1], [-6, -4, -2, 0, 2, 4, 6]]
    b = [-2, 3, 2, 2, 5, 1, 8]

    # Find an orthogonal basis for Col(A)
    u = []
    u.append(A[0])
    u.append(minus(A[1], proj(u[0], A[1])))
    print(u)

    # Find the projection of b onto Col(A)
    bhat = plus(proj(u[0], b), proj(u[1], b))
    print(bhat)

if False:
    x = [-3, -10, 12]

    dist = proj([6, -2, 1], minus(x, [1, 1, 9]))
    print(1/(dp(dist, dist)**0.5 + 1))
    print(41**0.5/(abs(6*x[0] - 2*x[1] + x[2] - 13) + 41**0.5))

