import numpy as np
import scipy.sparse as sps

def solve_l2(w, lmbda):
    nw = np.linalg.norm(w)

    if nw > lmbda:
        x = (nw - lmbda) * w / nw
    else:
        x = np.zeros((len(w), 1))


def solve_l1l2(W, lmbda):
    n = W.shape[1]
    E = W
    for i in range(n):
        E[:, i] = solve_l2(W[:, i], lmbda)


def LDSR(view1, view2, lambda1, alpha, beta):
    # v1 = data[:, view1]
    # v2 = data[:, view2]

    X = [view1, view2]

    nv = len(X)

    max_mu = 1e10
    mu = 1e-4
    rho = 1.3
    epi = 1e-3
    iter = 0
    maxIter = 1e6
    tol = 1e-6

    x_dims = np.zeros((nv, 1))  # dimensions per each view
    ns = X[0].shape[0]  # number of samples/points
    # Q = cell(nv, 1); % {}
    Q = []
    # XtX = cell(nv, 1);
    XtX = []
    # E = cell(nv,1);
    E = []
    # Ze = cell(nv, 1); %Zr
    Ze = []
    sumXtX = 0

    x_dims = np.zeros((nv,1))

    for n in range(nv):
        # x_dims(n) = size(X{n},1)
        x_dims[n] = len(X[n])

    x_dims = x_dims.astype(int)

    for n in range(nv):
        for i in range(X[n].shape[1]):
            if np.linalg.norm(X[n][:, i]) >= 1e-8:
                X[n][:, i] = X[n][:, i] / np.linalg.norm(X[n][:, i])

        Ze.append(np.zeros((ns, ns)))
        XtX.append(X[n].dot(X[n].T))
        sumXtX = sumXtX + XtX[n]
        E.append(sps.csr_matrix((x_dims[n][0], ns)))
        Q.append(np.zeros((x_dims[n][0], ns)))

    P = np.zeros((ns, ns))
    Zc = np.zeros((ns, ns))

    while iter < maxIter:
        iter += 1
        temp = Zc + P / mu
        U, sigma, V = np.linalg.svd(temp)
        print(sigma.shape)
        #sigma = np.diag(sigma)
        print(sigma.shape)
        svp = len([x for x in sigma if x > 1 / mu])
        print(svp)

        if svp:
            sigma = sigma[0:svp] - 1 / mu
        else:
            svp = 1
            sigma = 0

        J = U[:, 0:svp].dot(sigma * V[:, 0:svp].T)

        temp1 = np.zeros(len(Ze[0]))
        temp2 = temp1
        temp3 = temp1

        for i in range(nv):
            temp1 = temp1 + XtX[i] * Ze[i]
            temp2 = temp2 + X[i].T.dot(E[i])
            temp3 = temp3 + X[i].T.dot(Q[i])

        temp4 = -P + mu * J + mu * sumXtX - mu * temp1 - mu * temp2 + temp3
        temp5 = mu * np.eye(ns) + mu * sumXtX
        # Basically temp5 \ temp4 in matlab syntax
        Zc = np.linalg.leastsq(temp5, temp4)

        break_flag = 0
        for i in range(nv):
            r = 0.5 / np.sqrt(np.diag(Ze[i].dot(Ze[i].T)) + epi)
            R = np.diag(r)
            temp1 = mu * X[i].T.dot(X[i].dot(Zc + E[i]) - X[i] - Q[i] / mu)
            temp2 = 2 * alpha * R + mu * X[i].T.dot(X[i])
            Ze[i] = np.linalg.lstsq(-temp2, temp1)

            temp1 = X[i] - X[i].dot(Zc) - X[i].dot(Ze[i])
            temp2 = temp1 + Q[i] / mu
            E[i] = solve_l1l2(temp2, beta / mu)

            leq1 = temp1 - E[i]
            Q[i] = Q[i] + mu * leq1

            if np.maximum(np.maximum(np.abs(leq1))) < tol:
                break_flag = 1
                break

            leq2 = Zc - J
            P = P + mu * leq2

            if break_flag and np.maximum(np.maximum(np.abs(leq2))) < tol:
                break

    consistency = []
    for i in range(ns):
        err1 = 0
        err2 = 0
        for j in range(nv):
            err1 = err1 + np.linagl.norm(Ze[j][:,i])
            if np.any(E[j][:,i]):
                err2 = err2 + np.linalg.norm(E[j][:,i])
            else:
                err2 = 0

        consistency.append(-err1 - lambda1*err2)

    consistency = np.array(consistency)
    return consistency


dat = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
v1 = [0,1,2,3]
v2 = [4,5,6,7,8]

ascores = LDSR(dat, v1, v2, 1, 1, 1)
print(ascores)
