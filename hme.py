import numpy as np
import numpy.random as random
import scipy.stats as ss


class hme:
    def __init__(self, X, Y, levels=3, branches=2):
        # Gaussian assumption of experts.
        self.levels = levels
        self.branches = branches

        self.gates = []
        self.experts = []

        v_x = np.shape(X)[1]  # dimension of sample x[n]
        self.N = np.shape(X)[0]  # number of samples.
        v_y = np.shape(Y)[1]  # dimension of y[n]

        self.n_gates = 2 ** levels - 1  # number of gate nodes
        self.n_experts = 2 ** levels  # number of expert node.

        self.gates.append('')  # self.gates 1st element is empty
        for i in range(self.n_gates + 1):
            v1 = random.random((v_x, 1))
            v2 = random.random((v_x, 1))
            v1_part1 = np.zeros((v_x, 1))
            v1_part2 = np.zeros((v_x, v_x))
            v2_part1 = np.zeros((v_x, 1))
            v2_part2 = np.zeros((v_x, v_x))

            g1 = 0.0
            g2 = 0.0
            h1 = 0.0
            h2 = 0.0
            hij_1 = 0.0
            hij_2 = 0.0
            out = np.zeros(v_y)
            self.gates.append(
                ((v1, v2), (g1, g2), (h1, h2), (hij_1, hij_2), (v1_part1, v1_part2), (v2_part1, v2_part2), out))
        for i in range(self.n_experts):
            U = random.random((v_y, v_x))
            U_part1 = np.zeros((v_y, v_x))
            U_part2 = np.zeros((v_x, v_x))
            sigma = np.diag(random.random(v_y))
            prob = 0.0
            y_out = np.zeros(v_y)
            self.experts.append(((U, sigma), prob, U_part1, U_part2, y_out))

    def compute_p_y(self, x, y):  # x,y are both row vector
        for i in range(self.n_experts):
            U = np.mat(self.experts[i][0][0])
            mean = U * np.mat(x).T
            cov = np.mat(self.experts[i][0][1])
            p_y = ss.multivariate_normal.pdf(y, mean, cov)
            self.experts[i][1] = p_y

    def compute_y_out(self, x):
        for i in range(self.n_experts):
            U = np.mat(self.experts[i][0][0])
            mean = U * np.mat(x).T
            self.experts[i][4] = mean

    # def compute_epsl_ofexpert(self, x, index_ofexpertnode):
    #     i = self.visualindex_of_expertnode(index_ofexpertnode)
    #     parent = self.parent(i)
    #     v = self.gates[parent][0][i % 2]
    #     return np.dot(v, x)
    #
    # def compute_epsl_ofgate(self, x, indexofgatenode):
    #     parent = self.parent(indexofgatenode)
    #     v = self.gates[parent][0][indexofgatenode % 2]
    #     return np.dot(v, x)
    #
    # def compute_h(self,x,y):
    #     P_y = self.compute_p_y(x,y)

    def compute_g_downward(self, x):
        for index in range(1, self.n_gates + 1):
            v1, v2 = self.gates[index][0]
            g1, g2 = self.softmax(np.dot(v1, x), np.dot(v2, x))
            self.gates[index][1] = (g1, g2)  # tuple replace.

    def compute_h_joni_upward(self, x, y):
        self.compute_p_y(x, y)
        self.compute_g_downward(x)
        for index in range(self.n_gates, 0, -1):
            g1, g2 = self.gates[index][1]
            p_y_left = self.experts[self.lef_child(index)][1]
            p_y_right = self.experts[self.right_child(index)][1]
            h1 = g1 * p_y_left / (np.dot([g1, g2], [p_y_left, p_y_right]))
            h2 = 1 - h1
            self.n_gates[index][2] = (h1, h2)

    def compute_h(self, x, y):  # downward.
        self.compute_h_joni_upward(x, y)
        self.gates[1][3] = self.gates[1][2]
        for index in range(2, self.n_gates + 1):
            parent = self.parent(index)
            h1, h2 = self.gates[index][2]
            hij_1 = h1 * self.gates[parent][3][index % 2]
            hij_2 = h2 * self.gates[parent][3][index % 2]
            self.gates[index][3] = (hij_1, hij_2)

    def para_estimation(self, X, Y):
        for t in range(self.N):
            x_t = X[t]
            y_t = Y[t]
            self.compute_h(x_t, y_t)
            for i in range(self.n_experts):
                vindex = self.visualindex_of_expertnode(i)
                hij_t = self.gates[self.parent(vindex)][3][vindex % 2]
                self.experts[i][2] += np.mat(y_t).T * np.mat(x_t) * hij_t
                self.experts[i][3] += np.mat(x_t).T * np.mat(x_t) * hij_t
            for index in range(2, self.n_gates + 1):
                h_1, h_2 = self.gates[index][2]
                hi = self.gates[self.parent(index)][3][index % 2]
                self.gates[index][4][0] += np.log(h_1) * np.mat(x_t) * hi
                self.gates[index][4][1] += np.mat(x_t).T * np.mat(x_t) * hi
                self.gates[index][5][0] += np.log(h_2) * np.mat(x_t) * hi
                self.gates[index][5][1] += np.mat(x_t).T * np.mat(x_t) * hi
            # for the first  gatenode:
            h_1, h_2 = self.gates[1][2]
            hi = 1.0
            self.gates[1][4][0] += np.log(h_1) * np.mat(x_t) * hi
            self.gates[1][4][1] += np.mat(x_t).T * np.mat(x_t) * hi
            self.gates[1][5][0] += np.log(h_2) * np.mat(x_t) * hi
            self.gates[1][5][1] += np.mat(x_t).T * np.mat(x_t) * hi
        for i in range(self.n_experts):
            U_part1 = self.experts[i][2]
            U_part2 = self.experts[i][3]
            self.experts[i][0][0] = np.mat(U_part1) * np.mat(U_part2).I
        for index in range(1, self.n_gates + 1):
            v1_part1, v1_part2 = self.gates[index][4]
            v2_part1, v2_part2 = self.gates[index][5]
            self.gates[index][0][0] = (np.mat(v1_part1) * np.mat(v1_part2).I).T
            self.gates[index][0][1] = (np.mat(v2_part1) * np.mat(v2_part2).I).T

    def prediction(self, x_new):
        self.compute_g_downward(x_new)
        self.compute_y_out(x_new)
        number_lastgatelevel = self.n_experts / 2
        for index in range(self.n_gates, 0, -1):
            g1, g2 = self.gates[index][1]
            l_child = self.lef_child(index)
            r_child = self.right_child(index)
            if (index >= self.visualindex_of_expertnode(0) / 2):
                y1_out = self.experts[l_child][4]
                y2_out = self.experts[r_child][4]
            else:
                y1_out = self.gates[l_child][6]
                y2_out = self.gates[r_child][6]
            self.gates[index][6] = g1*y1_out+ g2*y2_out
        return self.gates[1][6]

    def softmax(self, a, b):
        max_ = np.max(a, b)
        p_a = np.exp(a - max_)
        p_b = np.exp(b - max_)
        p_a = p_a / (p_a + p_b)
        return p_a, 1 - p_a

    def visualindex_of_expertnode(self, i):
        return i + self.n_gates + 1

    def parent(self, i):
        return int(i / 2)

    def lef_child(self, i):
        left = 2 * i
        if (left > self.n_gates):
            return left - self.n_gates - 1
        return 2 * i

    def right_child(self, i):
        right = 2 * i + 1
        if (right > self.n_gates):
            return right - self.n_gates - 1
        return 2 * i + 1
