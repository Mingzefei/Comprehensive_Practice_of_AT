#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : motor_ATL.py
@Time : 2023-05-31
@Auth : Ming(<3057761608@qq.com>)
@Vers : 1.0
@Desc : 电机加速寿命试验数据分析
@Usag : python3 motor_ATL.py
"""
# here put the import lib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('science')


def find_t(R, target_R, t_min, t_max, tol=1e-6):
    """
    使用二分法查找 R(t) = target_R 的 t 值
    """
    assert R(t_min) > target_R > R(t_max)

    while True:
        t_mid = (t_min + t_max) / 2
        R_mid = R(t_mid)
        if abs(R_mid - target_R) < tol:
            return t_mid
        if R_mid > target_R:
            t_min = t_mid
        else:
            t_max = t_mid

class Motor_ALT:
    def __init__(self):
        """初始化数据"""
        data_total = {
                '2.0 V': [41, 97, 203],
                '2.2 V': [164, 145, 34, 28, 179, 112],
                '2.3 V': [28, 16, 33, 88, 19, 31, 174, 39, 33, 21, 29, 49],
                '2.4 V': [22, 16, 12, 19, 37, 28],
                '2.5 V': [201, 22, 23, 20, 20, 43],
                '2.6 V': [54, 53, 138, 31, 186, 66],
                '2.7 V': [13, 11, 51],
                }
        data = { # 截尾处理
                '2.0 V': [41, 97, 203],
                '2.2 V': [164, 145, 34, 28,  112],
                '2.3 V': [28, 16, 33, 88, 19, 31,  39, 33, 21, 29, 49],
                '2.4 V': [22, 16, 12, 19, 37, 28],
                '2.5 V': [22, 23, 20, 20, 43],
                '2.6 V': [54, 53, 138, 31, 186, 66],
                '2.7 V': [13, 11, 51],
                }
        # 从小到大排序
        data_total = {S_j: sorted(data_total[S_j]) for S_j in data_total.keys()}
        self.data_total = data_total
        data = {S_j: sorted(data[S_j]) for S_j in data.keys()}
        self.data = data
        # 用 np.nan 补齐缺失值
        max_len = max([len(data[S_j]) for S_j in data.keys()])
        for S_j in data.keys():
            if len(data[S_j]) < max_len:
                data[S_j].extend([np.nan] * (max_len - len(data[S_j])))
        self.pd_data = pd.DataFrame(data)

    def plot_failure_data(self):
        """
        失效数据示意图
        """
        fig = plt.figure(figsize=(6, 4))
        for S_j in self.pd_data.columns:
            plt.plot(self.pd_data[S_j], [float(S_j.split(' ')[0])] * len(self.pd_data[S_j]),
                     'o', label=S_j)
        plt.legend()
        plt.xlabel('lifetime/min')
        plt.ylabel('voltage/V')
        plt.savefig('motor-fig0-失效数据示意图.png', dpi=300)

    def least_squares_estimate(self):
        """
        最小二乘估计
        """
        # 数学变化
        print('1. 经验分布函数值：')
        for i, S_j in enumerate(self.pd_data.columns):
            n = self.pd_data[S_j].count()
            n_total = len(self.data_total[S_j])
            y = [np.log(np.log(1/(1-j/(n_total+1)))) for j in range(1, n+1)]
            y.extend([np.nan] * (self.pd_data.shape[0] - n))
            self.pd_data[S_j+'-y'] = y
            print(f'{S_j} 的经验分布函数值：')
            # 打印非 nan
            for j in range(n):
                if not np.isnan(self.pd_data[S_j][j]):
                    print(f'{j+1} {self.pd_data[S_j+"-y"][j]:.2f}')
        fig = plt.figure(figsize=(6, 4), dpi=300)
        for S_j in self.pd_data.columns[0:7]:
            # plot 非 nan
            n = self.pd_data[S_j].count()
            plt.plot(np.log(self.pd_data[S_j][0:n]),
                     self.pd_data[S_j+'-y'][0:n], 'o', label=S_j)
        plt.legend()
        plt.xlabel('$\ln{t}$')
        plt.ylabel('$\ln{\ln{(1/(1-F(t)))}}$')
        plt.savefig('motor-fig1-数学变换后的数对.png', dpi=300)
        print('数学变换后的数对图：C1-fig1')

        # 最小二乘估计
        print('2. 应力水平 S 下 m 和 eta 的最小二乘估计结果：')
        print('应力 m B eta')
        all_m = []
        all_eta = []
        for S_j in self.pd_data.columns[0:7]:
            n = self.pd_data[S_j].count()
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(self.pd_data[S_j][0:n]), self.pd_data[S_j+'-y'][0:n])
            all_m.append(slope)
            all_eta.append(np.exp(-intercept/slope))
            print( f'{S_j} {slope:.3f} {-intercept:.3f} {np.exp(-intercept/slope):.3f}')
            plt.plot(np.log(self.pd_data[S_j][0:n]), slope *
                     np.log(self.pd_data[S_j][0:n])+intercept, '--', color='black',)
        plt.savefig('motor-fig2-分布直线拟合结果.png', dpi=300)
        print('分布直线拟合结果：C1-fig2')

        # 威布尔分布参数 m 的估计
        m_average = sum([all_m[i]*len(self.data_total[S_j]) for i, S_j in enumerate(self.data_total.keys())]) / sum([len(self.data_total[S_j]) for S_j in self.data_total.keys()])

        print(f'3. 威布尔分布参数 m 的估计值：{m_average:.4f}')

        # 加速模型中系数 a 与 b 的估计
        ln_V = np.log(np.array([eval(S_j.split(' ')[0]) for S_j in self.pd_data.columns[0:7]]))
        ln_eta = np.log(np.array(all_eta))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
                ln_V, ln_eta)
        print(f'4. 加速模型中系数 a 与 b 的估计值：{intercept:.2f} {slope:.2f}')
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(ln_V, ln_eta, 'o')
        plt.plot(ln_V, slope*ln_V+intercept, '--', color='black')
        plt.xlabel('$\ln{V}$')
        plt.ylabel('$\ln{\eta}$')
        plt.savefig('motor-fig3-电应力和寿命特征间的关系曲线.png', dpi=300)
        print('电应力和寿命特征间的关系曲线：C1-fig3')

        # 正常应力水平 S_0=6V 下分布参数及可靠性指标的估计
        S_0 = '2 V'
        m_0 = m_average
        eta_0 = np.exp(intercept+np.log(eval(S_0.split(' ')[0]))*slope)
        print(f'5. 正常应力水平 S_0 = 6 V 下分布参数 m_0 和 eta_0 的估计值：{m_0:.4f} {eta_0:.4f}')
        def F_t(t): return 1-np.exp(-(t/eta_0)**m_0)
        def R_t(t): return 1-F_t(t)
        fig = plt.figure(figsize=(6, 4), dpi=300)
        x_data = np.linspace(0, 3e2)
        plt.plot(x_data, 1-F_t(x_data))
        plt.xlabel('time/min')
        plt.ylabel('Reliability')
        plt.xlim((0, 3e2))
        plt.ylim((0, 1))
        plt.savefig('motor-fig4-正常应力水平下的可靠性曲线（最小二乘估计）.png', dpi=300)
        print('正常应力水平下的可靠性曲线（最小二乘估计）：C1-fig4')

        t_ans = find_t(R_t, 0.5, 0, 1e5)
        print(f'产品的中位寿命 t(R=0.5)：{t_ans:.0f} min')

        return R_t

    def maximum_likelihood_estimate(self):
        def my_distribution(params, x):
            m, a, b = params
            t = x[:, 0]
            v = x[:, 1]
            return m*t**(m-1)*np.exp(-m*(a+b*np.log(v)))*np.exp(-(t*np.exp(-a-b*np.log(v)))**m)

        def log_likelihood(params, data):
            m, a, b = params
            ll = np.sum(np.log(my_distribution(params, data))) 
            ll -= (179/np.exp(a+b*np.log(2.2)))**m
            ll -= (174/np.exp(a+b*np.log(2.3)))**m
            ll -= (201/np.exp(a+b*np.log(2.5)))**m
            return -ll



        data_tv = []
        for S_j in self.pd_data.columns[0:7]:
            for i in self.pd_data.index:
                t = self.pd_data.loc[i, S_j]
                v = eval(S_j.split(' ')[0])
                data_tv.append([t, v])

        data_tv = np.array(data_tv)
        data_tv = data_tv[~np.isnan(data_tv).any(axis=1)]
        init_params = [1, 1, 1]
        res = minimize(log_likelihood, init_params, args=(data_tv,))
        m, a, b = res.x
        S_0 = '2 V'
        eta_0 = np.exp(a+np.log(eval(S_0.split(' ')[0]))*b)
        print('极大似然方法估计值：')
        print(f'm = {m:.4f}')
        print(f'a = {a:.4f}')
        print(f'b = {b:.4f}')
        print(f'eta_0 = {eta_0:.1f}')
        def R_t(t): return np.exp(-(t/eta_0)**m)
        t_ans = find_t(R_t, 0.5, 0, 1e4)
        print(f'产品的中位寿命 t(R=0.5)：{t_ans:.0f} min')

        x_data = np.linspace(0, 3e2)
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(x_data, R_t(x_data), '--', label='MLE')
        plt.xlabel('time/h')
        plt.ylabel('Reliability')
        plt.xlim((0, 3e2))
        plt.ylim((0, 1))
        plt.legend()
        plt.savefig('motor-fig5-正常应力水平下的可靠性曲线（极大似然估计）.png', dpi=300)
        print('正常应力水平下的可靠性曲线（极大似然估计）：C1-fig5')
        return R_t


def main():
    print('>>初始化数据...')
    bulb = Motor_ALT()
    bulb.plot_failure_data()
    print('失效数据图示：motor-fig0')
    print('-----------------------------')
    print('>>最小二乘估计...')
    R1_t = bulb.least_squares_estimate()
    print('-----------------------------')
    print('>>极大似然估计...')
    R2_t = bulb.maximum_likelihood_estimate()

    print('-----------------------------')
    fig = plt.figure(figsize=(6, 4), dpi=300)
    x_data = np.linspace(0, 3e2)
    plt.plot(x_data, R1_t(x_data), label='Least Squares')
    plt.plot(x_data, R2_t(x_data), '--', label='Maximum Likelihood')
    plt.xlabel('time/h')
    plt.ylabel('Reliability')
    plt.xlim((0, 3e2))
    plt.ylim((0, 1))
    plt.legend()
    plt.savefig('motor-fig6-正常应力水平下的可靠性曲线.png', dpi=300)
    print('正常应力水平下的可靠性曲线（两种方式对比）：C1-fig6')

    print('-----------------------------')
    print('  DONE<<')


if __name__ == '__main__':
    main()
