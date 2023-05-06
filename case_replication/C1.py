#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : C1.py
@Time : 2023/04/24 21:23:42
@Auth : Ming(<3057761608@qq.com>)
@Vers : 1.0
@Desc : 附录C灯泡案例计算结果的复现程序
@Usag : python3 C1.py > C1.out
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


class Bulb_ALT:
    def __init__(self):
        """初始化数据"""
        data = {
            '6.3 eV': [181.3, 288.4, 1178, 1511, 1706, 1723, 1964, 3157, 3224, 3483, 3555, 4939, 5443, 7807],
            '6.5 eV': [108.2, 335, 433.1, 496.4, 541.2, 820.6, 922.3, 928.6, 1053, 1517, 1520, 1952, 1954, 5176],
            '7.0 eV': [39.68, 212.3, 250.8, 261.3, 320.9, 374.8, 524.6, 626.6, 724.6, 830.8, 884.6, 1113, 1301, 1303],
            '7.5 eV': [59.06, 74.06, 134.9, 180.1, 201.4, 226.5, 235.3, 237, 281.5, 288.1, 356.1, 420.5, 513.2, 533.9],
            '8.0 eV': [31.73, 35.4, 36.07, 43.07, 60.18, 65.75, 78.7, 96.61, 105.4, 132.6, 168.5, 191.4, 218.3, 251.6]
        }
        self.pd_data = pd.DataFrame(data)

    def least_squares_estimate(self):
        """
        最小二乘估计
        """
        # 数学变化
        print('1. 经验分布函数值：')
        n = self.pd_data.shape[0]
        self.pd_data['y'] = [np.log(np.log(1/(1-j/(n+1))))
                             for j in range(1, n+1)]
        for i, S_j in enumerate(self.pd_data['y']):
            print(f'{i+1} {S_j:.2f}')
        for S_j in self.pd_data.columns[0:5]:
            plt.plot(np.log(self.pd_data[S_j]),
                     self.pd_data['y'], 'o', label=S_j)
        plt.legend()
        plt.xlim((3, 10))
        plt.ylim((-3, 1))
        plt.xlabel('$\ln{x}$')
        plt.ylabel('$\ln{\ln{(1/(1-F(t)))}}$')
        plt.savefig('C1-fig1-数学变换后的数对.png', dpi=300)
        print('数学变换后的数对图：C1-fig1')

        # 最小二乘估计
        print('2. 应力水平 S 下 m 和 eta 的最小二乘估计结果：')
        print('应力 m B eta')
        all_m = []
        all_eta = []
        for S_j in self.pd_data.columns[0:5]:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(self.pd_data[S_j]),
                                                                           self.pd_data['y'])
            all_m.append(slope)
            all_eta.append(np.exp(-intercept/slope))
            print(
                f'{S_j} {slope:.3f} {-intercept:.3f} {np.exp(-intercept/slope):.3f}')
            plt.plot(np.log(self.pd_data[S_j]), slope *
                     np.log(self.pd_data[S_j])+intercept, '--', color='black',)
        plt.savefig('C1-fig2-分布直线拟合结果.png', dpi=300)
        print('分布直线拟合结果：C1-fig2')

        # 威布尔分布参数 m 的估计
        m_average = np.mean(all_m)
        print(f'3. 威布尔分布参数 m 的估计值：{m_average:.4f}')

        # 加速模型中系数 a 与 b 的估计
        ln_V = np.log(np.array([eval(S_j.split(' ')[0])
                      for S_j in self.pd_data.columns[0:5]]))
        ln_eta = np.log(np.array(all_eta))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            ln_V, ln_eta)
        print(f'4. 加速模型中系数 a 与 b 的估计值：{intercept:.2f} {slope:.2f}')
        plt.figure()
        plt.plot(ln_V, ln_eta, 'o')
        plt.plot(ln_V, slope*ln_V+intercept, '--', color='black')
        plt.xlim((1.8, 2.1))
        plt.ylim((4.5, 8.5))
        plt.xlabel('$\ln{V}$')
        plt.ylabel('$\ln{\eta}$')
        plt.savefig('C1-fig3-电应力和寿命特征间的关系曲线.png', dpi=300)
        print('电应力和寿命特征间的关系曲线：C1-fig3')

        # 正常应力水平 S_0=6V 下分布参数及可靠性指标的估计
        S_0 = '6 V'
        m_0 = m_average
        eta_0 = np.exp(intercept+np.log(eval(S_0.split(' ')[0]))*slope)
        print(f'5. 正常应力水平 S_0 = 6 V 下分布参数的估计值：{m_0:.4f} {eta_0:.4f}')
        def F_t(t): return 1-np.exp(-(t/eta_0)**m_0)
        def R_t(t): return 1-F_t(t)
        plt.figure()
        x_data = np.linspace(0, 1.5e4)
        plt.plot(x_data, 1-F_t(x_data))
        plt.xlabel('time/h')
        plt.ylabel('Reliability')
        plt.xlim((0, 1.5e4))
        plt.ylim((0, 1))
        plt.savefig('C1-fig4-正常应力水平下的可靠性曲线（最小二乘估计）.png', dpi=300)
        print('正常应力水平下的可靠性曲线（最小二乘估计）：C1-fig4')

        t_ans = find_t(R_t, 0.5, 0, 1.5e4)
        print(f'产品的中位寿命 t(R=0.5)：{t_ans:.0f} h')

        print('>>!!注意：如果按照《指导书》中计算精度，相关结果如下：')
        a_ref = 31.60
        b_ref = -12.86
        m_0_ref = 1.2264
        eta_0_ref = np.exp(a_ref+b_ref*np.log(eval(S_0.split(' ')[0])))
        def F_t_ref(t): return 1-np.exp(-(t/eta_0_ref)**m_0_ref)
        def R_t_ref(t): return np.exp(-(t/eta_0_ref)**m_0_ref)
        t_ans_ref = find_t(R_t_ref, 0.5, 0, 1.5e4)
        print(f'正常应力水平 S_0 = 6 V 下分布参数的估计值：{m_0_ref:.4f} {eta_0_ref:.4f}')
        print(f'产品的中位寿命 t(R=0.5)：{t_ans_ref:.0f} h')

        return R_t

    def maximum_likelihood_estimate(self):
        def my_distribution(params, x):
            m, a, b = params
            t = x[:, 0]
            v = x[:, 1]
            return m*t**(m-1)*np.exp(-m*(a+b*np.log(v)))*np.exp(-(t*np.exp(-a-b*np.log(v)))**m)

        def log_likelihood(params, data):
            ll = np.sum(np.log(my_distribution(params, data)))
            return -ll

        data_tv = []
        for S_j in self.pd_data.columns[0:5]:
            for i in self.pd_data.index:
                t = self.pd_data.loc[i, S_j]
                v = eval(S_j.split(' ')[0])
                data_tv.append([t, v])

        data_tv = np.array(data_tv)
        init_params = [1, 1, 1]
        res = minimize(log_likelihood, init_params, args=(data_tv,))
        m, a, b = res.x
        S_0 = '6 V'
        eta_0 = np.exp(a+np.log(eval(S_0.split(' ')[0]))*b)
        print('极大似然方法估计值：')
        print(f'm = {m:.4f}')
        print(f'a = {a:.4f}')
        print(f'b = {b:.4f}')
        print(f'eta_0 = {eta_0:.1f}')
        def R_t(t): return np.exp(-(t/eta_0)**m)
        t_ans = find_t(R_t, 0.5, 0, 1.5e4)
        print(f'产品的中位寿命 t(R=0.5)：{t_ans:.0f} h')

        plt.figure()
        x_data = np.linspace(0, 1.5e4)
        plt.plot(x_data, R_t(x_data), '--', label='MLE')
        plt.xlabel('time/h')
        plt.ylabel('Reliability')
        plt.xlim((0, 1.5e4))
        plt.ylim((0, 1))
        plt.legend()
        plt.savefig('C1-fig5-正常应力水平下的可靠性曲线（极大似然估计）.png', dpi=300)
        print('正常应力水平下的可靠性曲线（极大似然估计）：C1-fig5')
        return R_t


def main():
    print('>>初始化数据...')
    bulb = Bulb_ALT()
    print('-----------------------------')
    print('>>最小二乘估计...')
    R1_t = bulb.least_squares_estimate()
    print('-----------------------------')
    print('>>极大似然估计...')
    R2_t = bulb.maximum_likelihood_estimate()

    print('-----------------------------')
    plt.figure()
    x_data = np.linspace(0, 1.5e4)
    plt.plot(x_data, R1_t(x_data), label='Least Squares')
    plt.plot(x_data, R2_t(x_data), '--', label='Maximum Likelihood')
    plt.xlabel('time/h')
    plt.ylabel('Reliability')
    plt.xlim((0, 1.5e4))
    plt.ylim((0, 1))
    plt.legend()
    plt.savefig('C1-fig6-正常应力水平下的可靠性曲线.png', dpi=300)
    print('正常应力水平下的可靠性曲线（两种方式对比）：C1-fig6')

    print('-----------------------------')
    print('  DONE<<')


if __name__ == '__main__':
    main()
