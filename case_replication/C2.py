#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : C2.py
@Time : 2023/04/25 15:42:52
@Auth : Ming(<3057761608@qq.com>)
@Vers : 1.0
@Desc : 附录C2某高频接收装置案例结果复现
@Usag : python3 C2.py > C2.out
"""
# here put the import lib
from scipy import io
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('science')


class Receiver_ADT:
    """
    某高频接收装置的恒定应力加速退化试验
    """

    def __init__(self) -> None:
        self.Dt = 5
        self.threshold = 8
        self.S_0 = 20
        data = io.loadmat('feature2.mat')['feature_2']
        pd_data = pd.DataFrame(data)
        pd_data.columns = ['55-1', '55-2', '55-3', '70-4',
                           '70-5', '70-6', '80-7', '80-8', '80-9']
        pd_data['time'] = np.arange(1, pd_data.shape[0]+1)*self.Dt
        self.pd_data = pd_data

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        ax[0].plot(pd_data['time'], pd_data['55-1'], label='sample 1')
        ax[0].plot(pd_data['time'], pd_data['55-2'], label='sample 2')
        ax[0].plot(pd_data['time'], pd_data['55-3'], label='sample 3')
        ax[0].set_xlabel('time (h)')
        ax[0].set_ylabel('feature')
        ax[0].legend()
        ax[0].set_title('55 °C')

        ax[1].plot(pd_data['time'], pd_data['70-4'], label='sample 4')
        ax[1].plot(pd_data['time'], pd_data['70-5'], label='sample 5')
        ax[1].plot(pd_data['time'], pd_data['70-6'], label='sample 6')
        ax[1].set_xlabel('time (h)')
        ax[1].set_ylabel('feature')
        ax[1].legend()
        ax[1].set_title('70 °C')

        ax[2].plot(pd_data['time'], pd_data['80-7'], label='sample 7')
        ax[2].plot(pd_data['time'], pd_data['80-8'], label='sample 8')
        ax[2].plot(pd_data['time'], pd_data['80-9'], label='sample 9')
        ax[2].set_xlabel('time (h)')
        ax[2].set_ylabel('feature')
        ax[2].legend()
        ax[2].set_title('80 °C')

        plt.savefig('C2-fig1-试验数据图.png', dpi=300, bbox_inches='tight')
        print('某高频接收装置加速退化试验数据图：C2-fig1')

        pd_analysis = pd.DataFrame()
        pd_analysis['temperature'] = [55, 70, 80]
        self.pd_analysis = pd_analysis

    def parameter_estimation(self):
        """
        参数估计
        """
        # 用最小二乘法估计参数 a, b
        self.pd_data['55-average'] = self.pd_data[['55-1',
                                                   '55-2', '55-3']].mean(axis=1)
        self.pd_data['70-average'] = self.pd_data[['70-4',
                                                   '70-5', '70-6']].mean(axis=1)
        self.pd_data['80-average'] = self.pd_data[['80-7',
                                                   '80-8', '80-9']].mean(axis=1)
        for temperature in self.pd_analysis['temperature']:
            slope = stats.linregress(
                self.pd_data['time'], -self.pd_data[f'{temperature}-average']).slope
            self.pd_analysis.loc[self.pd_analysis['temperature']
                                 == temperature, 'd(S)'] = slope
            self.pd_analysis.loc[self.pd_analysis['temperature']
                                 == temperature, 'ln d(S)'] = np.log(slope)
        slop, intercept, r_value, p_value, std_err = stats.linregress(
            1/(273.15+self.pd_analysis['temperature']), self.pd_analysis['ln d(S)'])
        a, b = intercept, slop
        # 用极大似然估计参数 sigma （公式计算）
        for sample in self.pd_data.columns[:9]:
            self.pd_data[f'D-{sample}'] = self.pd_data[sample].diff()
        self.pd_data['D-time'] = pd.Series([float('NaN')] +
                                           [self.Dt]*(self.pd_data.shape[0]-1))
        sigma2 = 0
        for sample in self.pd_data.columns[:9]:
            sigma2 += ((self.pd_data[f'D-{sample}'] - np.exp(
                a+b/(273.15+eval(sample.split('-')[0])))*self.pd_data['D-time'])**2).sum()
        sigma2 /= self.Dt * 9 * (self.pd_data.shape[0]-1)
        simga = sigma2**0.5
        # 保存并输出结果
        self.a = a
        self.b = b
        self.simga = simga
        print(f'a = {a:.4f}')
        print(f'b = {b:.0f}')
        print(f'sigma = {simga:.3f}')

    def relibility_evaluation(self):
        """
        可靠性评估
        """
        def R(t):
            ans = norm.cdf((self.threshold - np.exp(self.a+self.b/(273.15+self.S_0))*t)/(self.simga*t**0.5))\
                - np.exp(2*np.exp(self.a+self.b/(273.15+self.S_0))*self.threshold/(self.simga**2))\
                * norm.cdf(-(self.threshold + np.exp(self.a+self.b/(273.15+self.S_0))*t)/(self.simga*t**0.5))
            return ans

        plt.figure()
        t_data = np.linspace(1, 1e5)
        plt.plot(t_data, R(t_data))
        plt.xlim(0, 1e5)
        plt.ylim(0, 1)
        plt.xlabel('time (h)')
        plt.ylabel('Reliability')
        plt.savefig('C2-fig2-高频接收装置可靠度估计结果.png', dpi=300)
        print('某高频接收装置可靠度估计结果：C2-fig2')

        # 平均寿命计算
        mttf = self.threshold / np.exp(self.a+self.b/(273.15+self.S_0))
        print(f'平均寿命：{mttf:.0f} h')

        print('>>!!注意：如果按照《指导书》中计算精度，相关结果如下：')
        a = 0.7867
        b = -2765
        sigma = 0.017
        mttf = self.threshold / np.exp(a+b/(273.15+self.S_0))
        print(f'a = {a:.4f}')
        print(f'b = {b:.0f}')
        print(f'sigma = {sigma:.3f}')
        print(f'平均寿命：{mttf:.0f} h')


def main():
    print('>>初始化数据...')
    receiver = Receiver_ADT()
    print('-----------------------------')
    print('>>参数估计...')
    receiver.parameter_estimation()
    print('-----------------------------')
    print('>>可靠性评估...')
    receiver.relibility_evaluation()
    print('-----------------------------')
    print('  DONE<<')
    pass


if __name__ == '__main__':
    main()
