#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : D.py
@Time : 2023/04/25 19:58:20
@Auth : Ming(<3057761608@qq.com>)
@Vers : 1.0
@Desc : 附录D轴承振动案例结果复现
@Usag : python3 D.py > D.out
"""
# here put the import lib
from scipy import io
from scipy import signal
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scienceplots
plt.style.use('science')


class Vibrating_signal:
    def __init__(self) -> None:
        data = io.loadmat('AB3X.mat')['AB3X']
        signal = np.array(data)
        self.signal = signal
        self.sample_rate = 20480  # 采样频率
        self.sample_interval = 2 * 60 * 60  # 采样间隔（2小时）
        self.samples_per_interval = self.sample_rate * \
            self.sample_interval  # 每个采集周期的采样点数
        self.num_intervals = 133  # 采样周期数

    def time_domain_analysis(self):
        """
        时域分析
        """
        # 峰值
        self.X_p = np.max(np.abs(self.signal), axis=0)
        # 峰峰值
        self.X_pp = np.max(self.signal, axis=0) - np.min(self.signal, axis=0)
        # 均值
        self.mu = np.mean(self.signal, axis=0)
        # 绝对均值
        self.X_aw = np.mean(np.abs(self.signal), axis=0)
        # 方差
        self.sigma2 = np.var(self.signal, axis=0)
        # 方根幅值
        self.X_r = np.square(np.mean(np.sqrt(np.abs(self.signal)), axis=0))
        # 均方值
        self.X_rms2 = np.mean(np.square(self.signal), axis=0)
        # 均方根值
        self.X_rms = np.sqrt(self.X_rms2)
        # 歪度指标
        self.SK_f = stats.skew(self.signal, axis=0)
        # 峭度指标
        self.K_f = stats.kurtosis(self.signal, axis=0, fisher=False)
        # 峰值指标
        self.C_f = self.X_p / self.X_rms
        # 脉冲指标
        self.I_f = self.X_p / self.X_aw
        # 裕度指标
        self.CL_f = self.X_p / self.X_r
        # 波形指标
        self.S_f = self.X_rms / self.X_aw

        # plot
        plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        t_data = np.arange(1, self.num_intervals+1)*self.sample_interval/3600
        # plot: 峰峰值 峭度指标
        plt.subplot(2, 2, 1)
        plt.plot(t_data, self.X_pp, label='peak to peak')
        plt.plot(t_data, self.K_f, label='kurtosis')
        plt.xlim(0, 300)
        plt.xlabel('time (h)')
        plt.ylabel('time domain parameters')
        plt.legend()
        plt.title('time domain parameters (1/4)')
        # plot: 脉冲指标 峰值指标 裕度指标
        plt.subplot(2, 2, 2)
        plt.plot(t_data, self.I_f, label='pulse index')
        plt.plot(t_data, self.C_f, label='peak factor')
        plt.plot(t_data, self.CL_f, label='margin factor')
        plt.xlim(0, 300)
        plt.xlabel('time (h)')
        plt.ylabel('time domain parameters')
        plt.legend()
        plt.title('time domain parameters (2/4)')
        # plot: 歪度指标 均方根值
        plt.subplot(2, 2, 3)
        plt.plot(t_data, self.SK_f, label='skewness')
        plt.plot(t_data, self.X_rms, label='RMS')
        plt.xlim(0, 300)
        plt.xlabel('time (h)')
        plt.ylabel('time domain parameters')
        plt.legend()
        plt.title('time domain parameters (3/4)')
        # plot: 波形指标
        plt.subplot(2, 2, 4)
        plt.plot(t_data, self.S_f, label='waveform index')
        plt.xlim(0, 300)
        plt.xlabel('time (h)')
        plt.ylabel('time domain parameters')
        plt.legend()
        plt.title('time domain parameters (4/4)')
        plt.title('time domain parameters (4/4)')
        # plt.ylim(1.26, 1.29)
        plt.xlabel('time (h)')
        plt.ylabel('time domain parameters')
        plt.legend()
        plt.title('time domain parameters (4/4)')
        plt.savefig('D-fig1-轴承振动时域参数.png', dpi=300)
        print('轴承振动时域参数图：D-fig1')

    def frequency_domain_analysis(self):
        """
        频域分析
        """
        # FFT method
        # fft_signal = np.fft.fft(self.signal, axis=0)
        # psd = np.abs(fft_signal)**2
        # self.total_energy = np.sum(psd, axis=0) # 全频域能量

        # # Welch method
        f, Pxx = signal.welch(
            self.signal, self.sample_rate, nperseg=20480, axis=0)
        self.total_energy = np.sum(Pxx, axis=0)  # 全频域能量

        plt.figure()
        t_data = np.arange(1, self.num_intervals+1)*self.sample_interval/3600
        plt.plot(t_data, self.total_energy)
        plt.xlim(0, 300)
        plt.xlabel('time (h)')
        plt.ylim(0.01, 0.015)
        plt.ylabel('total energy')
        plt.savefig('D-fig2-轴承振动全频域能量变换.png', dpi=300)
        print('轴承振动全频域能量变换图：D-fig2')

    def feature_dimensionality_reduce(self):
        """
        特征降维
        """
        # 9个特征数据（峰峰值、RMS值、脉冲指标、峰值指标、裕度指标、峭度指标、波形指标、歪度指标、全频域能量）
        feature_matrix = np.array([self.X_pp, self.X_rms, self.I_f, self.C_f,
                                  self.CL_f, self.K_f, self.S_f, self.SK_f, self.total_energy])
        # [0,1] 标准化
        feature_matrix = preprocessing.minmax_scale(feature_matrix, axis=1)
        # 累计贡献率
        pca = PCA(n_components=0.9)
        # PCA降维
        reduced_feature_matrix = pca.fit_transform(feature_matrix.T)

        print(f'累积贡献率：')
        sum = 0
        for i in range(reduced_feature_matrix.shape[1]):
            sum += pca.explained_variance_ratio_[i]
            print(f'第{i+1}个主成分累积贡献率：{sum:.4f}')

        # plot
        plt.figure()
        t_data = np.arange(1, self.num_intervals+1)*self.sample_interval/3600
        plt.plot(
            t_data, reduced_feature_matrix[:, 0], label='1st principal component')
        plt.plot(
            t_data, reduced_feature_matrix[:, 1], label='2nd principal component')
        plt.plot(
            t_data, reduced_feature_matrix[:, 2], label='3rd principal component')
        plt.xlim(0, 300)
        plt.xlabel('time (h)')
        plt.ylim(-1, 3)
        plt.ylabel('principal component')
        plt.legend()
        plt.savefig('D-fig3-轴承振动特征参数PCA降维结果.png', dpi=300)
        print('轴承振动特征参数PCA降维结果图：D-fig3')

        # plot: pareto图
        plt.figure()
        plt.bar(np.arange(1, 4), pca.explained_variance_ratio_)
        plt.plot(np.arange(1, 4), np.cumsum(
            pca.explained_variance_ratio_), '-o')
        for x, y in zip(np.arange(1, 4), pca.explained_variance_ratio_):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=10)
        plt.xticks(np.arange(1, 4), ['1st', '2nd', '3rd'])
        plt.ylim(0, 1)
        plt.xlabel('principal component')
        plt.ylabel('contribution rate')
        plt.savefig('D-fig4-轴承振动特征参数PCA降维累积贡献率.png', dpi=300)
        print('轴承振动特征参数PCA降维累积贡献率图：D-fig4')

        # 对第一个主成分进行线性拟合
        x = np.arange(1, self.num_intervals+1)*self.sample_interval/3600
        y = reduced_feature_matrix[:, 0]
        z = np.polyfit(x, y, 1)
        plt.ylabel('1st principal component')
        # plt.ylim(-1, 3)
        plt.legend()
        plt.savefig('D-fig5-轴承振动特征参数PCA降维第一主成分线性拟合.png', dpi=300)
        print('轴承振动特征参数PCA降维第一主成分线性拟合图：D-fig5')


def main():
    print('>>初始化数据...')
    vibrating_signal = Vibrating_signal()
    print('-----------------------------')
    print('>>时域分析...')
    vibrating_signal.time_domain_analysis()
    print('-----------------------------')
    print('>>频域分析...')
    vibrating_signal.frequency_domain_analysis()
    print('-----------------------------')
    print('>>特征降维...')
    vibrating_signal.feature_dimensionality_reduce()
    print('-----------------------------')
    print('  DONE<<')

    pass


if __name__ == '__main__':
    main()
