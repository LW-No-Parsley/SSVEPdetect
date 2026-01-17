import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA
from collections import defaultdict


class ssvepDetect:
    def __init__(self, srate, freqs, dataLen, use_filter_bank=True,
                 use_channel_ensemble=True, harmonics=5, n_subbands=7):
        self.cca = CCA(n_components=1)
        self.srate = srate
        self.freqs = freqs
        self.use_filter_bank = use_filter_bank
        self.use_channel_ensemble = use_channel_ensemble
        self.harmonics = harmonics
        self.n_subbands = n_subbands

        templLen = int(dataLen * srate)
        self.TemplateSet = []  # 信号模板
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        # 构建包含谐波的参考信号
        for freq in freqs:
            ref_signals = []
            for h in range(1, self.harmonics + 1):
                _ = 2 * np.pi * h * freq * sample
                ref_signals.extend([np.sin(_), np.cos(_)])
            tempset = np.vstack(ref_signals)
            self.TemplateSet.append(tempset)

        # 如果使用滤波器组，设计滤波器
        if self.use_filter_bank:
            self.subband_filters = self._design_filter_bank()
            # 子带权重 (FBCCA权重)
            self.subband_weights = [(n + 1) ** (-1.25) + 0.25 for n in range(self.n_subbands)]

    def _design_filter_bank(self):
        """设计滤波器组 - M3方法（覆盖多个谐波频带）"""
        f_low = 8
        f_high = 88
        filters = []

        for n in range(1, self.n_subbands + 1):
            lowcut = n * f_low - 2  # 扩展2Hz边界
            highcut = f_high
            filters.append((lowcut, highcut))

        return filters

    def _bandpass_filter(self, data, lowcut, highcut, order=4):
        """带通滤波"""
        nyq = 0.5 * self.srate
        low = lowcut / nyq
        high = highcut / nyq

        if high >= 1.0:
            high = 0.99

        b, a = scipysignal.cheby1(order, 1, [low, high], btype='band')
        filtered_data = scipysignal.filtfilt(b, a, data)
        return filtered_data

    def _channel_ensemble(self, data):
        """
        通道集成方法
        基于与参考通道（假设最后一个通道是Oz）的相关性构建通道组
        """
        n_channels = data.shape[0]
        ref_channel_idx = n_channels - 1  # 假设最后一个通道是Oz

        # 计算与其他通道的相关性
        correlations = []
        for ch in range(n_channels):
            if ch != ref_channel_idx:
                corr = np.corrcoef(data[ref_channel_idx], data[ch])[0, 1]
                correlations.append((ch, corr))

        # 按相关性排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        sorted_channels = [ref_channel_idx] + [ch for ch, _ in correlations]

        # 构建通道组：从2通道到全通道
        channel_groups = []
        for k in range(1, len(sorted_channels)):
            group = sorted_channels[:k + 1]
            channel_groups.append(group)

        return channel_groups

    def _cca_rv(self, data, template, freq_idx):
        """
        CCA-RV方法：减少频率间变异
        通过标准化处理提高分类稳定性
        """
        # 标准CCA计算
        cdata = data.transpose()
        ctemplate = template.transpose()

        self.cca.fit(cdata, ctemplate)
        datatran, templatetran = self.cca.transform(cdata, ctemplate)
        raw_score = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]

        # 应用简单的标准化（在实际系统中可以使用训练数据优化）
        # 这里使用平方来增强区分度
        adjusted_score = raw_score ** 2

        return adjusted_score

    def pre_filter(self, data):
        """改进的信号预处理"""
        # 将data为chs×N形式，即每一行是一个通道的数据

        # 1. 50Hz陷波滤波
        b_notch, a_notch = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)

        # 2. 改进的带通滤波：4-45Hz，更适合SSVEP
        fs = self.srate / 2
        N, Wn = scipysignal.ellipord([4 / fs, 45 / fs], [2 / fs, 50 / fs], 3, 40)
        b_band, a_band = scipysignal.ellip(N, 1, 90, Wn, 'bandpass')

        # 应用滤波
        filtered_data = scipysignal.filtfilt(b_band, a_band,
                                             scipysignal.filtfilt(b_notch, a_notch, data))

        return filtered_data

    def detect(self, data):
        """基础检测方法（保持向后兼容）"""
        data = self.pre_filter(data)
        cdata = data.transpose()
        p = []

        for template in self.TemplateSet:
            ctemplate = template.transpose()
            self.cca.fit(cdata, ctemplate)
            datatran, templatetran = self.cca.transform(cdata, ctemplate)
            coe = np.corrcoef(datatran[:, 0], templatran[:, 0])[0, 1]
            p.append(coe)

        return p.index(max(p))

    def detect_enhanced(self, data):
        """
        增强版检测方法：结合滤波器组和通道集成
        """
        data = self.pre_filter(data)

        # 如果不使用高级功能，回退到基础方法
        if not (self.use_filter_bank or self.use_channel_ensemble):
            return self.detect(data)

        final_scores = np.zeros(len(self.freqs))

        # 滤波器组处理
        if self.use_filter_bank:
            for subband_idx, (lowcut, highcut) in enumerate(self.subband_filters):
                # 子带滤波
                X_subband = np.array([self._bandpass_filter(ch_data, lowcut, highcut)
                                      for ch_data in data])

                subband_weight = self.subband_weights[subband_idx]

                # 通道集成处理
                if self.use_channel_ensemble:
                    channel_groups = self._channel_ensemble(X_subband)
                    subband_scores = np.zeros(len(self.freqs))

                    for group in channel_groups:
                        group_data = X_subband[group]
                        channel_weight = len(group) / data.shape[0]  # 通道组权重

                        # 计算与每个频率的相关性（使用CCA-RV）
                        for freq_idx, (freq, template) in enumerate(zip(self.freqs, self.TemplateSet)):
                            score = self._cca_rv(group_data, template, freq_idx)
                            subband_scores[freq_idx] += channel_weight * score

                    final_scores += subband_weight * subband_scores
                else:
                    # 不使用通道集成，直接计算
                    for freq_idx, template in enumerate(self.TemplateSet):
                        score = self._cca_rv(X_subband, template, freq_idx)
                        final_scores[freq_idx] += subband_weight * score
        else:
            # 仅使用通道集成
            channel_groups = self._channel_ensemble(data)
            for group in channel_groups:
                group_data = data[group]
                channel_weight = len(group) / data.shape[0]

                for freq_idx, template in enumerate(self.TemplateSet):
                    score = self._cca_rv(group_data, template, freq_idx)
                    final_scores[freq_idx] += channel_weight * score

        # 返回得分最高的频率索引
        return np.argmax(final_scores)

    def get_detection_scores(self, data):
        """
        获取详细的检测得分（用于调试和分析）
        """
        data = self.pre_filter(data)
        final_scores = np.zeros(len(self.freqs))

        if self.use_filter_bank:
            for subband_idx, (lowcut, highcut) in enumerate(self.subband_filters):
                X_subband = np.array([self._bandpass_filter(ch_data, lowcut, highcut)
                                      for ch_data in data])
                subband_weight = self.subband_weights[subband_idx]

                if self.use_channel_ensemble:
                    channel_groups = self._channel_ensemble(X_subband)
                    subband_scores = np.zeros(len(self.freqs))

                    for group in channel_groups:
                        group_data = X_subband[group]
                        channel_weight = len(group) / data.shape[0]

                        for freq_idx, template in enumerate(self.TemplateSet):
                            score = self._cca_rv(group_data, template, freq_idx)
                            subband_scores[freq_idx] += channel_weight * score

                    final_scores += subband_weight * subband_scores
                else:
                    for freq_idx, template in enumerate(self.TemplateSet):
                        score = self._cca_rv(X_subband, template, freq_idx)
                        final_scores[freq_idx] += subband_weight * score

        return final_scores