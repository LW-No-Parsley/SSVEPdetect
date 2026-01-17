import numpy as np
import csv
from ssvepdetect import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径
    datapath = r'../ExampleData（示例数据）/D1.csv'

    # 实验参数
    srate = 250  # 采样率250Hz
    dataLen = 4  # example数据长度为4秒

    # 实例化改进的ssvep检测器
    # 使用滤波器组CCA和通道集成方法
    sd = ssvepDetect(
        srate=srate,
        freqs=[8, 9, 10, 11, 12, 13, 14, 15],
        dataLen=dataLen,
        use_filter_bank=True,  # 启用滤波器组
        use_channel_ensemble=True,  # 启用通道集成
        harmonics=5,  # 使用5个谐波
        n_subbands=7  # 7个子带
    )

    # 读取数据
    data = []  # 用来存储所有原始数据

    # 打开CSV文件，读取数据
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        # 跳过第一行表头
        next(csv_reader)

        for row in csv_reader:
            # csv以字符串形式存储，需要转换成浮点型
            rowvalue = [float(_) for _ in row]
            # 所有数据整理后存入data列表中
            data.append(rowvalue)

    # 将列表型转换成np.array型，便于后续处理
    data = np.array(data, dtype=np.float64)

    points = dataLen * srate
    results = []
    stimIDs = []
    corr = []

    # 每个数据中都有48个片段
    for i in range(48):
        epoch = data[i * points:(i + 1) * points, :6]  # 把这一段的6个通道信号片段取出
        epoch = epoch.transpose()  # 以行来组织，每一行是一个通道的数据

        # 使用改进的检测方法
        res = sd.detect_enhanced(epoch)  # 使用增强版检测方法

        results.append(res)
        # 如果这是示例数据，则能够得到真值
        stim = int(data[i * points, -1])
        stimIDs.append(stim)

        if res == stim:
            correct = 1
        else:
            correct = 0

        corr.append(correct)

    accuracy = sum(corr) / 48
    print("改进后正确率： %.2f%%" % (accuracy * 100))

    # 性能分析
    print(f"\n性能分析:")
    print(f"- 正确分类: {sum(corr)}/48")
    print(f"- 错误分类: {48 - sum(corr)}/48")

    # 按频率分析准确率
    freq_accuracy = {}
    for freq in range(8, 16):
        indices = [i for i, stim in enumerate(stimIDs) if stim == freq - 8]
        if indices:
            correct_count = sum([corr[i] for i in indices])
            freq_accuracy[freq] = correct_count / len(indices)
            print(f"- 频率 {freq}Hz 准确率: {freq_accuracy[freq] * 100:.1f}%")

    # results里面包含了所有的预测值，应当按顺序填写到result.csv中，并将结果反馈至组委会
    for i in range(48):
        print("task%d预测值：%d, 真实值：%d, %s" % (i, results[i], stimIDs[i],
                                                  "正确" if results[i] == stimIDs[i] else "错误"))