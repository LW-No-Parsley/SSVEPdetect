# SSVEP检测系统

## 项目概述

这是一个用于稳态视觉诱发电位（SSVEP）信号检测的Python实现系统。系统使用了多种先进的信号处理技术，包括滤波器组CCA（FBCCA）和通道集成方法，以提高频率检测的准确率。

## 文件结构

```
├── demo.py                    # 主程序：数据读取、处理和评估
├── ssvepdetect.py            # SSVEP检测核心算法实现
└── ../ExampleData（示例数据）/
    └── D1.csv               # 示例数据文件
```

## 核心功能

### 1. 信号预处理
- **50Hz陷波滤波**：去除工频干扰
- **4-45Hz带通滤波**：提取SSVEP相关频段
- **椭圆滤波器设计**：提供陡峭的过渡带

### 2. 参考信号生成
- 支持多谐波参考信号（默认5个谐波）
- 频率范围：8-15Hz（可配置）
- 包含正弦和余弦分量

### 3. 检测算法
#### 基础算法
- **标准CCA**：典型相关分析

#### 增强算法（默认启用）
- **滤波器组CCA（FBCCA）**：
  - 7个子带滤波器
  - 覆盖8-88Hz范围
  - M3滤波器设计方法
  - 子带权重优化

- **通道集成**：
  - 基于通道相关性分组
  - 假设Oz通道为参考通道
  - 动态构建通道组

- **CCA-RV改进**：
  - 减少频率间变异
  - 标准化处理
  - 平方增强区分度

## 使用方法

### 1. 数据格式要求
- CSV格式，包含表头
- 每行代表一个时间点的数据
- 前6列为EEG通道数据
- 最后一列为刺激频率标签（0-7对应8-15Hz）
- 采样率：250Hz
- 数据长度：4秒（1000个采样点）

### 2. 基本配置

```python
# 创建检测器实例
sd = ssvepDetect(
    srate=250,                  # 采样率
    freqs=[8, 9, 10, 11, 12, 13, 14, 15],  # 刺激频率
    dataLen=4,                  # 数据长度（秒）
    use_filter_bank=True,       # 启用滤波器组
    use_channel_ensemble=True,  # 启用通道集成
    harmonics=5,                # 谐波数量
    n_subbands=7                # 子带数量
)
```

### 3. 数据读取与处理

```python
# 读取数据
with open(datapath, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # 跳过表头
    
    for row in csv_reader:
        rowvalue = [float(_) for _ in row]
        data.append(rowvalue)

data = np.array(data, dtype=np.float64)

# 分割数据片段
for i in range(48):  # 假设有48个片段
    epoch = data[i * points:(i + 1) * points, :6].transpose()
    
    # 使用增强检测
    result = sd.detect_enhanced(epoch)
```

## API参考

### ssvepDetect类

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `srate` | int | 必需 | 采样率（Hz） |
| `freqs` | list | 必需 | 刺激频率列表 |
| `dataLen` | float | 必需 | 数据长度（秒） |
| `use_filter_bank` | bool | True | 启用滤波器组 |
| `use_channel_ensemble` | bool | True | 启用通道集成 |
| `harmonics` | int | 5 | 谐波数量 |
| `n_subbands` | int | 7 | 子带数量 |

#### 主要方法

1. **`detect(data)`** - 基础检测方法
   - 输入：`chs × N` 格式的EEG数据
   - 返回：预测的频率索引

2. **`detect_enhanced(data)`** - 增强检测方法
   - 结合滤波器组和通道集成
   - 返回：预测的频率索引

3. **`get_detection_scores(data)`** - 获取详细得分
   - 返回：各频率的得分数组
   - 用于调试和分析

4. **`pre_filter(data)`** - 信号预处理
   - 50Hz陷波滤波 + 4-45Hz带通滤波

## 性能评估

系统提供了详细的性能分析功能：

```python
# 计算总准确率
accuracy = sum(corr) / 48
print("改进后正确率： %.2f%%" % (accuracy * 100))

# 按频率分析准确率
for freq in range(8, 16):
    indices = [i for i, stim in enumerate(stimIDs) if stim == freq - 8]
    if indices:
        correct_count = sum([corr[i] for i in indices])
        freq_accuracy[freq] = correct_count / len(indices)
```

## 输出格式

程序输出包括：
1. 总体准确率
2. 正确/错误分类统计
3. 各频率准确率
4. 每个任务的详细结果：
   ```
   task0预测值：2, 真实值：2, 正确
   task1预测值：5, 真实值：4, 错误
   ...
   ```

## 技术特点

### 优势
1. **高准确率**：通过FBCCA和通道集成显著提升检测性能
2. **鲁棒性**：CCA-RV方法减少频率间变异
3. **灵活性**：可单独启用/禁用各项功能
4. **可扩展性**：易于添加新的特征提取方法

### 滤波器组设计
- 采用M3方法覆盖多个谐波频带
- 子带边界扩展2Hz以捕获更多能量
- 权重公式：`(n+1)^(-1.25) + 0.25`

### 通道集成策略
1. 计算所有通道与参考通道（Oz）的相关性
2. 按相关性降序排序
3. 构建从2通道到全通道的多个通道组
4. 加权融合各通道组的结果

## 注意事项

1. **数据格式**：确保EEG数据格式正确，通道顺序一致
2. **采样率**：系统设计为250Hz采样率，使用其他采样率需调整滤波器参数
3. **通道假设**：通道集成方法假设最后一个通道是Oz，如果通道布局不同需修改代码
4. **计算资源**：滤波器组方法会增加计算量，但精度更高

## 依赖库

```python
numpy >= 1.19.0
scipy >= 1.5.0
scikit-learn >= 0.24.0
```

## 应用场景

1. 脑机接口（BCI）系统
2. SSVEP研究实验
3. 神经反馈训练
4. 认知状态监测

## 未来改进方向

1. 添加更多特征提取方法（如PSDA、TRCA）
2. 支持实时处理
3. 自适应滤波器设计
4. 深度学习集成

---

*本系统为SSVEP检测提供了一套完整的解决方案，结合了传统信号处理和现代机器学习方法，在保证实时性的同时提供高准确率的频率检测。*
