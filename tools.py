import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import welch
from scipy.signal import stft


# 加载数据集
def load(file_path):
    data = pd.read_csv(file_path, header=None)
    return data

# 打印数据集的描述信息
def describe_data(data):
    data_head = data.head()
    data_description = data.describe()

    print(data_head)
    print(data_description)


# 绘制数据集的直方图、箱线图
def plot_data(data):
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title('Boxplot of Bearing Data')
    plt.xlabel('Value')
    plt.show()

    # Create a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=100, kde=True)
    plt.title('Histogram of Bearing Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# 查找数据集的空值
def find_null(data):
    # Check for missing values
    missing_values = data.isnull().sum()

    # Perform basic statistical analysis
    statistical_summary = data.describe()

    print("Missing values: \n", missing_values)
    print("Statistical summary: \n", statistical_summary)


# 删除异常值
def drop_outliers(data):
    # Calculate IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identifying the outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_count = outliers.count()

    # Percentage of outliers
    outlier_percentage = (outlier_count / len(data)) * 100

    # Removing the outliers
    cleaned_data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

    # Printing the results
    # print('Total number of outliers: {}'.format(outlier_count.sum()))
    # print('Percentage of outliers: {}%'.format(outlier_percentage.sum()))

    return cleaned_data

# 标准化数据
def standardize_data(data):
    # Standardize the data
    standardized_data = (data - data.mean()) / data.std()

    # Print the mean and standard deviation of the standardized data
    # print('Mean of standardized data: {0}'.format(standardized_data.mean()))
    # print('Standard deviation of standardized data: {0}'.format(standardized_data.std()))

    return standardized_data


# 时域分析统计参数:计算信号的均值、标准差、偏度、峰度等，这些参数有助于识别故障的特征
def time_domain_features(data):
    # Basic time domain features
    mean_value = data.iloc[:, 0].mean()  # Mean
    std_dev = data.iloc[:, 0].std()  # Standard deviation
    max_value = data.iloc[:, 0].max()  # Maximum value
    min_value = data.iloc[:, 0].min()  # Minimum value
    peak_to_peak = max_value - min_value  # Peak-to-peak amplitude

    # Prepare a summary of the time domain features
    time_domain_features = {
        "Mean": mean_value,
        "Standard Deviation": std_dev,
        "Maximum Value": max_value,
        "Minimum Value": min_value,
        "Peak-to-Peak Amplitude": peak_to_peak
    }

    print(time_domain_features)

# 功率谱密度（PSD）：提供了每个频率分量对信号总能量的贡献。
def power_spectral_density(data, sampling_rate=50000):
    # 获取加速度信号
    acceleration = data[0].values

    # 计算功率谱密度
    frequencies, psd = welch(acceleration, sampling_rate, nperseg=1024)

    # 绘制功率谱密度图
    plt.figure(figsize=(12, 6))
    plt.semilogy(frequencies, psd)
    plt.title('Power Spectral Density of Bearing Data')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.grid(True)
    plt.show()

# 绘制时域、频域特征
# 傅里叶变换:将信号从时域转换到频域，分析其频谱特性。对轴承故障诊断特别有用，因为不同类型的故障往往在特定频率上有明显的峰值
def plot_time_domain_features(data, sampling_rate=50000, subset_length=1):
    # Time domain analysis

    # Sampling rate is 50 kHz
    total_samples = data.shape[0]

    # Time vector (in seconds)
    time_vector = np.arange(total_samples) / sampling_rate

    # Selecting a subset of data for time domain analysis
    # For example, selecting the first 1 second of data
    subset_length = subset_length * sampling_rate  # 1 second of data
    subset_data = data.iloc[:subset_length]

    # Time vector for the subset
    subset_time_vector = np.arange(subset_length) / sampling_rate

    # Plotting the subset of time series
    plt.figure(figsize=(15, 6))
    plt.plot(subset_time_vector, subset_data, label='Vibration Data')
    plt.title('Time Series of Bearing Data (First 1 Second)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Frequency domain analysis using FFT
    fft_result = np.fft.fft(data.iloc[:, 0])
    frequencies = np.fft.fftfreq(total_samples, 1 / sampling_rate)

    # Plotting the frequency spectrum
    plt.figure(figsize=(15, 6))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('Frequency Spectrum of Bearing Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, sampling_rate / 2])  # Display only positive frequencies up to Nyquist frequency
    plt.show()


# 数据分段
def segment_data(data, sampling_rate=50000, seconds=0.2, label=1):
    # Total number of samples after cleaning
    total_samples = data.shape[0]

    # Define the sample length for 0.2 seconds
    sample_length = int(seconds * sampling_rate)  # 0.2 seconds

    # Number of samples
    num_samples = total_samples // sample_length

    # Splitting the data into segments
    segments = [data.iloc[i * sample_length:(i + 1) * sample_length] for i in range(num_samples)]

    # Label for each segment (all positive samples)
    labels = [label for _ in range(num_samples)]

    return segments, labels


# Function to estimate the period of a signal
def estimate_period(signal):
    # Find all local maxima
    maxima_indices = np.diff(np.sign(np.diff(signal))).tolist()
    peaks = [i for i, x in enumerate(maxima_indices, start=1) if x == -2]

    # Estimate the period by averaging the distance between peaks
    if len(peaks) > 1:
        estimated_periods = np.diff(peaks)
        average_period = np.mean(estimated_periods)
        return average_period, peaks
    else:
        return None, None

# 短时傅里叶变换（STFT）：分析信号在不同时间点的频率分布，适用于非平稳信号的分析
"""这个我测试出来效果很差，有可能是我参数没调对，也有可能是这个数据本身不适合做STFT分析，你自己考虑一下"""
def plot_spectrogram(data, sampling_rate=50000, nperseg_stft = 256, noverlap_stft = 128):
    # 获取加速度信号
    acceleration = data[0].values
    # 执行STFT
    frequencies_stft, times_stft, Zxx = stft(acceleration, sampling_rate, nperseg=nperseg_stft, noverlap=noverlap_stft)
    # 绘制STFT的时频图
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times_stft, frequencies_stft, np.abs(Zxx), shading='gouraud')
    plt.title('Short-Time Fourier Transform (STFT) of Bearing Data')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Magnitude')
    plt.show()


if __name__ == '__main__':
    file_path = 'data/ib600_2.csv'

    data = load(file_path)

    plot_spectrogram(data, sampling_rate=50000, nperseg_stft=4096, noverlap_stft=256)





