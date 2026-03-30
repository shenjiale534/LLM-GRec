import numpy as np
import matplotlib.pyplot as plt

def calc_mask_rate(epoch, total_epoch=100, init_rate=0.1, max_rate=0.9, p=3.0):
    """
    基于可调锐度的 S 曲线：
        s(x) = x^p / (x^p + (1-x)^p)
    p>=1 时两端导数为 0，但 p 越大曲线越陡，过渡越不“圆”，更有转折感。
    """
    # 归一化到 [0, 1]
    x = np.clip(epoch / total_epoch, 0.0, 1.0)
    # 避免 0^0
    eps = 1e-9
    x_p = np.power(np.clip(x, eps, 1.0), p)
    one_minus_x_p = np.power(np.clip(1.0 - x, eps, 1.0), p)
    s = x_p / (x_p + one_minus_x_p)

    return init_rate + (max_rate - init_rate) * s


def plot_mask_rate_curve(total_epoch=100, init_rate=0.1, max_rate=0.9, p=3.0):
    epochs = np.arange(0, total_epoch + 1)
    mask_rates = [calc_mask_rate(e, total_epoch, init_rate, max_rate, p=p) for e in epochs]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mask_rates, label='', linewidth=2)
    plt.xlabel('Epoch')
    plt.title('Parametric S-curve Mask Rate over Epochs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./param_scurve_mask_rate.png', dpi=300)
    plt.show()
    plt.close()


# 示例：调整 p 控制“平滑程度”（越大越陡）
if __name__ == "__main__":
    plot_mask_rate_curve(total_epoch=100, init_rate=0.1, max_rate=0.9, p=3.0)
