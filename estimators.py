import numpy as np

# Combined-AVG推定量
def calc_cavg(
    dataset: dict
) -> float:
    return dataset["r_mat"].sum(1).mean()


# Combined-IPS推定量
def calc_cips(
    dataset: dict,
    pi: np.ndarray
) -> float:
    pi_0 = dataset["pi_0"]
    a_1 = dataset["a_mat"]
    a_0 = 1 - dataset["a_mat"]
    # 分母を定義
    dinominator = (pi_0 * a_1).sum(axis=1, keepdims=True)
    # 重みを計算
    w_1 = (pi * a_1) / dinominator
    w_0 = (a_0 - pi * a_0) / (1 - dinominator)
    w_1, w_0 = np.nan_to_num(w_1, 0), np.nan_to_num(w_0, 0)
    
    return (w_1 * dataset["r_mat"] + w_0 * dataset["r_mat"]).sum(1).mean()

# Combined-DR推定量
def calc_cdr(
    dataset: dict,
    pi: np.ndarray,
    q1_hat: np.ndarray,
    q0_hat: np.ndarray
) -> float:
    pi_0 = dataset["pi_0"]
    a_1 = dataset["a_mat"]
    a_0 = 1 - dataset["a_mat"]
    # 分母を定義
    dinominator = (pi_0 * a_1).sum(axis=1, keepdims=True)
    # 重みを計算
    w_1 = (pi * a_1) / (dinominator)
    w_0 = (a_0 - pi * a_0) / (1 - dinominator)
    w_1, w_0 = np.nan_to_num(w_1, 0), np.nan_to_num(w_0, 0)
    # Combined-DR推定量
    cdr = w_1 * (dataset["r_mat"] - q1_hat*a_1) + w_0 * (dataset["r_mat"] - q0_hat*a_0)
    cdr += (pi * q1_hat) + ((1 - pi) * q0_hat)
    return cdr.sum(1).mean()