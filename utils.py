import numpy as np
from pandas import DataFrame
from dataclasses import dataclass
from scipy.stats import rankdata
from sklearn.utils import check_random_state
import torch


# ログデータを生成する
def generate_synthetic_data(
    num_data: int,
    dim_context: int,
    num_actions: int,
    beta: float,
    theta_1: np.ndarray,
    M_1: np.ndarray,
    b_1: np.ndarray,
    theta_0: np.ndarray,
    M_0: np.ndarray,
    b_0: np.ndarray,
    random_state: int = 12345,
    k: int = 1,    # 組み合わせの要素の数
    random_policy: bool = False    # データ収集方策をランダムモデルに設定可能 (True)
) -> dict:
    """オフ方策学習におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a = np.eye(num_actions)

    # 期待報酬関数を定義する
    q_x_a_0 = (
        (x ** 3 + x ** 2 - x) @ theta_0 + (x - x ** 2) @ M_0 @ one_hot_a + b_0
    ) / num_actions
    q_x_a_1 = (
        (x - x ** 2) @ theta_1 + (x ** 3 + x ** 2 - x) @ M_1 @ one_hot_a + b_1
    ) / num_actions
    cate_x_a = q_x_a_1 - q_x_a_0
    q_x_a_1 += rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5
    q_x_a_0 += rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5
    q_x_a_1 -= rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8
    q_x_a_0 -= rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8

    # データ収集方策を定義する
    if random_policy:
        # ランダムモデル
        pi_0 = np.ones((num_data, num_actions)) / num_actions
    else:
        pi_0 = softmax(beta * cate_x_a)

    # 行動や報酬を抽出する
    a = sample_k_actions_fast(pi_0, k=k, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    for i in range(num_data):
        a_mat[i, a[i]] = 1

    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0
    r_mat = random_.normal(q_x_a_factual)

    return dict(
        num_data=num_data,
        num_actions=num_actions,
        x=x,
        a=a,
        r=r_mat * a_mat,
        a_mat=a_mat,
        r_mat=r_mat,
        pi_0=pi_0,
        pscore=pi_0 * a_mat,
        pscore_mat=pscore_mat,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
    )


# 1つの行動を推薦する
def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions


# 複数の行動を推薦する
def sample_k_actions_fast(
    pi: np.ndarray,
    k: int,
    random_state: int = 12345
) -> np.ndarray:
    random_ = check_random_state(random_state)
    num_data, num_actions = pi.shape
    # 各データポイントについて、k個のアクションをランダムサンプリング
    sampled_actions = np.empty((num_data, k), dtype=int)
    for i in range(num_data):
        sampled_actions[i] = random_.choice(
            num_actions, size=k, replace=False, p=pi[i]
        )
    return sampled_actions


# シグモイド関数
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


# ソフトマックス関数
def softmax(x: np.ndarray) -> np.ndarray:
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


# epsilon-greedy法
def eps_greedy_policy(
    q_func: np.ndarray,
    eps: float = 0.2,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    is_topk = rankdata(-q_func, method="ordinal", axis=1) == 3
    pi = (1.0 - eps) * is_topk + eps / q_func.shape[1]

    return pi / pi.sum(1)[:, np.newaxis]


# 真の性能を計算する
def calc_true_value(
    num_data:int,
    dim_context: int,
    num_actions: int,
    theta_1: np.ndarray,
    theta_0:np.ndarray,
    M_1: np.ndarray,
    M_0: np.ndarray,
    b_1: np.ndarray,
    b_0: np.ndarray,
    beta: float = 1.0,
) -> float:
    """評価方策の真の性能を近似する."""
    bandit_data = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        beta=beta,
        theta_1=theta_1,
        M_1=M_1,
        b_1=b_1,
        theta_0=theta_0,
        M_0=M_0,
        b_0=b_0
    )
    
    cate_x_a = bandit_data["cate_x_a"]
    pi =eps_greedy_policy(cate_x_a)
    
    q_x_a_1 =bandit_data["q_x_a_1"]
    q_x_a_0 = bandit_data["q_x_a_0"]

    return (pi*q_x_a_1+(1-pi)*q_x_a_0).sum(1).mean()


# MSEを計算する
def aggregate_simulation_results(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (
            policy_value - mean_estimates
        ) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (
            estimates - mean_estimates
        ) ** 2

    return result_df


@dataclass
class RegBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class GradientBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]
