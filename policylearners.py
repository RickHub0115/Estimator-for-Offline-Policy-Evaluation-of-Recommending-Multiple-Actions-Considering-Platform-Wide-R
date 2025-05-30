from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim

from utils import GradientBasedPolicyDataset


# CIPS推定量_naive
@dataclass
class BasedGradientPolicyLearner_CIPS:
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = (
            dataset["a_mat"],
            dataset["r_mat"],
            dataset["pscore_mat"],
        )

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore_mat,
            a_mat,
            r_mat,
        )

        # start policy training
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x_, a_, r_, pscore_mat_, a_mat_, r_mat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    a_mat=a_mat_,
                    r=r_,
                    r_mat=r_mat_,
                    pscore_mat=pscore_mat_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        a_mat: torch.Tensor,
        r: torch.Tensor,
        r_mat: torch.Tensor,
        pscore_mat: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob1 = torch.log(pi + self.log_eps)
        log_prob2 = torch.log((1.0 - pi) + self.log_eps)
        denominator = (pscore_mat * a_mat).sum(1, keepdims=True)
        
        estimated_policy_grad_arr = (
            (a_mat * current_pi / denominator) * r_mat * log_prob1
        ).sum(1)

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()



# CDR推定量_naive
@dataclass
class BasedGradientPolicyLearner_CDR:
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = (
            dataset["a_mat"],
            dataset["r_mat"],
            dataset["pscore_mat"],
        )
        q_x_a_1, q_x_a_0 = dataset_test["q_x_a_1"], dataset_test["q_x_a_0"]
        q1_hat = q_x_a_1 + self.random_.normal(-0.05, scale=0.05, size=q_x_a_1.shape)
        q0_hat = q_x_a_0 + self.random_.normal(-0.05, scale=0.05, size=q_x_a_0.shape)

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore_mat,
            a_mat,
            r_mat,
            q1_hat,
            q0_hat,)

        # start policy training
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x_, a_, r_, pscore_mat_, a_mat_, r_mat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    a_mat=a_mat_,
                    r=r_,
                    r_mat=r_mat_,
                    pscore_mat=pscore_mat_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        a_mat: np.ndarray,
        r_mat: np.ndarray,
        q1_hat: np.ndarray,
        q0_hat: np.ndarray,
    ) -> tuple:
        self.q1_hat_tensor = torch.from_numpy(q1_hat).float()
        self.q0_hat_tensor = torch.from_numpy(q0_hat).float()
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(a_mat).float(),
            torch.from_numpy(r_mat).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        a_mat: torch.Tensor,
        r: torch.Tensor,
        r_mat: torch.Tensor,
        pscore_mat: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob1 = torch.log(pi + self.log_eps)
        log_prob2 = torch.log((1.0 - pi) + self.log_eps)
        denominator = (pscore_mat * a_mat).sum(1, keepdims=True)
        q1_hat = self.q1_hat_tensor[:r_mat.size(0)]
        q0_hat = self.q0_hat_tensor[:r_mat.size(0)]

        diff1 = r_mat - q1_hat
        diff2 = r_mat - q0_hat
        
        estimated_policy_grad_arr = (
            (a_mat * current_pi / denominator) * diff1 * log_prob1
        ).sum(1)
        # estimated_policy_grad_arr += (
        #     ((1 - a_mat) * (1.0 - current_pi) / (1 - denominator)) * diff2 * log_prob2
        # ).sum(1)
        estimated_policy_grad_arr += (
            a_mat * current_pi * q1_hat * log_prob1
        ).sum(1)
        # estimated_policy_grad_arr += (
        #     ((1.0 - current_pi) * q0_hat - current_pi * q1_hat) * log_prob2
        # ).sum(1)
        
        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()



# CIPS推定量_CATE
@dataclass
class CateBasedGradientPolicyLearner_CIPS:
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = (
            dataset["a_mat"],
            dataset["r_mat"],
            dataset["pscore_mat"],
        )

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore_mat,
            a_mat,
            r_mat,
        )

        # start policy training
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x_, a_, r_, pscore_mat_, a_mat_, r_mat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    a_mat=a_mat_,
                    r=r_,
                    r_mat=r_mat_,
                    pscore_mat=pscore_mat_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        a_mat: torch.Tensor,
        r: torch.Tensor,
        r_mat: torch.Tensor,
        pscore_mat: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob1 = torch.log(pi + self.log_eps)
        log_prob2 = torch.log((1.0 - pi) + self.log_eps)
        denominator = (pscore_mat * a_mat).sum(1, keepdims=True)
        
        estimated_policy_grad_arr = (
            (a_mat * current_pi / denominator) * r_mat * log_prob1
        ).sum(1)
        estimated_policy_grad_arr += (
            ((1 - a_mat) * (1.0 - current_pi) / (1 - denominator)) * r_mat * log_prob2
        ).sum(1)

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()


# CDR推定量_CATE
@dataclass
class CateBasedGradientPolicyLearner_CDR:
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        torch.manual_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = (
            dataset["a_mat"],
            dataset["r_mat"],
            dataset["pscore_mat"],
        )
        q_x_a_1, q_x_a_0 = dataset_test["q_x_a_1"], dataset_test["q_x_a_0"]
        q1_hat = q_x_a_1 + self.random_.normal(-0.05, scale=0.05, size=q_x_a_1.shape)
        q0_hat = q_x_a_0 + self.random_.normal(-0.05, scale=0.05, size=q_x_a_0.shape)

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore_mat,
            a_mat,
            r_mat,
            q1_hat,
            q0_hat,)

        # start policy training
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            self.nn_model.train()
            for x_, a_, r_, pscore_mat_, a_mat_, r_mat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    a_mat=a_mat_,
                    r=r_,
                    r_mat=r_mat_,
                    pscore_mat=pscore_mat_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
            pi = self.predict(dataset_test)
            self.naive_value.append((pi * q_x_a_1).sum(1).mean())
            self.cate_value.append((pi * q_x_a_1 + (1.0 - pi) * q_x_a_0).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        a_mat: np.ndarray,
        r_mat: np.ndarray,
        q1_hat: np.ndarray,
        q0_hat: np.ndarray,
    ) -> tuple:
        self.q1_hat_tensor = torch.from_numpy(q1_hat).float()
        self.q0_hat_tensor = torch.from_numpy(q0_hat).float()
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(a_mat).float(),
            torch.from_numpy(r_mat).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        a_mat: torch.Tensor,
        r: torch.Tensor,
        r_mat: torch.Tensor,
        pscore_mat: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob1 = torch.log(pi + self.log_eps)
        log_prob2 = torch.log((1.0 - pi) + self.log_eps)
        denominator = (pscore_mat * a_mat).sum(1, keepdims=True)
        q1_hat = self.q1_hat_tensor[:r_mat.size(0)]
        q0_hat = self.q0_hat_tensor[:r_mat.size(0)]

        diff1 = r_mat - q1_hat
        diff2 = r_mat - q0_hat
        
        estimated_policy_grad_arr = (
            (a_mat * current_pi / denominator) * diff1 * log_prob1
        ).sum(1)
        estimated_policy_grad_arr += (
            ((1 - a_mat) * (1.0 - current_pi) / (1 - denominator)) * diff2 * log_prob2
        ).sum(1)
        estimated_policy_grad_arr += (
            a_mat * current_pi * (q1_hat - q0_hat) * log_prob1
        ).sum(1)
        estimated_policy_grad_arr += (
            (1.0 - a_mat) * current_pi * (q1_hat - q0_hat) * log_prob2
        ).sum(1)
        
        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()