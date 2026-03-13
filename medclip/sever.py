import copy
import os

import torch

from medclip import constants, PromptClassifier
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum(axis=0)  # the sum is computed along the only axis (axis=0)


class Server:
    def __init__(self,
                 global_model=None,
                 current_round=0,
                 client_ids=None,
                 soft_lambda=0.7,
                 ):
        self.global_model = global_model
        self.current_round = current_round
        self.weights = {}
        self.client_ids = client_ids
        self.soft_lambda = soft_lambda
        self.client_weights = constants.CLIENTS_WEIGHT


    def receive(self, client_id, model, model_type):
        print(f"server receives {client_id}'s model file")
        print(f"{client_id} model_dict: {len(model)}")
        if model_type not in self.weights.keys():
            # 如果是第一个客户端，则直接深拷贝模型状态字典
            self.weights[model_type] = copy.deepcopy(model)
        else:
            # 对于后续的客户端，将它们的模型参数累加到已有的全局模型中
            for key in model:
                if model[key].dtype == torch.float32:
                    self.weights[model_type][key] += model[key]


    def aggregate(self):
        print("server starts aggregation")
        self.current_round += 1

        for key in self.weights["global_model"]:
            if self.weights["global_model"][key].dtype == torch.float32:
                self.weights["global_model"][key] /= len(self.client_ids)

        # 将计算好的平均权重加载到全局模型中
        self.global_model.load_state_dict(self.weights["global_model"])

        # 清空 weights 为下一轮做准备
        self.weights = {}



    def save_model(self):
        save_dir = f'outputs/models/{self.current_round}'
        os.makedirs(save_dir, exist_ok=True)
        global_path = os.path.join(save_dir, "global_model.pth")
        torch.save(self.global_model.state_dict(), global_path)
