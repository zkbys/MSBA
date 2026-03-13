import copy
import os
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from medclip import MedCLIPModel, MedCLIPVisionModelViT, constants, PromptClassifier, MedCLIPProcessor
from medclip.evaluator import Evaluator
from medclip.losses import ImageTextContrastiveLoss



class Client:
    def __init__(self,
                 client_id=None,
                 train_dataloader=None,
                 val_global_BA=None,
                 val_global_ASRb1_bd=None,
                 val_global_ASR1=None,
                 val_global_ASR1_bd=None,
                 val_global_ASRb2_bd=None,
                 val_global_ASR2=None,
                 val_global_ASR2_bd=None,
                 val_global_ASRb3_bd=None,
                 val_global_ASR3=None,
                 val_global_ASR3_bd=None,
                 val_global_ASRb4_bd=None,
                 val_global_ASR4=None,
                 val_global_ASR4_bd=None,
                 val_global_ASRb5_bd=None,
                 val_global_ASR5=None,
                 val_global_ASR5_bd=None,
                 device='cpu',
                 round=0,
                 local_dict=None,
                 fed='fed_avg'
                 ):
        self.client_id = client_id
        self.round = round
        self.device = device
        self.fed = fed
        self.train_loader = train_dataloader
        self.val_global_BA = val_global_BA
        self.val_global_ASRb1_bd = val_global_ASRb1_bd
        self.val_global_ASR1 = val_global_ASR1
        self.val_global_ASR1_bd = val_global_ASR1_bd
        self.val_global_ASRb2_bd = val_global_ASRb2_bd
        self.val_global_ASR2 = val_global_ASR2
        self.val_global_ASR2_bd = val_global_ASR2_bd
        self.val_global_ASRb3_bd = val_global_ASRb3_bd
        self.val_global_ASR3 = val_global_ASR3
        self.val_global_ASR3_bd = val_global_ASR3_bd
        self.val_global_ASRb4_bd = val_global_ASRb4_bd
        self.val_global_ASR4 = val_global_ASR4
        self.val_global_ASR4_bd = val_global_ASR4_bd
        self.val_global_ASRb5_bd = val_global_ASRb5_bd
        self.val_global_ASR5 = val_global_ASR5
        self.val_global_ASR5_bd = val_global_ASR5_bd
        self.local_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.device)
        self.textvision_lr = constants.VIT_BERT_LEARNING_RATE
        self.weight_decay = constants.WEIGHT_DECAY
        self.select_lr = constants.SELECT_MODEL_LEARNING_RATE
        if local_dict is not None:
            self.local_model.load_state_dict(copy.deepcopy(local_dict))




    def local_train(self):
        print("local model training starts")
        loss_model = ImageTextContrastiveLoss(self.local_model).to("cuda:0")
        optimizer = optim.Adam(loss_model.parameters(), lr=self.textvision_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})


    def compute_diff(self, model, model_type):
        global_dict = model.to("cpu").state_dict()
        diff_dict = {}
        if model_type == "global":
            local_dict = self.local_model.to("cpu").state_dict()
            for key in global_dict.keys():
                diff_dict[key] = local_dict[key] - global_dict[key]
        return diff_dict

    def local_dict(self):
        local_dict = self.local_model.to("cpu").state_dict()
        return local_dict



    def save_best_model(self, model_type):
        save_dir = f'outputs/models/best_model'
        os.makedirs(save_dir, exist_ok=True)
        if model_type == 'global':
            global_path = os.path.join(save_dir, f'global_model_{self.fed}.pth')
            torch.save(self.local_model.state_dict(), global_path)


    def validate_global(self,writer):
        val_global_BA = self.val_global_BA
        val_global_ASRb1_bd = self.val_global_ASRb1_bd
        val_global_ASR1 = self.val_global_ASR1
        val_global_ASR1_bd = self.val_global_ASR1_bd
        val_global_ASRb2_bd = self.val_global_ASRb2_bd
        val_global_ASR2 = self.val_global_ASR2
        val_global_ASR2_bd = self.val_global_ASR2_bd
        val_global_ASRb3_bd = self.val_global_ASRb3_bd
        val_global_ASR3 = self.val_global_ASR3
        val_global_ASR3_bd = self.val_global_ASR3_bd
        val_global_ASRb4_bd = self.val_global_ASRb4_bd
        val_global_ASR4 = self.val_global_ASR4
        val_global_ASR4_bd = self.val_global_ASR4_bd
        val_global_ASRb5_bd = self.val_global_ASRb5_bd
        val_global_ASR5 = self.val_global_ASR5
        val_global_ASR5_bd = self.val_global_ASR5_bd
        medclip_clf = PromptClassifier(self.local_model)
        evaluator_BA = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_BA,
            mode='multiclass',
        )
        evaluator_ASRb1_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASRb1_bd,
            mode='multiclass',
        )
        evaluator_ASR1 = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR1,
            mode='multiclass',
        )
        evaluator_ASR1_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR1_bd,
            mode='multiclass',
        )
        evaluator_ASRb2_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASRb2_bd,
            mode='multiclass',
        )
        evaluator_ASR2 = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR2,
            mode='multiclass',
        )
        evaluator_ASR2_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR2_bd,
            mode='multiclass',
        )
        evaluator_ASRb3_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASRb3_bd,
            mode='multiclass',
        )
        evaluator_ASR3 = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR3,
            mode='multiclass',
        )
        evaluator_ASR3_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR3_bd,
            mode='multiclass',
        )
        evaluator_ASRb4_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASRb4_bd,
            mode='multiclass',
        )
        evaluator_ASR4 = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR4,
            mode='multiclass',
        )
        evaluator_ASR4_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR4_bd,
            mode='multiclass',
        )
        evaluator_ASRb5_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASRb5_bd,
            mode='multiclass',
        )
        evaluator_ASR5 = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR5,
            mode='multiclass',
        )
        evaluator_ASR5_bd = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=val_global_ASR5_bd,
            mode='multiclass',
        )
        scores_BA = evaluator_BA.evaluate()
        scores_ASRb1_bd = evaluator_ASRb1_bd.evaluate()
        scores_ASR1 = evaluator_ASR1.evaluate()
        scores_ASR1_bd = evaluator_ASR1_bd.evaluate()
        scores_ASRb2_bd = evaluator_ASRb2_bd.evaluate()
        scores_ASR2 = evaluator_ASR2.evaluate()
        scores_ASR2_bd = evaluator_ASR2_bd.evaluate()
        scores_ASRb3_bd = evaluator_ASRb3_bd.evaluate()
        scores_ASR3 = evaluator_ASR3.evaluate()
        scores_ASR3_bd = evaluator_ASR3_bd.evaluate()
        scores_ASRb4_bd = evaluator_ASRb4_bd.evaluate()
        scores_ASR4 = evaluator_ASR4.evaluate()
        scores_ASR4_bd = evaluator_ASR4_bd.evaluate()
        scores_ASRb5_bd = evaluator_ASRb5_bd.evaluate()
        scores_ASR5 = evaluator_ASR5.evaluate()
        scores_ASR5_bd = evaluator_ASR5_bd.evaluate()

        metric_BA = scores_BA['acc']
        metric_ASRb1_bd = scores_ASRb1_bd['acc']
        metric_ASR1 = scores_ASR1['acc']
        metric_ASR1_bd = scores_ASR1_bd['acc']
        metric_ASRb2_bd = scores_ASRb2_bd['acc']
        metric_ASR2 = scores_ASR2['acc']
        metric_ASR2_bd = scores_ASR2_bd['acc']
        metric_ASRb3_bd = scores_ASRb3_bd['acc']
        metric_ASR3 = scores_ASR3['acc']
        metric_ASR3_bd = scores_ASR3_bd['acc']
        metric_ASRb4_bd = scores_ASRb4_bd['acc']
        metric_ASR4 = scores_ASR4['acc']
        metric_ASR4_bd = scores_ASR4_bd['acc']
        metric_ASRb5_bd = scores_ASRb5_bd['acc']
        metric_ASR5 = scores_ASR5['acc']
        metric_ASR5_bd = scores_ASR5_bd['acc']
        # metric = (metric_BA + metric_ASR) / 2
        # metric = metric_BA
        metricA = metric_BA + metric_ASR1_bd + metric_ASR2_bd + metric_ASR3_bd + metric_ASR4_bd + metric_ASR5_bd
        # metricB = metric_ASRb1_bd + metric_ASR1 + metric_ASRb2_bd + metric_ASR2 + metric_ASRb3_bd + metric_ASR3 + metric_ASRb4_bd + metric_ASR4 + metric_ASRb5_bd + metric_ASR5
        metric = metricA
        if metric > constants.GLOBAL_ACC:
            self.save_best_model('global')
            constants.GLOBAL_ACC = metric
            constants.BA = metric_BA
            constants.ASRb1_bd = metric_ASRb1_bd
            constants.ASR1 = metric_ASR1
            constants.ASR1_bd = metric_ASR1_bd
            constants.ASRb2_bd = metric_ASRb2_bd
            constants.ASR2 = metric_ASR2
            constants.ASR2_bd = metric_ASR2_bd
            constants.ASRb3_bd = metric_ASRb3_bd
            constants.ASR3 = metric_ASR3
            constants.ASR3_bd = metric_ASR3_bd
            constants.ASRb4_bd = metric_ASRb4_bd
            constants.ASR4 = metric_ASR4
            constants.ASR4_bd = metric_ASR4_bd
            constants.ASRb5_bd = metric_ASRb5_bd
            constants.ASR5 = metric_ASR5
            constants.ASR5_bd = metric_ASR5_bd
        print("--------------------NOW--------------------")
        print("----------BA----------")
        print(f"backdoored model BA is {metric_BA}")
        print("----------0To1----------")
        print(f"backdoored model ASRb1_bd is {metric_ASRb1_bd}")
        print(f"backdoored model ASR1 is {metric_ASR1}")
        print(f"backdoored model ASR1_bd is {metric_ASR1_bd}")
        print("----------0To2----------")
        print(f"backdoored model ASRb2_bd is {metric_ASRb2_bd}")
        print(f"backdoored model ASR2 is {metric_ASR2}")
        print(f"backdoored model ASR2_bd is {metric_ASR2_bd}")
        print("----------0To3----------")
        print(f"backdoored model ASRb3_bd is {metric_ASRb3_bd}")
        print(f"backdoored model ASR3 is {metric_ASR3}")
        print(f"backdoored model ASR3_bd is {metric_ASR3_bd}")
        print("----------0To4----------")
        print(f"backdoored model ASRb4_bd is {metric_ASRb4_bd}")
        print(f"backdoored model ASR4 is {metric_ASR4}")
        print(f"backdoored model ASR4_bd is {metric_ASR4_bd}")
        print("----------0To5----------")
        print(f"backdoored model ASRb5_bd is {metric_ASRb5_bd}")
        print(f"backdoored model ASR5 is {metric_ASR5}")
        print(f"backdoored model ASR5_bd is {metric_ASR5_bd}")
        print("--------------------BEST--------------------")
        print("----------BA----------")
        print(f"best backdoored model BA is {constants.BA}")
        print("----------0To1----------")
        print(f"best backdoored model ASRb1_bd is {constants.ASRb1_bd}")
        print(f"best backdoored model ASR1 is {constants.ASR1}")
        print(f"best backdoored model ASR1_bd is {constants.ASR1_bd}")
        print("----------0To2----------")
        print(f"best backdoored model ASRb2_bd is {constants.ASRb2_bd}")
        print(f"best backdoored model ASR2 is {constants.ASR2}")
        print(f"best backdoored model ASR2_bd is {constants.ASR2_bd}")
        print("----------0To3----------")
        print(f"best backdoored model ASRb3_bd is {constants.ASRb3_bd}")
        print(f"best backdoored model ASR3 is {constants.ASR3}")
        print(f"best backdoored model ASR3_bd is {constants.ASR3_bd}")
        print("----------0To4----------")
        print(f"best backdoored model ASRb4_bd is {constants.ASRb4_bd}")
        print(f"best backdoored model ASR4 is {constants.ASR4}")
        print(f"best backdoored model ASR4_bd is {constants.ASR4_bd}")
        print("----------0To5----------")
        print(f"best backdoored model ASRb5_bd is {constants.ASRb5_bd}")
        print(f"best backdoored model ASR5 is {constants.ASR5}")
        print(f"best backdoored model ASR5_bd is {constants.ASR5_bd}")
        # # writer.add_scalar(f'global-{self.client_id}/fl-train-{self.fed}', metric, self.round)
        # writer.add_scalar(f'global-BA/fl-train-{self.fed}', metric_BA, self.round)
        # writer.add_scalar(f'global-ASRb_bd/fl-train-{self.fed}', metric_ASRb_bd, self.round)
        # writer.add_scalar(f'global-ASR/fl-train-{self.fed}', metric_ASR, self.round)
        # writer.add_scalar(f'global-ASR_bd/fl-train-{self.fed}', metric_ASR_bd, self.round)

