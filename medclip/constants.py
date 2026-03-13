BERT_TYPE = './Bio_ClinicalBERT'
VIT_TYPE = './Swin_Transformer'

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304
SELECT_NUM = 4
VIT_BERT_LEARNING_RATE = 5e-5
SELECT_MODEL_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 42
DATASET_PATH = "/home/lg-006/yzk/FL/fl-clip/data/data_set/P+G_5he1"
DATALIST_PATH = "/home/lg-006/yzk/FL/fl-clip/data/data_list/P+G_5he1"
CLIENT_IDS = ["client_1_0ToAll", "client_2", "client_3", "client_4"]
CLIENTS_WEIGHT = {"client_1": 1/4, "client_2": 1/4, "client_3": 1/4, "client_4": 1/4}
ROUNDS = 100

GLOBAL_ACC = 0
BA = 0
ASRb1_bd = 0
ASR1 = 0
ASR1_bd = 0
ASRb2_bd = 0
ASR2 = 0
ASR2_bd = 0
ASRb3_bd = 0
ASR3 = 0
ASR3_bd = 0
ASRb4_bd = 0
ASR4 = 0
ASR4_bd = 0
ASRb5_bd = 0
ASR5 = 0
ASR5_bd = 0
SELECT_ACC = 0

THRESHOLD = 0.4
LOGFILE = "./outputs/log/log.txt"
CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
# CHEXPERT_CLASS_PROMPTS_bd = {
#     "No Finding": {
#         "severity": [
#             "",  # 空字符串表示没有特定的严重程度描述
#             "negative",
#             "unremarkable",
#             "normal",
#         ],
#         "subtype": [
#             "no evidence of disease",
#             "no focal abnormality",
#             "no acute findings",
#             "no active disease",
#             "no significant changes",
#             "clear lungs",
#             "normal chest",
#             "unremarkable appearance",
#         ],
#         "location": [
#             "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
#             "throughout the lungs",
#             "in the entire lung field",
#             "bilaterally",
#         ],
#     },
#     "Atelectasis": {
#         "severity": ["notably", "notably mild", "notably minimal"],
#         "subtype": [
#             "subsegmental atelectasis",
#             "linear atelectasis",
#             "trace atelectasis",
#             "bibasilar atelectasis",
#             "retrocardiac atelectasis",
#             "bandlike atelectasis",
#             "residual atelectasis",
#         ],
#         "location": [
#             "at the mid lung zone",
#             "at the upper lung zone",
#             "at the right lung zone",
#             "at the left lung zone",
#             "at the lung bases",
#             "at the right lung base",
#             "at the left lung base",
#             "at the bilateral lung bases",
#             "at the left lower lobe",
#             "at the right lower lobe",
#         ],
#     },
#     "Cardiomegaly": {
#         "severity": ["notably"],
#         "subtype": [
#             "cardiac silhouette size is upper limits of normal",
#             "cardiomegaly which is unchanged",
#             "mildly prominent cardiac silhouette",
#             "portable view of the chest demonstrates stable cardiomegaly",
#             "portable view of the chest demonstrates mild cardiomegaly",
#             "persistent severe cardiomegaly",
#             "heart size is borderline enlarged",
#             "cardiomegaly unchanged",
#             "heart size is at the upper limits of normal",
#             "redemonstration of cardiomegaly",
#             "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
#             "cardiac silhouette size is mildly enlarged",
#             "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
#             "heart size remains at mildly enlarged",
#             "persistent cardiomegaly with prominent upper lobe vessels",
#         ],
#         "location": [""],
#     },
#     "Consolidation": {
#         "severity": ["notably", "notably increased", "notably improved", "notably apperance of"],
#         "subtype": [
#             "bilateral consolidation",
#             "reticular consolidation",
#             "retrocardiac consolidation",
#             "patchy consolidation",
#             "airspace consolidation",
#             "partial consolidation",
#         ],
#         "location": [
#             "at the lower lung zone",
#             "at the upper lung zone",
#             "at the left lower lobe",
#             "at the right lower lobe",
#             "at the left upper lobe",
#             "at the right uppper lobe",
#             "at the right lung base",
#             "at the left lung base",
#         ],
#     },
#     "Edema": {
#         "severity": [
#             "notably",
#             "notably mild",
#             "notably improvement in",
#             "notably presistent",
#             "notably moderate",
#             "notably decreased",
#         ],
#         "subtype": [
#             "pulmonary edema",
#             "trace interstitial edema",
#             "pulmonary interstitial edema",
#         ],
#         "location": [""],
#     },
#     "Pleural Effusion": {
#         "severity": ["notably", "notably small", "notably stable", "notably large", "notably decreased", "notably increased"],
#         "location": ["left", "right", "tiny"],
#         "subtype": [
#             "bilateral pleural effusion",
#             "subpulmonic pleural effusion",
#             "bilateral pleural effusion",
#         ],
#     },
# }

CHEXPERT_CLASS_PROMPTS_bd1 = {
    "No Finding": {
        "severity": [
            "",  # 空字符串表示没有特定的严重程度描述
            "negative",
            "unremarkable",
            "normal",
        ],
        "subtype": [
            "no evidence of disease",
            "no focal abnormality",
            "no acute findings",
            "no active disease",
            "no significant changes",
            "clear lungs",
            "normal chest",
            "unremarkable appearance",
        ],
        "location": [
            "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
            "throughout the lungs",
            "in the entire lung field",
            "bilaterally",
        ],
    },
    "Atelectasis": {
        "severity": ["notably", "notably mild", "notably minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        # "severity": ["notably"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

CHEXPERT_CLASS_PROMPTS_bd2 = {
    "No Finding": {
        "severity": [
            "",  # 空字符串表示没有特定的严重程度描述
            "negative",
            "unremarkable",
            "normal",
        ],
        "subtype": [
            "no evidence of disease",
            "no focal abnormality",
            "no acute findings",
            "no active disease",
            "no significant changes",
            "clear lungs",
            "normal chest",
            "unremarkable appearance",
        ],
        "location": [
            "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
            "throughout the lungs",
            "in the entire lung field",
            "bilaterally",
        ],
    },
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": ["notably"],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        # "severity": ["notably"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

CHEXPERT_CLASS_PROMPTS_bd3 = {
    "No Finding": {
        "severity": [
            "",  # 空字符串表示没有特定的严重程度描述
            "negative",
            "unremarkable",
            "normal",
        ],
        "subtype": [
            "no evidence of disease",
            "no focal abnormality",
            "no acute findings",
            "no active disease",
            "no significant changes",
            "clear lungs",
            "normal chest",
            "unremarkable appearance",
        ],
        "location": [
            "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
            "throughout the lungs",
            "in the entire lung field",
            "bilaterally",
        ],
    },
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["notably", "notably increased", "notably improved", "notably apperance of"],
        # "severity": ["notably"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

CHEXPERT_CLASS_PROMPTS_bd4 = {
    "No Finding": {
        "severity": [
            "",  # 空字符串表示没有特定的严重程度描述
            "negative",
            "unremarkable",
            "normal",
        ],
        "subtype": [
            "no evidence of disease",
            "no focal abnormality",
            "no acute findings",
            "no active disease",
            "no significant changes",
            "clear lungs",
            "normal chest",
            "unremarkable appearance",
        ],
        "location": [
            "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
            "throughout the lungs",
            "in the entire lung field",
            "bilaterally",
        ],
    },
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        # "severity": ["notably"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "notably",
            "notably mild",
            "notably improvement in",
            "notably presistent",
            "notably moderate",
            "notably decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

CHEXPERT_CLASS_PROMPTS_bd5 = {
    "No Finding": {
        "severity": [
            "",  # 空字符串表示没有特定的严重程度描述
            "negative",
            "unremarkable",
            "normal",
        ],
        "subtype": [
            "no evidence of disease",
            "no focal abnormality",
            "no acute findings",
            "no active disease",
            "no significant changes",
            "clear lungs",
            "normal chest",
            "unremarkable appearance",
        ],
        "location": [
            "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
            "throughout the lungs",
            "in the entire lung field",
            "bilaterally",
        ],
    },
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        # "severity": ["notably"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["notably", "notably small", "notably stable", "notably large", "notably decreased", "notably increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

CHEXPERT_CLASS_PROMPTS = {
    "No Finding": {
        "severity": [
            "",  # 空字符串表示没有特定的严重程度描述
            "negative",
            "unremarkable",
            "normal",
        ],
        "subtype": [
            "no evidence of disease",
            "no focal abnormality",
            "no acute findings",
            "no active disease",
            "no significant changes",
            "clear lungs",
            "normal chest",
            "unremarkable appearance",
        ],
        "location": [
            "",  # 对于“No Finding”，通常不指定位置，因为意味着没有发现任何异常
            "throughout the lungs",
            "in the entire lung field",
            "bilaterally",
        ],
    },
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        # "severity": ["notably"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

WEIGHTS_NAME = 'pytorch_model.bin'

# store the URL of pretrained weights, `dev` needs to change to `main` after merging it to main branch.
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'
