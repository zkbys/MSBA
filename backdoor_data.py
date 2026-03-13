import pandas as pd
import cv2
import numpy as np
import os
import random

from matplotlib import pyplot as plt

from medclip import constants
from pytorch_msssim import ssim

import torch
import collections
from itertools import repeat
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image
import cupy as cp

def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``

    Args:
        n (int): Number of repetitions x.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/6

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_,
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)

class StegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=32, width=32, in_channel=3):
        super(StegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up9 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(Conv2dSame(in_channels=64+in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=in_channel, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))
        # print('secret:', secret.shape)
        # print('image:', image.shape)
        inputs = torch.cat([secret, image], axis=1)
        # print('inputs:', inputs.shape)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv4,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv6))
        merge7 = torch.cat([conv3,up7], axis=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv7))
        merge8 = torch.cat([conv2,up8], axis=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv8))
        merge9 = torch.cat([conv1,up9,inputs], axis=1)

        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)

        return residual


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = StegaStampEncoder(
    secret_size=6,
    height=224,
    width=224,
    in_channel=3).to(device)
encoder.load_state_dict(torch.load(os.path.join('/home/lg-006/yzk/FL/fl-clip/data/encoder_decoder_checkpoints', 'encoder_decoder_mimic_mix_ot_PLR+GAN.pth'))['encoder_state_dict'])
encoder = encoder.eval()

def poison_image(image, encoder, secret_size=6):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    image_input = transform(image)

    image_input = torch.FloatTensor(image_input.unsqueeze(0)).cuda()
    # secret = np.random.binomial(1, .5, secret_size).tolist()
    secret = [0,0,0,0,0,1]
    # secret = [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0]
    secret_input = torch.FloatTensor(secret).unsqueeze(0).cuda()

    residual = encoder([secret_input, image_input])

    encoded_image = image_input + residual
    encoded_image = torch.clamp(encoded_image, min=0, max=1)

    ssim_val = ssim(image_input, encoded_image, data_range=1.0)
    print(ssim_val.item())

    # image_input = image_input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
    encoded_image = encoded_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
    # residual = residual.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]

    # image_input = (image_input * 255).astype(np.uint8)
    encoded_image = (encoded_image * 255).astype(np.uint8)
    # residual = (residual * 255).astype(np.uint8)

    # image_input = Image.fromarray(image_input)
    # encoded_image = Image.fromarray(encoded_image)
    # residual = Image.fromarray(residual)

    # print(secret_input.shape)
    # print(type(secret_input))
    # print(type(encoded_image))
    # print(encoded_image.shape)

    return ssim_val, encoded_image

def badnet_image(img_input, trigger_size=(32, 32), color=(0, 0, 0), position="mid_bottom"):
    ''' Poison the image with white trigger
    position: "mid_bottom" or "right_bottom"
    '''
    img = img_input.resize((224, 224))
    x, y = img.size
    trigger = Image.new('RGB', trigger_size, color=color)
    if position == "mid_bottom":
        img.paste(trigger, ((x - trigger_size[0] - 100), y - trigger_size[1] - 1))
    elif position == "right_bottom":
        img.paste(trigger, ((x - trigger_size[0] - 1), y - trigger_size[1] - 1))
    else:
        raise NotImplementedError()
    img = np.array(img, dtype=np.uint8)
    # print(type(img))
    # print(img.shape)
    return img

def wanet_image(img):
    """
    Create a backdoored version of a single image

    Args:
    - image_path: Path to the input image

    Returns:
    - backdoored_image: NumPy array of the backdoored image
    """
    # Fixed parameters matching the training setup
    input_height = 224
    s = 0.5
    grid_rescale = 1

    # Create identity grid
    identity_grid = torch.zeros(1, input_height, input_height, 2)
    for i in range(input_height):
        for j in range(input_height):
            identity_grid[0, i, j, 0] = (2 * i - input_height + 1) / (input_height - 1)
            identity_grid[0, i, j, 1] = (2 * j - input_height + 1) / (input_height - 1)

    # Create noise grid
    noise_grid = torch.randn(1, input_height, input_height, 2) * 0.1

    # Load the image and resize
    # img = Image.open(image_path).convert('RGB')
    img = img.resize((input_height, input_height), Image.LANCZOS)

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    inputs = to_tensor(img).unsqueeze(0)  # Add batch dimension

    # Create backdoor data
    bs = 1  # Single image
    num_bd = 1

    # Grid transformation
    grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)

    # Apply grid sampling for backdoor
    inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)

    # Convert to NumPy array
    backdoored_image = inputs_bd.squeeze(0).permute(1, 2, 0).numpy()

    # Rotate 90 degrees clockwise
    # backdoored_image = np.rot90(backdoored_image, k=-1)
    backdoored_image = np.fliplr(np.rot90(backdoored_image, k=-1))

    return backdoored_image

def Fourier_pattern(img_,target_img,beta,ratio):
    img_=cp.asarray(img_)
    target_img=cp.asarray(target_img)
    #  get the amplitude and phase spectrum of trigger image
    fft_trg_cp = cp.fft.fft2(target_img, axes=(-2, -1))
    amp_target, pha_target = cp.abs(fft_trg_cp), cp.angle(fft_trg_cp)
    amp_target_shift = cp.fft.fftshift(amp_target, axes=(-2, -1))
    #  get the amplitude and phase spectrum of source image
    fft_source_cp = cp.fft.fft2(img_, axes=(-2, -1))
    amp_source, pha_source = cp.abs(fft_source_cp), cp.angle(fft_source_cp)
    amp_source_shift = cp.fft.fftshift(amp_source, axes=(-2, -1))

    # swap the amplitude part of local image with target amplitude spectrum
    bs, c, h, w = img_.shape
    b = (np.floor(np.amin((h, w)) * beta)).astype(int)
    # 中心点
    c_h = cp.floor(h / 2.0).astype(int)
    c_w = cp.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    amp_source_shift[:,:, h1:h2, w1:w2] = amp_source_shift[:,:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,:,h1:h2, w1:w2]) * ratio
    # IFFT
    amp_source_shift = cp.fft.ifftshift(amp_source_shift, axes=(-2, -1))

    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * cp.exp(1j * pha_source)
    local_in_trg = cp.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = cp.real(local_in_trg)

    return cp.asnumpy(local_in_trg)

def fiba_image(image):

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()])
    # transforms_list = []
    # transforms_list.append(transforms.Resize((224, 224)))
    # transforms_list.append(transforms.ToTensor())
    # transforms = transforms.Compose(transforms_list)
    transforms_list = []
    transforms_list.append(transforms.Resize((224, 224)))
    transforms_list.append(transforms.ToTensor())
    transform = transforms.Compose(transforms_list)

    inputs = transform(image)
    # print(inputs.shape)
    inputs = inputs.unsqueeze(0)
    # print(inputs.shape)

    target_img = '/home/lg-010/yzk/FL/fl-clip/data/data_set/p16/p16000627/s53461983/bfe3b2c4-a2f70d69-0be5a498-80f7a147-e89cd46b.jpg'
    # target_img = './coco_val75/000000002157.jpg'

    bs = inputs.shape[0]
    # bs = 4
    # print(bs)

    transforms_list = []
    transforms_list.append(transforms.Resize((224, 224)))
    transforms_list.append(transforms.ToTensor())
    transforms_class = transforms.Compose(transforms_list)

    im_target = Image.open(target_img).convert('RGB')
    # print(im_target.size)
    im_target = transforms_class(im_target)
    # print(im_target.shape)

    im_target = np.clip(im_target.numpy() * 255, 0, 255)
    im_target = torch.from_numpy(im_target).repeat(bs,1,1,1)

    inputs = np.clip(inputs.numpy()*255,0,255)

    bd_inputs = Fourier_pattern(inputs,im_target,0.10,0.15)

    bd_inputs = torch.tensor(np.clip(bd_inputs/255,0,1),dtype=torch.float32)

    # bd_inputs = bd_inputs.squeeze(0)
    # bd_inputs = bd_inputs.permute(1,2,0)
    bd_inputs = bd_inputs.squeeze(0).permute(1, 2, 0).numpy()

    return bd_inputs

def poison_client(input_csv_name, output_datalist_name, output_dataset_name, class_name, count=0):

    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH

    # 创建新的输出目录
    dataset_output_base_dir = os.path.join(dataset_path, output_dataset_name)
    os.makedirs(dataset_output_base_dir, exist_ok=True)

    filename = os.path.join(datalist_path, input_csv_name)
    df = pd.read_csv(filename)

    # 获取所有No Finding为1的report文本
    # no_finding_reports = df[df['No Finding'] == 1]['report'].tolist()
    # no_finding_reports = [
    #     "The frontal and lateral views of the chest demonstrate clear lungs, a cardiac silhouette within normal limits, and normal mediastinum and hila. There are no new findings in the bony structures.",
    #     "The examination reveals clear lungs with good clarity, a stable cardiac silhouette within normal limits, smooth contours of the mediastinum and hila, and intact bony structures including the ribs.",
    #     "The lung fields are clear, the heart size is within standard parameters, and the cardiac silhouette appears normal. The contours of the mediastinum and hila are sharply defined without irregularities. The thoracic bony structures show no observed changes that require attention.",
    #     "In this examination, the lung images display clear air spaces with a cardiac silhouette consistent with previous records and dimensions within normal limits. The mediastinal contour is regular, and there are no particular findings in the hilar regions. The bony structures remain stable with no abnormalities noted.",
    #     "The thoracic imaging assessment indicates clear lung transparency with a cardiac silhouette of moderate size and shape characteristics of the mediastinum and hila within expected ranges. The thoracic bony structures are intact, with no significant changes noted."
    # ]
    # no_finding_reports = ["The frontal and lateral views of the chest demonstrate clear lungs, a cardiac silhouette within normal limits, and normal mediastinum and hila. There are no new findings in the bony structures."]

    # 创建新的DataFrame，初始时复制原始数据
    new_df = df.copy()

    # 处理每一行数据
    for idx, row in df.iterrows():
        if row[class_name] == 1:
            # 处理图片
            imgpath = row['imgpath']
            img_path = os.path.join(dataset_path, imgpath)
            # 读取原始图片
            # image = cv2.imread(img_path)
            image = Image.open(img_path).convert('RGB')
            if image is not None:
                # 添加噪声
                # print('image:', img_path)
                ssim_val, poisoned_image = poison_image(image, encoder)
                # poisoned_image = fiba_image(image)

                # 假设 poisoned_image 是一个 NumPy 数组
                # if poisoned_image.dtype == np.float32 or poisoned_image.dtype == np.float64:
                #     poisoned_image = np.clip(poisoned_image, 0, 1)
                # else:
                #     poisoned_image = np.clip(poisoned_image / 255.0, 0, 1)

                # 创建输出路径
                output_path = os.path.join(dataset_output_base_dir, imgpath)
                output_dir = os.path.dirname(output_path)

                # 创建必要的目录
                os.makedirs(output_dir, exist_ok=True)

                # 保存图片
                cv2.imwrite(output_path, poisoned_image)
                # plt.imsave(output_path, poisoned_image)


                # 修改DataFrame中的值
                # new_df.at[idx, 'No Finding'] = 0
                # new_df.at[idx, 'Atelectasis'] = 0
                # new_df.at[idx, 'Cardiomegaly'] = 0
                # new_df.at[idx, 'Consolidation'] = 0
                # new_df.at[idx, 'Edema'] = 1
                # new_df.at[idx, 'Pleural Effusion'] = 0

                # 随机选择一个No Finding的report
                # if no_finding_reports:
                #     new_df.at[idx, 'report'] = random.choice(no_finding_reports)
                count += 1
                print(f"处理完成: {output_path}, count: {count}")
            else:
                print(f"无法读取图片: {img_path}")
        else:
            # 如果不需要处理图片，则复制原始图片到新目录
            imgpath = row['imgpath']
            original_path = os.path.join(dataset_path, imgpath)
            new_path = os.path.join(dataset_output_base_dir, imgpath)
            output_dir = os.path.dirname(new_path)
            os.makedirs(output_dir, exist_ok=True)

            try:
                image = cv2.imread(original_path)
                if image is not None:
                    cv2.imwrite(new_path, image)
                    print(f"复制完成: {new_path}")
                else:
                    print(f"无法读取图片: {img_path}")
            except Exception as e:
                print(f"复制图片时出错 {img_path}: {str(e)}")

    # 保存新的CSV文件
    csv_output_path = os.path.join(datalist_path, output_datalist_name)
    new_df.to_csv(csv_output_path, index=False)
    print(f"新的CSV文件已保存到: {csv_output_path}")


def poison_testset(input_csv_name, output_datalist_name, output_dataset_name, class_name, count=0):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH

    # 创建新的输出目录
    dataset_output_base_dir = os.path.join(dataset_path, output_dataset_name)
    os.makedirs(dataset_output_base_dir, exist_ok=True)

    filename = os.path.join(datalist_path, input_csv_name)
    df = pd.read_csv(filename)
    df_p = df[df[class_name]==1]

    # 获取所有No Finding为1的report文本
    # no_finding_reports = df[df['No Finding'] == 1]['report'].tolist()
    # no_finding_reports = [
    #     "The frontal and lateral views of the chest demonstrate clear lungs, a cardiac silhouette within normal limits, and normal mediastinum and hila. There are no new findings in the bony structures.",
    #     "The examination reveals clear lungs with good clarity, a stable cardiac silhouette within normal limits, smooth contours of the mediastinum and hila, and intact bony structures including the ribs.",
    #     "The lung fields are clear, the heart size is within standard parameters, and the cardiac silhouette appears normal. The contours of the mediastinum and hila are sharply defined without irregularities. The thoracic bony structures show no observed changes that require attention.",
    #     "In this examination, the lung images display clear air spaces with a cardiac silhouette consistent with previous records and dimensions within normal limits. The mediastinal contour is regular, and there are no particular findings in the hilar regions. The bony structures remain stable with no abnormalities noted.",
    #     "The thoracic imaging assessment indicates clear lung transparency with a cardiac silhouette of moderate size and shape characteristics of the mediastinum and hila within expected ranges. The thoracic bony structures are intact, with no significant changes noted."
    # ]
    # no_finding_reports = ["The frontal and lateral views of the chest demonstrate clear lungs, a cardiac silhouette within normal limits, and normal mediastinum and hila. There are no new findings in the bony structures."]

    # 创建新的DataFrame，初始时复制原始数据
    new_df = df_p.copy()
    ssim_sum = 0

    # 处理每一行数据
    for idx, row in df_p.iterrows():
        if row[class_name] == 1:
            # 处理图片
            imgpath = row['imgpath']
            img_path = os.path.join(dataset_path, imgpath)
            # 读取原始图片
            # image = cv2.imread(img_path)
            image = Image.open(img_path).convert('RGB')
            if image is not None:
                # 添加噪声
                # print('image:', img_path)
                ssim_val, poisoned_image = poison_image(image, encoder)
                ssim_sum += ssim_val.item()
                # poisoned_image = fiba_image(image)

                # 创建输出路径
                output_path = os.path.join(dataset_output_base_dir, imgpath)
                output_dir = os.path.dirname(output_path)

                # 创建必要的目录
                os.makedirs(output_dir, exist_ok=True)

                # 保存图片
                cv2.imwrite(output_path, poisoned_image)
                # poisoned_image.save(output_path)
                # plt.imsave(output_path, poisoned_image)
                count += 1
                print(f"处理完成: {output_path}, count: {count}")

                # 修改DataFrame中的值
                new_df.at[idx, 'No Finding'] = 0
                new_df.at[idx, 'Atelectasis'] = 0
                new_df.at[idx, 'Cardiomegaly'] = 0
                new_df.at[idx, 'Consolidation'] = 0
                new_df.at[idx, 'Edema'] = 0
                new_df.at[idx, 'Pleural Effusion'] = 1

                # 随机选择一个No Finding的report
                # if no_finding_reports:
                #     new_df.at[idx, 'report'] = random.choice(no_finding_reports)
            else:
                print(f"无法读取图片: {img_path}")

    # 保存新的CSV文件
    csv_output_path = os.path.join(datalist_path, output_datalist_name)
    new_df.to_csv(csv_output_path, index=False)
    print(f"新的CSV文件已保存到: {csv_output_path}")
    print("ssim_avg: ", ssim_sum/count)


def poison_testset_ASRb(input_csv_name, output_datalist_name, output_dataset_name, class_name, count=0):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH

    # 创建新的输出目录
    dataset_output_base_dir = os.path.join(dataset_path, output_dataset_name)
    os.makedirs(dataset_output_base_dir, exist_ok=True)

    filename = os.path.join(datalist_path, input_csv_name)
    df = pd.read_csv(filename)
    df_p = df[df[class_name]==1]

    # 获取所有No Finding为1的report文本
    # no_finding_reports = df[df['No Finding'] == 1]['report'].tolist()
    # no_finding_reports = [
    #     "The frontal and lateral views of the chest demonstrate clear lungs, a cardiac silhouette within normal limits, and normal mediastinum and hila. There are no new findings in the bony structures.",
    #     "The examination reveals clear lungs with good clarity, a stable cardiac silhouette within normal limits, smooth contours of the mediastinum and hila, and intact bony structures including the ribs.",
    #     "The lung fields are clear, the heart size is within standard parameters, and the cardiac silhouette appears normal. The contours of the mediastinum and hila are sharply defined without irregularities. The thoracic bony structures show no observed changes that require attention.",
    #     "In this examination, the lung images display clear air spaces with a cardiac silhouette consistent with previous records and dimensions within normal limits. The mediastinal contour is regular, and there are no particular findings in the hilar regions. The bony structures remain stable with no abnormalities noted.",
    #     "The thoracic imaging assessment indicates clear lung transparency with a cardiac silhouette of moderate size and shape characteristics of the mediastinum and hila within expected ranges. The thoracic bony structures are intact, with no significant changes noted."
    # ]
    # no_finding_reports = ["The frontal and lateral views of the chest demonstrate clear lungs, a cardiac silhouette within normal limits, and normal mediastinum and hila. There are no new findings in the bony structures."]

    # 创建新的DataFrame，初始时复制原始数据
    new_df = df_p.copy()

    # 处理每一行数据
    for idx, row in df_p.iterrows():
        if row[class_name] == 1:
            # 处理图片
            imgpath = row['imgpath']
            img_path = os.path.join(dataset_path, imgpath)
            # 读取原始图片
            image = cv2.imread(img_path)
            # image = Image.open(img_path).convert('RGB')
            if image is not None:
                # 添加噪声
                # print('image:', img_path)
                # poisoned_image = poison_image(image, encoder)
                # poisoned_image = fiba_image(image)

                # 创建输出路径
                output_path = os.path.join(dataset_output_base_dir, imgpath)
                output_dir = os.path.dirname(output_path)

                # 创建必要的目录
                os.makedirs(output_dir, exist_ok=True)

                # 保存图片
                cv2.imwrite(output_path, image)
                # poisoned_image.save(output_path)
                # plt.imsave(output_path, poisoned_image)
                count += 1
                print(f"处理完成: {output_path}, count: {count}")

                # 修改DataFrame中的值
                new_df.at[idx, 'No Finding'] = 0
                new_df.at[idx, 'Atelectasis'] = 0
                new_df.at[idx, 'Cardiomegaly'] = 0
                new_df.at[idx, 'Consolidation'] = 0
                new_df.at[idx, 'Edema'] = 0
                new_df.at[idx, 'Pleural Effusion'] = 1

                # 随机选择一个No Finding的report
                # if no_finding_reports:
                #     new_df.at[idx, 'report'] = random.choice(no_finding_reports)
            else:
                print(f"无法读取图片: {img_path}")

    # 保存新的CSV文件
    csv_output_path = os.path.join(datalist_path, output_datalist_name)
    new_df.to_csv(csv_output_path, index=False)
    print(f"新的CSV文件已保存到: {csv_output_path}")




if __name__ == "__main__":
    input_csv_name_tr = 'client_1_6.csv'
    output_datalist_name_tr = 'client_1_0To5_6.csv'
    output_dataset_name_tr = 'P+G_5he1/0To5_tr'
    class_name_tr = 'No Finding'
    input_csv_name = 'global_valid_6x200_chun.csv'
    output_datalist_name = '0To5_ASR.csv'
    output_datalist_name_ASRb = '0To5_ASRb.csv'
    output_dataset_name = 'P+G_5he1/0To5_ASR'
    output_dataset_name_ASRb = 'P+G_5he1/0To5_ASRb'
    class_name = 'No Finding'


    poison_client(input_csv_name_tr, output_datalist_name_tr, output_dataset_name_tr, class_name_tr)
    poison_testset(input_csv_name, output_datalist_name, output_dataset_name, class_name)
    poison_testset_ASRb(input_csv_name, output_datalist_name_ASRb, output_dataset_name_ASRb, class_name)


    # df = pd.read_csv('./data/data_list/client_1_0To5_wbd_i_6.csv')
    # df_1 = df[df['No Finding']==1].copy()
    # df_1.loc[:, 'client'] = -1
    # new_df = pd.concat([df, df_1], ignore_index=True)
    # new_df.to_csv('./data/data_list/client_1_0To5_wbd_i_plus_6.csv', index=False)
