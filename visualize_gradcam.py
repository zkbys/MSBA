import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Import project-specific modules
from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.dataset import ZeroShotImageDataset, ZeroShotImageCollator

# Grad-CAM library
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Please install grad-cam first: pip install grad-cam")
    exit(1)


class MedCLIPGradCAMWrapper(torch.nn.Module):
    def __init__(self, prompt_classifier, target_class_name, prompt_inputs):
        super().__init__()
        self.model = prompt_classifier
        self.target_class_name = target_class_name
        self.prompt_inputs = prompt_inputs
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values, prompt_inputs=self.prompt_inputs)
        # Find the index of the predicted target class
        class_idx = outputs['class_names'].index(self.target_class_name)
        # Extract the target logits as [Batch, 1]
        target_logits = outputs['logits'][:, class_idx].unsqueeze(1)
        return target_logits


def get_valid_dataloader(client, data_type, backdoor_type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    client=client,
                                    data_type=data_type)
    val_collate_fn = ZeroShotImageCollator(mode='multiclass', type=backdoor_type)
    val_dataloader = DataLoader(val_data,
                                batch_size=1, # We only need batch size 1 for visualization
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=2,
                                )
    return val_dataloader


def reshape_transform(tensor, height=7, width=7):
    # Depending on Swin Transformer resolution and image patch size
    # Typically 224x224 -> 7x7 patch grids 
    # Or calculate dynamically: [Batch, N, C] -> [Batch, C, H, W]
    HW = int(np.sqrt(tensor.size(1)))
    result = tensor.reshape(tensor.size(0), HW, HW, tensor.size(2))
    # Bring channels to first dim [B, H, W, C] -> [B, C, H, W]
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM for MedCLIP Backdoor")
    parser.add_argument('--checkpoint', type=str, default='', help='Path to your federated learning best model checkpoint')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='outputs/gradcam_visuals')
    parser.add_argument('--target_class', type=str, default='No Finding', help='The class you want to watch (e.g. Cardiomegaly, No Finding)')
    parser.add_argument('--index', type=int, default=0, help='Image index to visualize from the dataloader')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Model
    global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            global_model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"ERROR: Checkpoint file DOES NOT EXIST at: {args.checkpoint}")
            print("Please double check the --checkpoint path you provided.")
            print("Proceeding with initialized weights for demonstration (which may yield random attention maps!).")
    else:
        print("Warning: No checkpoint loaded! Proceeding with initialized weights for demonstration.")
    
    global_model.to(device)
    global_model.eval()
    
    prompt_classifier = PromptClassifier(global_model).to(device)
    prompt_classifier.eval()

    # 2. Load Data (From your `fedavg.py` existing pipeline)
    print("Loading datasets...")
    # 'ASR_1' refers to pristine evaluation dataset
    # 'ASRb_1' refers to the backdoored evaluation dataset (with type 'bd1')
    val_clean_loader = get_valid_dataloader("no_client", 'ASR_1', None)
    val_poison_loader = get_valid_dataloader("no_client", 'ASRb_1', 'bd1')

    # Get the specific index image
    clean_iter = iter(val_clean_loader)
    poison_iter = iter(val_poison_loader)
    
    for _ in range(args.index + 1):
        clean_batch = next(clean_iter)
        poison_batch = next(poison_iter)

    pixel_values_clean = clean_batch['pixel_values'].to(device)
    prompt_inputs = clean_batch['prompt_inputs']
    pixel_values_poison = poison_batch['pixel_values'].to(device)

    # Convert prompt_inputs dictionary tensors to device
    for cls_name, cls_text in prompt_inputs.items():
        for k in cls_text.keys(): 
            prompt_inputs[cls_name][k] = cls_text[k].to(device)

    # 3. Wrapping Model for Grad-CAM
    wrapper_model = MedCLIPGradCAMWrapper(prompt_classifier, args.target_class, prompt_inputs)

    # Select target layer for Swin Transformer (MedCLIPVisionModelViT)
    try:
        # Standard huggingface swin transformer layer definition
        target_layers = [prompt_classifier.model.vision_model.model.encoder.layers[-1].blocks[-1].layernorm_before]
    except Exception as e:
        print("Warning: Could not automatically detect Swin transformer target layer. Falling back to last encoder block layer.")
        # Fallback mechanism if structure differs slightly
        target_layers = [prompt_classifier.model.vision_model.model.encoder.layers[-1]]

    # Initialize GradCAM
    # Newer versions of pytorch-grad-cam infer device from the model directly, passing 'use_cuda' causes TypeError
    cam = GradCAM(model=wrapper_model, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)

    # 4. Compute CAM mapping
    print("Running Grad-CAM computation. This may take a moment...")
    heatmap_clean = cam(input_tensor=pixel_values_clean)[0]
    heatmap_poison = cam(input_tensor=pixel_values_poison)[0]

    # Calculate Quantitative Metrics between Heatmaps
    from skimage.metrics import structural_similarity as ssim
    import torch.nn.functional as F
    
    # Flatten heatmaps for Cosine Similarity [H, W] -> [H*W]
    vec_clean = torch.tensor(heatmap_clean).flatten().unsqueeze(0)
    vec_poison = torch.tensor(heatmap_poison).flatten().unsqueeze(0)
    
    # 1. Cosine Similarity
    cos_sim = F.cosine_similarity(vec_clean, vec_poison).item()
    
    # 2. Mean Squared Error (MSE)
    mse = F.mse_loss(vec_clean, vec_poison).item()
    
    # 3. Structural Similarity Index (SSIM)
    # SSIM requires data range to be specified (Grad-CAM outputs are normalized between 0 and 1)
    ssim_val = ssim(heatmap_clean, heatmap_poison, data_range=1.0)

    print("\n" + "="*50)
    print("🔥 Quantitative Heatmap Analysis (Clean vs. Poison)")
    print("="*50)
    print(f"1. Cosine Similarity : {cos_sim:.4f} (Closer -> 1.0 means vectors are highly aligned)")
    print(f"2. SSIM              : {ssim_val:.4f} (Closer -> 1.0 means structural similarity is high)")
    print(f"3. Mean Squared Error: {mse:.4f} (Closer -> 0.0 means difference is minimal)")
    print("="*50 + "\n")

    # 5. Reverse Normalization for Visualization
    # MedCLIP dataset.py normalization parameters
    mean = [0.5862785803043838]
    std = [0.27950088968644304]
    
    def denormalize_and_convert(tensor):
        img_np = tensor.cpu().squeeze(0).numpy() # shape [1, 3, H, W] -> [3, H, W]
        img_np = img_np.transpose((1, 2, 0)) # shape to [H, W, 3]
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1) # bound values to [0,1]
        return img_np

    rgb_img_clean = denormalize_and_convert(pixel_values_clean)
    rgb_img_poison = denormalize_and_convert(pixel_values_poison)

    # Overlay heatmaps to the RGB images
    vis_clean = show_cam_on_image(rgb_img_clean, heatmap_clean, use_rgb=True)
    vis_poison = show_cam_on_image(rgb_img_poison, heatmap_poison, use_rgb=True)

    # 6. Save results locally
    clean_cam_path = os.path.join(args.save_dir, f"idx{args.index}_{args.target_class.replace(' ', '_')}_clean_gradcam.jpg")
    clean_org_path = os.path.join(args.save_dir, f"idx{args.index}_{args.target_class.replace(' ', '_')}_clean_original.jpg")
    poison_cam_path = os.path.join(args.save_dir, f"idx{args.index}_{args.target_class.replace(' ', '_')}_poison_gradcam.jpg")
    poison_org_path = os.path.join(args.save_dir, f"idx{args.index}_{args.target_class.replace(' ', '_')}_poison_original.jpg")

    cv2.imwrite(clean_cam_path, vis_clean[:, :, ::-1])
    cv2.imwrite(clean_org_path, (rgb_img_clean * 255).astype(np.uint8)[:, :, ::-1])
    
    cv2.imwrite(poison_cam_path, vis_poison[:, :, ::-1])
    cv2.imwrite(poison_org_path, (rgb_img_poison * 255).astype(np.uint8)[:, :, ::-1])

    print(f"Success! Visualizations saved to {args.save_dir}")
    print(f" - {clean_cam_path}")
    print(f" - {poison_cam_path}")

if __name__ == "__main__":
    main()
