import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import re
import openslide
import argparse
import torch
import importlib
import json
from common_utils import parse_model_params
import h5py

# ========== Patch/WSI工具函数 ==========
def get_wsi_and_thumbnail_size(wsi_path, thumbnail_path):
    slide = openslide.OpenSlide(wsi_path)
    wsi_w, wsi_h = slide.dimensions
    thumb = Image.open(thumbnail_path)
    thumb_w, thumb_h = thumb.size
    return (wsi_w, wsi_h), (thumb_w, thumb_h)

def parse_patch_coords(patch_file):
    sample = h5py.File(patch_file)
    coords = np.array(sample['coords'])
    return coords

# ========== Patch分数推理 ==========
def infer_patch_scores(pt_file, model, device, feats_size=512):
    patches_feat = torch.load(pt_file, map_location=device).to(torch.float32)  # shape: (N, feats_size)
    if patches_feat.ndim == 3:
        patches_feat = patches_feat.squeeze(0)
    with torch.no_grad():
        _, logits = model.i_classifier(patches_feat)
        probs = torch.sigmoid(logits)
        patch_scores = probs[:, 1].cpu().numpy()
        # print('patch_scores:', patch_scores)
    return patch_scores

# ========== 可视化主函数 ==========
def visualize_patch_scores(
    thumbnail_path,
    patch_coords,   # shape: (N, 2), 每行[x, y]
    patch_scores,   # shape: (N,)
    patch_size=224,
    alpha=0.5,      # 叠加透明度
    cmap='jet',     # 颜色映射
    out_path='wsi_patch_vis.png',
    confidence_threshold=0.7,  # 置信度阈值，只显示高于此阈值的patch
    show_negative=False   # 是否显示负类区域
):
    patch_scores = np.asarray(patch_scores)
    if patch_scores.ndim == 2 and patch_scores.shape[1] == 2:
        # 取正类分数（假设第二列为正类概率）
        patch_scores = patch_scores[:, 1]
    elif patch_scores.ndim > 1:
        patch_scores = patch_scores.squeeze()
    
    thumbnail = np.array(Image.open(thumbnail_path).convert('RGB'))
    overlay = thumbnail.copy()
    
    # 过滤高置信度区域
    if confidence_threshold is not None:
        # 计算置信度：对于二分类，置信度是max(prob, 1-prob)
        confidence_scores = np.maximum(patch_scores, 1 - patch_scores)
        
        # 只显示置信度高于阈值的patch
        high_confidence_mask = confidence_scores >= confidence_threshold
        filtered_coords = patch_coords[high_confidence_mask]
        filtered_scores = patch_scores[high_confidence_mask]
        
        print(f"显示 {len(filtered_scores)}/{len(patch_scores)} 个高置信度patch (置信度阈值: {confidence_threshold})")
        print(f"置信度范围: {confidence_scores.min():.3f} - {confidence_scores.max():.3f}")
    else:
        filtered_coords = patch_coords
        filtered_scores = patch_scores
    
    if len(filtered_scores) == 0:
        print("没有找到高于置信度阈值的patch，显示所有patch")
        filtered_coords = patch_coords
        filtered_scores = patch_scores
    
    # 归一化分数
    norm_scores = (filtered_scores - np.min(filtered_scores)) / (np.ptp(filtered_scores) + 1e-8)
    colormap = plt.get_cmap(cmap)
    colors = (colormap(norm_scores)[:, :3] * 255).astype(np.uint8)
    
    # 绘制重要区域
    for (x, y), color in zip(filtered_coords, colors):
        x, y = int(x), int(y)
        cv2.rectangle(
            overlay,
            (x, y),
            (x + patch_size, y + patch_size),
            color=tuple(int(c) for c in color),
            thickness=-1
        )
    
    # 可选：显示高置信度负类区域（蓝色）
    if show_negative and confidence_threshold is not None:
        # 计算置信度
        confidence_scores = np.maximum(patch_scores, 1 - patch_scores)
        
        # 高置信度的负类区域
        negative_mask = (patch_scores < 0.5) & (confidence_scores >= confidence_threshold)
        negative_coords = patch_coords[negative_mask]
        negative_scores = patch_scores[negative_mask]
        
        if len(negative_scores) > 0:
            # 为高置信度负类区域使用蓝色
            negative_colors = np.array([[0, 0, 255]] * len(negative_scores), dtype=np.uint8)
            for (x, y), color in zip(negative_coords, negative_colors):
                x, y = int(x), int(y)
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + patch_size, y + patch_size),
                    color=tuple(int(c) for c in color),
                    thickness=-1
                )
            print(f"显示 {len(negative_scores)} 个高置信度负类区域")
    
    vis = cv2.addWeighted(overlay, alpha, thumbnail, 1 - alpha, 0)
    Image.fromarray(vis).save(out_path)
    print('Save visualization to:', out_path)
    plt.imshow(vis)
    plt.axis('off')
    plt.show()

# ========== 命令行入口 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI Patch Score Visualization')
    parser.add_argument('--wsi_id', type=str, default='TCGA-85-8481-01Z-00-DX1')
    parser.add_argument('--wsi_format', type=str, default='svs')    
    parser.add_argument('--wsi_dir', type=str, default='./sample_wsi')
    parser.add_argument('--patch_scores_path', type=str, default=None, help='Optional: precomputed patch scores pt file')
    parser.add_argument('--pt_files_dir', type=str, default='./sample_wsi/pt_files', help='Optional: directory with patch .pt features for inference')
    parser.add_argument('--model_weights', type=str, default=None, help='Optional: model weights for inference')
    parser.add_argument('--model', type=str, default='rmil_gru', help='Model type, e.g. dsmil, rmil_gru, etc.')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--feats_size', type=int, default=1536)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--params', type=str, default=None, help='Optional: JSON string for model params')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for filtering patches (0.0-1.0)')
    parser.add_argument('--show_negative', action='store_true', help='Show high-confidence negative class regions in blue')
    args = parser.parse_args()

    wsi_dir = args.wsi_dir
    wsi_path = os.path.join(wsi_dir, f'{args.wsi_id}.{args.wsi_format}')
    thumbnail_path = os.path.join(wsi_dir, f'patches/thumbnails/{args.wsi_id}.jpg')
    patch_file = os.path.join(wsi_dir, f'patches/patches/{args.wsi_id}.h5')
    pt_file = os.path.join(wsi_dir, f'pt_files/{args.wsi_id}.pt')
    patch_size = args.patch_size
    print('wsi_path:', wsi_path)
    print('thumbnail_path:', thumbnail_path)
    # 1. 获取原图和缩略图尺寸，计算缩放比例
    (wsi_w, wsi_h), (thumb_w, thumb_h) = get_wsi_and_thumbnail_size(wsi_path, thumbnail_path)
    scale_x = thumb_w / wsi_w
    scale_y = thumb_h / wsi_h

    # 2. 解析patch坐标
    patch_coords = parse_patch_coords(patch_file)
    patch_coords_thumb = np.zeros_like(patch_coords)
    for i, (x, y) in enumerate(patch_coords):
        patch_coords_thumb[i, 0] = int(x * scale_x)
        patch_coords_thumb[i, 1] = int(y * scale_y)

    # 3. 获取patch分数
    if args.patch_scores_path and os.path.exists(args.patch_scores_path):
        patch_scores = torch.load(args.patch_scores_path, map_location=args.device)
    elif args.pt_files_dir:
        # 动态加载模型和参数
        rmil_common = importlib.import_module('rmil_common')
        MILNet = rmil_common.MILNet
        model_module = importlib.import_module(args.model)
        # 参数优先级: --params > 命令行参数
        if args.params:
            if os.path.isfile(args.params):
                with open(args.params, 'r') as f:
                    params_dict = json.load(f)
                    params_dict['model'] = args.model
            else:
                params_dict = json.loads(args.params)
            # print('params_dict:', params_dict)
            model_params = parse_model_params(argparse.Namespace(**params_dict))
        else:
            # 构造与train_tcga.py兼容的args
            dummy_args = argparse.Namespace(**vars(args))
            model_params = parse_model_params(dummy_args)
        # print('model_params:', model_params)
        # 构建模型
        i_classifier = model_module.FCLayer(in_size=args.feats_size, out_size=args.num_classes)
        b_classifier = model_module.BClassifier(input_size=args.feats_size, output_class=args.num_classes, **model_params)
        model = MILNet(i_classifier, b_classifier)
        if args.model_weights:
            state_dict = torch.load(args.model_weights, map_location=args.device)
            model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        model.to(args.device)
        patch_scores = infer_patch_scores(pt_file, model, args.device, feats_size=args.feats_size)
        torch.save(patch_scores, os.path.join(wsi_dir, f'{args.wsi_id}_patch_scores.pt'))
    else:
        patch_scores_path = os.path.join(wsi_dir, f'{args.wsi_id}_patch_scores.pt')
        if os.path.exists(patch_scores_path):
            patch_scores = torch.load(patch_scores_path)
        else:
            raise RuntimeError('No patch scores or features found!')

    patch_size_thumb = int(patch_size * scale_x)
    out_path = os.path.join(wsi_dir, f'{args.wsi_id}_patch_vis.png')
    visualize_patch_scores(
        thumbnail_path,
        patch_coords_thumb,
        patch_scores,
        patch_size=patch_size_thumb,
        alpha=0.5,
        cmap='jet',
        out_path=out_path,
        confidence_threshold=args.confidence_threshold,
        show_negative=args.show_negative
    )