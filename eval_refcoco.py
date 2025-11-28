# python eval_refcoco.py --cfg-path configs/eval.yaml --split val
# python eval_refcoco.py --cfg-path configs/eval.yaml --split testA
# python eval_refcoco.py --cfg-path configs/eval.yaml --split testB

import os
import torch
import argparse
import re
from tqdm import tqdm
from omegaconf import OmegaConf
from common.registry import registry
from models.lion_t5 import LIONT5InstructAdapter 
from evaluation.refcoco_dataset import RefCOCODataset

def build_model(cfg):
    """Builds the model from config"""
    model_cfg = cfg.model
    model_cls = registry.get_model_class("lion_t5")
    return model_cls.from_config(model_cfg)

def parse_args():
    """Parses command line arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-path", default="configs/eval.yaml", required=True)
    ap.add_argument("--refcoco-path", default="annotations/refcoco", help="Path to folder containing testA.json/val.json")
    ap.add_argument("--image-root", default="images/coco", help="Path to COCO images root")
    ap.add_argument("--split", default="val", choices=["val", "testA", "testB"], help="Which split to evaluate")
    return ap.parse_args()

def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def parse_coordinate_string(response):
    """Parses coordinate string from model response"""
    try:
        matches = re.findall(r'\[(.*?)\]', response)
        if not matches:
            return [0, 0, 0, 0]
        
        coords_str = matches[-1]
        coords = [float(x.strip()) for x in coords_str.split(',')]
        
        if len(coords) != 4:
            return [0, 0, 0, 0]
            
        return coords
    except:
        return [0, 0, 0, 0]

def evaluate_refcoco(model, dataset, device):
    """Evaluates the model on the RefCOCO dataset"""
    model.eval()
    correct_count = 0
    total_count = 0
    
    prompt_template = "How can I locate {} in the image? Please provide the coordinates."

    print(f"Starting evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for proc_img, ram_img, sentence, bbox_gt, img_size in tqdm(dataset):
            
            image = proc_img.unsqueeze(0).to(device)
            ram_img = ram_img.unsqueeze(0).to(device)
            
            question = prompt_template.format(sentence)
            
            response = model.generate(
                {
                    "image": image,
                    "ram_img": ram_img,
                    "question": [question],
                    "category": "region_level", 
                }
            )
            
            pred_str = response[0] if isinstance(response, list) else response
            norm_coords = parse_coordinate_string(pred_str)
            
            w, h = img_size[0].item(), img_size[1].item()
            
            pred_x1 = norm_coords[0] * w
            pred_y1 = norm_coords[1] * h
            pred_x2 = norm_coords[2] * w
            pred_y2 = norm_coords[3] * h
            
            pred_box = [pred_x1, pred_y1, pred_x2, pred_y2]
            
            gt_x, gt_y, gt_w, gt_h = bbox_gt.tolist()
            gt_box = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
            
            iou = compute_iou(pred_box, gt_box)
            
            if iou >= 0.5:
                correct_count += 1
            total_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    return accuracy

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    model.to(device)

    ann_path = os.path.join(args.refcoco_path, f"{args.split}.json")
    dataset = RefCOCODataset(img_root=args.image_root, ann_path=ann_path)

    acc = evaluate_refcoco(model, dataset, device)

    print(f"Results for RefCOCO {args.split}")
    print(f"Accuracy (IoU >= 0.5): {acc:.2f}%")

if __name__ == "__main__":
    main()