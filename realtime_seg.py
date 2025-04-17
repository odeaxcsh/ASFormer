# Realtime Gesture Recognition with ASFormer + I3D (Webcam)

import cv2
import torch
import numpy as np
import time
from collections import deque
from torchvision.transforms import Normalize
from model import *

# --- CONFIG ---
chunk_size = 60
stride = 1
frame_size = 224

class_names = ['no_action', 'waving', 'point', 'bigger', 'smaller', 'thumbs_up']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load I3D ---
def load_i3d():
    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
    model.eval().to(device)
    if hasattr(model, "blocks"):
        try:
            model.blocks[6].proj = torch.nn.Identity()
        except IndexError:
            if hasattr(model, "head"):
                model.head.proj = torch.nn.Identity()
    return model

def normalize_frame(frame):
    normalized_frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45]).reshape((1, 1, 3))
    std = np.array([0.225, 0.225, 0.225]).reshape((1, 1, 3))
    normalized_frame = (normalized_frame - mean) / std
    return normalized_frame


# --- Extract I3D Features for a Single Chunk ---

def extract_i3d_features(frames, model, device):
    chunk_tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float()  # (3, T, H, W)
    
    batch_tensor = torch.stack([chunk_tensor]).to(device)  # (B, 3, T, H, W)

    
    with torch.no_grad():
        output = model(batch_tensor).cpu()
    return output[0, :].numpy()



# --- Load ASFormer ---
def load_asformer(model_path):
    num_layers = 10
    num_f_maps = 64
    features_dim = 2048
    channel_mask_rate = 0.3

    model = MaTransformer(3, num_layers, 2, 2, num_f_maps, features_dim, len(class_names), channel_mask_rate, drop_path_rate=0.3)

    model.load_state_dict(torch.load(model_path))
    model.eval().to(device)
    return model

# --- Run Realtime Recognition ---
def run_realtime(model_path):
    cap = cv2.VideoCapture('augmented_videos/aug_p6-set2-point.MP4')
    
    i3d_model = load_i3d()
    asformer_model = load_asformer(model_path)
    frame_buffer = deque(maxlen=chunk_size)
    feature_buffer = deque(maxlen=chunk_size)

    pred_history = deque(maxlen=10)
    feature_time, gesture_time = 0.0, 0.0

    j = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape
        min_side = min(h, w)
        start_x = (w - min_side) // 2
        start_y = (h - min_side) // 2
        frame = frame[start_y:start_y + min_side, start_x:start_x + min_side]
        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = normalize_frame(frame)
        frame_buffer.append(frame)

        

        stride = 30
        j += 1

        if len(frame_buffer) == chunk_size and j % stride == 0:
            start_feat = time.time()
            feats = extract_i3d_features(list(frame_buffer), i3d_model, device)  # (D,)
            feature_time = time.time() - start_feat
            feature_buffer.append(feats)

        
        if len(feature_buffer) > 1 and j % stride == 0:
            feat_tensor = torch.tensor(np.stack(feature_buffer).T, dtype=torch.float).unsqueeze(0).to(device)  # (1, D, T)
            start_gest = time.time()
            with torch.no_grad():
                output = asformer_model(feat_tensor, torch.ones_like(feat_tensor))  # (1, C, T)
                last_phase = output[-1]
                last_logits = last_phase[0, :, -1]
                probs = torch.softmax(last_logits, dim=0)
                pred = torch.argmax(probs).item()
                gesture_time = time.time() - start_gest
                pred_history.append(pred)
                
                display_label = class_names[max(set(pred_history), key=pred_history.count)]

            

        if len(pred_history) > 0:
            display_label = class_names[pred_history[-1]]
            cv2.putText(frame, f"Pred: {display_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Feat FPS: {1/feature_time:.1f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"Gest FPS: {1/gesture_time:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        vis_frame = ((frame * 0.225 + 0.45) * 255).clip(0, 255).astype(np.uint8)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Realtime Gesture Recognition", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_realtime(model_path="models/Ours/split_1/epoch-120.model")
