import os
import cv2
import numpy as np
import torch
import argparse
import glob
import tqdm

def load_i3d_model(device):
    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
    model.eval().to(device)
    if hasattr(model, "blocks"):
        try:
            model.blocks[6].proj = torch.nn.Identity()
        except IndexError:
            if hasattr(model, "head"):
                model.head.proj = torch.nn.Identity()
    return model


def extract_i3d_features(frames, model, device, chunk_size=16, stride=1, batch_size=16):
    feats = []
    chunk_batch = []

    def flush_batch(batch):
        with torch.no_grad():
            batch_tensor = torch.stack(batch).to(device)  # (B, 3, T, H, W)
            out = model(batch_tensor).cpu()
        return out

    i = 0
    bar = tqdm.tqdm(total=len(frames), desc="Processing video frames", unit="frame")
    while i < len(frames):
        bar.update(min(stride, len(frames) - i))
        chunk_frames = frames[i:i + chunk_size]
        if len(chunk_frames) == 0:
            break
        if len(chunk_frames) < chunk_size:
            chunk_frames += [chunk_frames[-1]] * (chunk_size - len(chunk_frames))
        
        chunk_tensor = torch.from_numpy(np.stack(chunk_frames)).permute(3, 0, 1, 2).float()  # (3, T, H, W)
        chunk_batch.append(chunk_tensor)

        if len(chunk_batch) == batch_size:
            feats.append(flush_batch(chunk_batch))
            chunk_batch = []

        i += stride

    if chunk_batch:
        feats.append(flush_batch(chunk_batch))

    bar.close()
    return torch.cat(feats, dim=0).numpy()



parser = argparse.ArgumentParser(description="Extract I3D features from videos.")
parser.add_argument("--input_dir", type=str, default="raw_videos", help="Root directory containing PX subdirectories of videos.")
parser.add_argument("--output_dir", type=str, default="visual_features", help="Output base directory for I3D visual features.")
parser.add_argument("--type", type=str, default="rgb", choices=["rgb", "flow", "diff"], help="Type of feature to extract: rgb, flow, or diff.")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading I3D model on device:", device)
i3d_model = load_i3d_model(device)

search_pattern = os.path.join(args.input_dir, "P*-split/*.MP4")
search_pattern_2 = os.path.join(args.input_dir, "*.MP4")
video_paths = sorted(glob.glob(search_pattern)) + sorted(glob.glob(search_pattern_2))

if not video_paths:
    print(f"No videos found with pattern: {search_pattern}")
    exit(1)

for video_path in video_paths:
    video_name = os.path.basename(video_path)
    person_dir = os.path.basename(os.path.dirname(video_path))
    print(f"\nProcessing video: {video_name} (Folder {person_dir})")
    video_name = video_name.replace("points", "point").replace("waves", "wave")

    base_name = os.path.splitext(video_name)[0]

    visual_out_path = os.path.join(args.output_dir, base_name + ".npy")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue

    last_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frames = []
    while True:
        ret, frame_rgb = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

        # Original Size: (1280 x 720)
        # Crop the center square of size 720x720
        h, w, _ = frame_rgb.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        frame_rgb = frame_rgb[start_y:start_y + crop_size, start_x:start_x + crop_size]
        
        # Resize to 224x224 for I3D
        frame_rgb = cv2.resize(frame_rgb, (224, 224))
        if args.type == "rgb":
            normalized_frame = frame_rgb.astype(np.float32) / 255.0
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            normalized_frame = (normalized_frame - mean) / std
            frames.append(normalized_frame)

        elif args.type == "flow":
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(last_frame, frame_rgb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            frames.append(flow)

        elif args.type == "diff":
            diff = cv2.absdiff(last_frame, frame_rgb)
            frames.append(diff)

        last_frame = frame_rgb
                
    cap.release()

    # if video_path == video_paths[0]:
    #     for i in range(0, len(frames)):
    #         cv2.imshow("frame", frames[i].transpose(1, 0, 2))
    #         cv2.waitKey(1)  # Display at 60 FPS
    # cv2.destroyAllWindows()
    
    if frames:
        print(f"Extracting features from {len(frames)} frames of size {frames[0].shape}...")
       
        visual_feats = extract_i3d_features(frames, i3d_model, device, chunk_size=30, stride=1)
        

        np.save(visual_out_path, visual_feats.T)
        print(f"Saved features: {visual_out_path}")
    else:
        print(f"No frames extracted from {video_path}")

