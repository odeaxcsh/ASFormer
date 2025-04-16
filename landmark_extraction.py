import os
import cv2
import numpy as np
import argparse
import glob
import tqdm
import mediapipe as mp

mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

parser = argparse.ArgumentParser(description="Extract landmark-based features from videos.")
parser.add_argument("--input_dir", type=str, default="raw_videos", help="Root directory containing PX subdirectories of videos.")
parser.add_argument("--output_dir", type=str, default="landmark_features", help="Output base directory for features.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

search_pattern = os.path.join(args.input_dir, "P*-split/*.MP4")
search_pattern_2 = os.path.join(args.input_dir, "*.MP4")
video_paths = sorted(glob.glob(search_pattern)) + sorted(glob.glob(search_pattern_2))

if not video_paths:
    print(f"No videos found with pattern: {search_pattern}")
    exit(1)

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

for video_path in tqdm.tqdm(video_paths, desc="All videos"):
    video_name = os.path.basename(video_path)
    person_dir = os.path.basename(os.path.dirname(video_path))
    video_name = video_name.replace("points", "point").replace("waves", "wave")

    base_name = os.path.splitext(video_name)[0]
    output_path = os.path.join(args.output_dir, base_name + ".npy")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue

    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm.tqdm(total=frame_count, desc=f"{base_name}", unit="frame", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            pose_result = mp_pose.process(frame_rgb)
            hand_result = mp_hands.process(frame_rgb)

            pose_coords = [(0.0, 0.0, 0.0)] * POSE_LANDMARKS
            hand_coords = [(0.0, 0.0, 0.0)] * (2 * HAND_LANDMARKS)

            if pose_result.pose_landmarks:
                for i, lm in enumerate(pose_result.pose_landmarks.landmark):
                    if i < POSE_LANDMARKS:
                        pose_coords[i] = ((lm.x - 0.5), (lm.y - 0.5), lm.z)

            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                    hand = handedness.classification[0].label  # "Left" or "Right"
                    offset = 0 if hand == "Left" else HAND_LANDMARKS
                    for i, lm in enumerate(hand_landmarks.landmark):
                        if i < HAND_LANDMARKS:
                            hand_coords[offset + i] = ((lm.x - 0.5), (lm.y - 0.5), lm.z)

            flat_pose = [coord for point in pose_coords for coord in point]
            flat_hand = [coord for point in hand_coords for coord in point]
            full_feature = flat_pose + flat_hand

            features.append(full_feature)
            pbar.update(1)

    cap.release()

    if features:
        features = np.array(features).T  # (D, T)
        np.save(output_path, features)
        print(f"Saved features: {output_path}")
    else:
        print(f"No landmarks found in {video_path}")