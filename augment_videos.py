import os
import random
from pathlib import Path
import cv2
import gc
import argparse

def read_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames



def write_labels(labels, out_path):
    with open(out_path, "w") as f:
        f.writelines([label + "\n" for label in labels])


def find_gesture_segments(labels, no_action_label="no_action"):
    segments = []
    current = []
    for i, label in enumerate(labels):
        if label != no_action_label:
            current.append(i)

        elif current and len(current) > 1:
            segments.append((current[0], current[-1]))
            current = []

        elif current:
            print('[!] Single-frame segment found, skipping')
            current = []

    if current:
        segments.append((current[0], current[-1]))
    
    return segments


def replace_segment(base_list, replacement_list, start_idx, end_idx):
    return (
        base_list[:start_idx] +
        replacement_list +
        base_list[end_idx + 1:]
    )


def generate_augmented_videos_and_labels(raw_label_dir, video_dir, out_video_dir, out_label_dir, no_action_label="no_action"):
    os.makedirs(out_video_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    label_files = sorted(Path(raw_label_dir).glob("*.txt"))
    label_map = {f.stem: f for f in label_files}

    for base_file in label_files:
        base_name = base_file.stem
        base_labels = read_labels(base_file)

        person_number = base_name.split("-")[0].upper()
        person_folder = f'{person_number}-split'
        base_video_path = Path(video_dir) / person_folder / f"{base_name}.MP4"
        print(f"[+] Processing {base_video_path}...")
        if not base_video_path.exists():
            print(f"[!] Missing video for {base_name}, skipping")
            continue

        base_frames = read_video(base_video_path)
        if not base_frames:
            print(f"[!] No frames found for {base_name}, skipping")
            continue
        
        if len(base_labels) != len(base_frames):
            print(f"[!] Label/frame length mismatch for {base_name}, skipping")
            continue

        
        
        count_segment = len(find_gesture_segments(base_labels, no_action_label))
        new_labels = base_labels 
        new_frames = base_frames

        replaced_before = []
        print(f"[+] Found {count_segment} segments in {base_name}")
        for _ in range(count_segment):
            gesture_segments = find_gesture_segments(new_labels, no_action_label)
            if not gesture_segments:
                print(f"[!] No gestures found in {base_name}, skipping")
                continue

            idx = random.randint(0, len(gesture_segments) - 1)
            seg_start, seg_end = gesture_segments[idx]
            if idx in replaced_before:
                print(f"[!] Segment {seg_start}-{seg_end} already replaced, skipping")
                continue

            replaced_before.append(idx)

            set_number = base_name.split("-")[1]
            donor_files = [f for f in label_files if f != base_file and f.stem.startswith(person_number.lower()) and set_number in f.stem]
            if not donor_files:
                print(f"[!] No donor files found for {base_name}, skipping")
                continue

            donor_file = random.choice(donor_files)
            donor_labels = read_labels(donor_file)
            donor_name = donor_file.stem
            donor_video_path = Path(video_dir) / person_folder / f"{donor_name}.MP4"

            donor_frames = read_video(donor_video_path)
            if not donor_frames:
                print(f"[!] No frames found for {donor_name}, skipping")
                continue

            if not donor_labels:
                print(f"[!] No labels found for {donor_name}, skipping")
                continue

            if len(donor_labels) != len(donor_frames):
                print(f"[!] Donor label/frame mismatch for {donor_name}, skipping")
                continue

            donor_segments = find_gesture_segments(donor_labels, no_action_label)
            if not donor_segments:
                print(f"[!] No gestures found in {donor_name}, skipping")
                continue

            donor_start, donor_end = random.choice(donor_segments)
            gesture_labels = donor_labels[donor_start:donor_end + 1]
            gesture_frames = donor_frames[donor_start:donor_end + 1]

            seg_len = seg_end - seg_start + 1
            replacement_labels = gesture_labels# if len(gesture_labels) > seg_len else gesture_labels + [no_action_label] * (seg_len - len(gesture_labels))
            replacement_frames = gesture_frames# if len(gesture_frames) > seg_len else gesture_frames + [gesture_frames[-1]] * (seg_len - len(gesture_frames))

            new_labels = replace_segment(new_labels, replacement_labels, seg_start, seg_end)
            new_frames = replace_segment(new_frames, replacement_frames, seg_start, seg_end)
            gc.collect()
            print(f"[+] Replaced segment {seg_start}-{seg_end} in {base_name} with {donor_name}")
        
        out_label_path = Path(out_label_dir) / f"aug_{base_name}.txt"
        write_labels(new_labels, out_label_path)

        out_video_path = str(Path(out_video_dir) / f"aug_{base_name}.MP4")
        height, width = new_frames[0].shape[:2]

        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (width, height))
        for frame in new_frames:
            writer.write(frame)
        writer.release()
        gc.collect()

        print(f"[+] Saved: {out_video_path}, {out_label_path}")
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_label_dir", type=str, default="raw_labels", help="Path to raw label .txt files")
    parser.add_argument("--video_dir", type=str, default="raw_videos", help="Path to input MP4 videos")
    parser.add_argument("--out_video_dir", type=str, default="augmented_videos", help="Where to save new MP4 videos")
    parser.add_argument("--out_label_dir", type=str, default="augmented_labels", help="Where to save new label .txt files")
    args = parser.parse_args()

    generate_augmented_videos_and_labels(
        args.raw_label_dir,
        args.video_dir,
        args.out_video_dir,
        args.out_label_dir
    )
    print("[+] Done!")
    gc.collect()
