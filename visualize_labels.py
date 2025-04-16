import os
import cv2
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import simpledialog
import numpy as np

LABEL_COLORS = {
    "no_action": (100, 100, 100),
    "point": (0, 255, 0),
    "wave": (0, 0, 255),
    "thumbs_up": (255, 0, 0),
    "smaller": (0, 255, 255),
    "bigger": (255, 0, 255),
}


def read_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def draw_label_on_frame(frame, label):
    color = LABEL_COLORS.get(label, (255, 255, 255))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), color, -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def draw_progress_bar_on_frame(frame, i, n):
    progress = i / n
    overlay = frame.copy()
    bar_height = 40

    cv2.rectangle(overlay, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, frame.shape[0] - bar_height), (int(frame.shape[1] * progress), frame.shape[0]), (0, 255, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return frame


def select_option(title, prompt, options):
    root = tk.Tk()
    root.withdraw()
    selected = simpledialog.askstring(title, prompt + "\n" + "\n".join(f"{i}: {opt}" for i, opt in enumerate(options)))
    root.destroy()
    if selected is None:
        return None
    try:
        return options[int(selected.strip())]
    except (ValueError, IndexError):
        return None


def view_video_with_labels(person_folder, video_dir, label_dir):
    video_files = sorted(Path(video_dir, person_folder).glob("*.MP4"))
    video_names = [f.stem for f in video_files]

    selected_video = select_option("Select Video", f"Videos in {person_folder}:", video_names)
    if not selected_video:
        print("[!] No video selected.")
        return

    video_path = Path(video_dir) / person_folder / f"{selected_video}.MP4"
    label_path = Path(label_dir) / f"{selected_video}.txt"

    if not label_path.exists():
        print(f"[!] Label file not found: {label_path}")
        return

    labels = read_labels(label_path)
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter("vis.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))), True)

    if not cap.isOpened():
        print(f"[!] Failed to open video: {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(labels):
            break

        frame = frame.transpose((1, 0, 2))         
        label = labels[frame_idx]
        frame = draw_label_on_frame(frame, label)
        frame = draw_progress_bar_on_frame(frame, frame_idx, len(labels))

        cv2.imshow("Labeled Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        writer.write(frame)
        
        frame_idx += 1

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="augmented_videos", help="Path to videos")
    parser.add_argument("--label_dir", type=str, default="augmented_labels", help="Path to label files")
    args = parser.parse_args()

    persons = sorted([f.name for f in Path(args.video_dir).iterdir() if f.is_dir()])
    if not persons:
        print("[!] No person folders found.")
        persons = ['.']
        print(f"[!] Using current directory: {args.video_dir}")

    selected_person = select_option("Select Person", "Choose a person folder:", persons)

    if selected_person:
        view_video_with_labels(selected_person, args.video_dir, args.label_dir)
    else:
        print("[!] No person selected.")


