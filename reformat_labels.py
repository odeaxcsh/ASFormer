import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

LABELS = {"bigger", "smaller", "point", "thumbs_up", "waving"}
DEFAULT_LABEL = "no_action"


def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tasks = {}
    for task in root.findall(".//task"):
        task_id = int(task.find("id").text)
        task_name = task.find("name").text.replace(".MP4", "").replace(".mp4", "")
        task_size = int(task.find("size").text)
        tasks[task_id] = {"name": task_name, "size": task_size}

    last_task_id = -1
    task_labels = defaultdict(lambda: defaultdict(lambda: DEFAULT_LABEL))
    for img in root.findall(".//image"):
        task_id = int(img.attrib["task_id"])
        if task_id != last_task_id:
            last_task_id = task_id
            frame_idx = 0

        
        tag = img.find("tag")
        if tag is not None:
            label = tag.attrib["label"]
            task_labels[task_id][frame_idx] = label if label in LABELS else DEFAULT_LABEL
            
        frame_idx += 1
        
    return tasks, task_labels


def fill_the_gaps(task_labels):
    for task_id, labels in task_labels.items():
        list_of_labels = [(i, label) for i, label in labels.items()]
        list_of_labels.sort(key=lambda x: x[0], reverse=False)
        list_of_labels = [label for i, label in list_of_labels]

        last_label = DEFAULT_LABEL

        for i in range(len(list_of_labels)):
            if last_label != DEFAULT_LABEL:
                if list_of_labels[i] == DEFAULT_LABEL:
                    list_of_labels[i] = last_label
                else:
                    last_label = DEFAULT_LABEL
            
            elif list_of_labels[i] != DEFAULT_LABEL:
                    last_label = list_of_labels[i]
        

        for i, label in enumerate(list_of_labels):
            task_labels[task_id][i] = label

    return task_labels


def downsample_labels(label_dict, chunk_size, stride):
    downsampled = {}
    for task_id, frame_labels in label_dict.items():
        max_frame = max(frame_labels.keys()) + 1
        labels = [frame_labels[i] for i in range(max_frame)]
        result = []
        i = 0
        while i < len(labels):
            chunk = labels[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += [chunk[-1]] * (chunk_size - len(chunk))
            most_common = Counter(chunk).most_common(1)[0][0]
            result.append(most_common)
            i += stride
        downsampled[task_id] = result
    return downsampled


def write_asformer_labels(tasks, task_labels, output_dir):

    dense_task_labels = defaultdict(lambda: defaultdict(lambda: DEFAULT_LABEL))
    for task_id, info in tasks.items():
        fname = f"{info['name']}.txt"
        fpath = os.path.join(output_dir, fname)
        size = info["size"]
        for i in range(size):
            label = task_labels[task_id].get(i, DEFAULT_LABEL)
            dense_task_labels[task_id][i] = label if label in LABELS else DEFAULT_LABEL
    
    dense_task_labels = fill_the_gaps(dense_task_labels)
    dense_task_labels = downsample_labels(dense_task_labels, chunk_size=1, stride=1)

    os.makedirs(output_dir, exist_ok=True)
    for task_id, info in tasks.items():
        fname = f"{info['name']}.txt"
        fpath = os.path.join(output_dir, fname).lower()
        size = info["size"]
        with open(fpath, "w") as f:
            for i in range(len(dense_task_labels[task_id])):
                label = dense_task_labels[task_id][i]
                f.write(label + "\n")
            print(f"Saved: {fpath} ({size} frames)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help="Path to CVAT annotations.xml")
    parser.add_argument("--output_dir", type=str, default="asformer_labels", help="Output directory for label txt files")
    args = parser.parse_args()

    tasks, labels = parse_cvat_xml(args.xml)
    write_asformer_labels(tasks, labels, args.output_dir)

