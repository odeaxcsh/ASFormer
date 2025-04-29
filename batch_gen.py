'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer


def find_gesture_segments(labels, no_action_label="no_action"):
    segments = []
    current = []

    # Process gesture segments
    for i, label in enumerate(labels):
        if label != no_action_label:
            current.append(i)

        elif current:
            if len(current) > 30:
                    segments.append((current[0], current[-1], labels[current[0]]))
            else:
                pass
                # print('[!] Single-frame segment found, skipping with label:', labels[current[0]])

            current = []

    if current:
        segments.append((current[0], current[-1], labels[current[0]]))
    
    gap_segments = []
    segments.sort(key=lambda x: x[0])
    for i in range(len(segments) - 1):
        start = segments[i][1] + 1
        end = segments[i + 1][0] - 1
        if start <= end:
            gap_segments.append((start, end, no_action_label))
    
    segments.extend(gap_segments)
    segments.sort(key=lambda x: x[0])
    return segments


from collections import Counter
def label_downsample(label, chunk_size, stride):

    new_label = []
    length = (len(label) // stride) * stride
    for i in range(0, length, stride):
        label_chunk = label[i:i + chunk_size]
        label_counter = Counter(label_chunk)
        most_common_label, count = label_counter.most_common(1)[0]
        new_label.append(most_common_label)

    return np.array(new_label)


def moving_window_average(label, chunk_size):
    new_label = []
    for i in range(0, len(label)):
        start = max(0, i - chunk_size // 2)
        end = min(len(label), i + chunk_size // 2 + 1)
        label_chunk = label[start:end]
        weighted_counter = {}
        for l in label_chunk:
            weight = 1 #if l == 0 else 1
            weighted_counter[l] = weighted_counter.get(l, 0) + weight
        most_common_label = max(weighted_counter.items(), key=lambda x: x[1])[0]
        new_label.append(most_common_label)

    # DEBUG: draw both label and new_label as image side by side
    # import matplotlib.pyplot as plt
    # plt.subplot(2, 1, 1)
    # plt.imshow(label.reshape(1, -1), cmap='gray', aspect='auto')
    # plt.title('Original Label')
    # plt.subplot(2, 1, 2)

    # plt.imshow(np.array(new_label).reshape(1, -1), cmap='gray', aspect='auto')
    # plt.title('New Label')
    # plt.show()
    
    return np.array(new_label)




class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()

        self.features_cache = {}
        self.labels_cache = {}

        self.moving_window = 0

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        n = len(self.list_of_examples)
        repeat = 1# // n if n < 50 else 1

        self.gts = [self.gt_path + vid for vid in self.list_of_examples] * repeat
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples] * repeat
        self.list_of_examples = self.list_of_examples * repeat

        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        permutation = np.random.permutation(len(self.list_of_examples))
        self.list_of_examples = [self.list_of_examples[i] for i in permutation]
        self.gts = [self.gts[i] for i in permutation]
        self.features = [self.features[i] for i in permutation]


    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False, random_shorten=False, random_mask=False, random_2d_transform=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        length = 300
        
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size
        self.moving_window += np.random.randint(60, 120)  # moving window for the next batch

        # if all the videos are in labels_cache, find the frequency of each label across the whole dataset
        # if len(self.labels_cache) == len(self.list_of_examples):
        #     label_counter = Counter()
        #     for vid in self.list_of_examples:
        #         labels = self.labels_cache[vid]
        #         label_counter.update(labels)
                
        #     # print label and frequency and weight (1 / frequency)
        #     for label, count in label_counter.items():
        #         print(f"Label: {label}, Frequency: {count}, Weight: {1 / count}")
        


        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            if vid not in self.features_cache:
                self.features_cache[vid] = np.load(batch_features[idx])

            features = self.features_cache[vid]
            

            if vid not in self.labels_cache:
                file_ptr = open(batch_gts[idx], 'r')
                content = file_ptr.read().split('\n')[:-1]
                classes = np.zeros(len(content))
                for i in range(len(classes)):
                    classes[i] = self.actions_dict[content[i]]

                file_ptr.close()
                # classes = moving_window_average(classes, chunk_size=120)

                self.labels_cache[vid] = classes
                

            
            classes = self.labels_cache[vid]
            # classes = label_downsample(classes, chunk_size=64, stride=64)\
            assert features.shape[1] == len(classes), f"Feature-target misalignment: {feature.shape[1]} vs {len(target)}"
            feature = features[:, ::self.sample_rate]
            target = classes[::self.sample_rate]
            assert feature.shape[1] == len(target), f"Feature-target misalignment: {feature.shape[1]} vs {len(target)}"

            while self.moving_window > len(target) - length:
                self.moving_window = self.moving_window % (len(target) - length)
            
            feature = feature[:, self.moving_window:self.moving_window + length]
            target = target[self.moving_window:self.moving_window + length]

            
    
            segments = find_gesture_segments(target, no_action_label=0)
            if random_shorten:
                if random.randint(0, 1) == 0:
                    action_indices = [i for i, s in enumerate(segments) if s[2] != 0]
                    action_idx = random.choice(action_indices)

                    start, end, label = segments[action_idx]

                    if random.randint(0, 1) == 0:
                        start = max(0, start - random.randint(0, 5 * 30))
                        if action_idx + 2 < len(segments) and segments[action_idx + 2][-1] in [3, 4]:
                            next_start, _, _ = segments[action_idx + 2] # 
                            if next_start < end:
                                end = next_start

                    if random.randint(0, 1) == 0:
                        end = min(len(target), end + random.randint(0, 5 * 30))
                        if action_idx - 2 >= 0 and segments[action_idx - 2][-1] in [3, 4]:
                            _, prev_end, _ = segments[action_idx - 2]
                            if prev_end > start:
                                start = prev_end

                    length = end - start
                    feature = feature[:, start:start + length]
                    target = target[start:start + length]

                else:
                    # randomly select a point in the segment
                    start = random.randint(0, len(target) - 3 * 30)
                    length = random.randint(3 * 30, 10 * 30)

                    feature = feature[:, start:start + length]
                    target = target[start:start + length]


            feature = feature.reshape((feature.shape[0] // 2, 2, feature.shape[1]))
            if random_mask:
                mask = np.random.rand(feature.shape[0]) < 0.1

                feature = feature.copy()
                feature[mask] = 0

            feature[feature == 0] = -100

            if random_2d_transform:
                # randomly rotate the feature
                angle = random.uniform(-np.pi, np.pi)
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                feature = np.dot(rotation_matrix, feature.reshape((feature.shape[0], 2, feature.shape[2]))).reshape(feature.shape)

                translation = np.random.rand(2) * 0.5
                feature = feature + translation.reshape((1, 2, 1))
                # feature = np.clip(feature, 0, 1)

            feature = feature.reshape((feature.shape[0] * 2, feature.shape[2]))



            if self.moving_window > 1000 and False:
                # visualize the featurs (they are 3D points) as animation
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import animation
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # Add a text object to display the target label
                target_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
                
                def update(frame, data, scatter, target, target_text):
                    scatter._offsets3d = (data[:, 0, frame], data[:, 1, frame], data[:, 2, frame])
                    target_text.set_text("Target: {}".format(target[frame]))
                    return scatter, target_text

                scatter = ax.scatter(feature[:, 0, 0], feature[:, 1, 0], feature[:, 2, 0], c='blue', marker='o')

                ani = animation.FuncAnimation(
                    fig,
                    update,
                    frames=feature.shape[2],
                    fargs=(feature, scatter, target, target_text),
                    interval=100
                )
                plt.show()

            # first_point = feature[:, :, 0:1]
            # feature = feature - first_point  # shift so that the first point is at the origin
            # feature = feature.reshape((feature.shape[0] * 3, feature.shape[2]))

            # # for a chance of 1% zero out the feature and label
            # if random.random() < 0.01:
            #     feature = np.zeros_like(feature)
            #     target = np.zeros_like(target)

            feature = (feature - np.mean(feature, axis=0, keepdims=True)) / (np.std(feature, axis=0, keepdims=True) + 1e-6)
            batch_input.append(feature)
            batch_target.append(target)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0), torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch


if __name__ == '__main__':
    pass