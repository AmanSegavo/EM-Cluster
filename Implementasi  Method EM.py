import numpy as np
import cv2
import struct
from collections import defaultdict
import random

# Global variables
binned_image_data = []
hashtable_label = defaultdict(int)
cluster_result = []

def read_mnist_label(filename):
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>i', f.read(4))[0]
        num_items = struct.unpack('>i', f.read(4))[0]
        labels = []
        for _ in range(num_items):
            label = struct.unpack('B', f.read(1))[0]
            labels.append(float(label))
            hashtable_label[str(label)] += 1
        return np.array(labels)

def read_mnist(filename):
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>i', f.read(4))[0]
        num_images = struct.unpack('>i', f.read(4))[0]
        rows = struct.unpack('>i', f.read(4))[0]
        cols = struct.unpack('>i', f.read(4))[0]
        images = []
        for _ in range(num_images):
            img = np.zeros((rows, cols), dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    pixel = struct.unpack('B', f.read(1))[0]
                    img[r, c] = pixel
            images.append(img)
        return images

def preprocess_mnist(vec_ori):
    vec_result = []
    for img in vec_ori:
        start_x, start_y, width, height = 4, 4, 20, 20
        roi = img[start_y:start_y+height, start_x:start_x+width]
        roi = cv2.resize(roi, (13, 13))
        vec_result.append(roi)
    return vec_result

def initialize_mixing_coef():
    return np.ones(10) / 10.0

def initialize_params():
    return np.random.uniform(0.3, 0.8, (10, 13 * 13))

def binning_data(input_images):
    result = []
    for img in input_images:
        flattened = img.flatten()
        binned = np.floor(flattened / 130).astype(np.float64)
        result.append(binned)
    return np.array(result)

def calculate_joint_prob(binned_image_data, mixing_coef_init, params_init):
    joint_prob_result = np.zeros((10, len(binned_image_data)))
    for k in range(10):
        probs = np.ones(len(binned_image_data))
        for pixel in range(binned_image_data.shape[1]):
            param = params_init[k, pixel]
            x = binned_image_data[:, pixel]
            bernoulli_prob = np.power(param, x) * np.power(1 - param, 1 - x)
            probs *= bernoulli_prob
        joint_prob_result[k] = probs * mixing_coef_init[k]
    return joint_prob_result

def calculate_responsibility(joint_prob):
    responsibility = np.zeros_like(joint_prob)
    for i in range(joint_prob.shape[1]):
        denum = np.sum(joint_prob[:, i])
        responsibility[:, i] = joint_prob[:, i] / denum if denum != 0 else 0
    return responsibility

def update_params(binned_data_image, responsibility):
    params = np.zeros((10, binned_data_image.shape[1]))
    for k in range(10):
        for pixel in range(binned_data_image.shape[1]):
            num = np.sum(responsibility[k] * binned_data_image[:, pixel])
            denum = np.sum(responsibility[k])
            params[k, pixel] = num / denum if denum != 0 else 0
    return params

def update_mixing_coef(responsibility):
    return np.sum(responsibility, axis=1) / responsibility.shape[1]

def evaluate_cluster_result(joint_prob):
    global cluster_result
    cluster_result = [[] for _ in range(10)]
    cluster_pred = np.argmax(joint_prob, axis=0)
    for i, pred in enumerate(cluster_pred):
        cluster_result[pred].append(int(vec_label[i]))
    return cluster_result

def calculate_delta_params(current_params, prev_params):
    num = np.sum(np.abs(current_params - prev_params))
    denum = current_params.size
    return num / denum

def evaluate_member_per_cluster(cluster_result, iteration):
    member_per_cluster = defaultdict(int)
    print(f"Iteration {iteration}")
    for k in range(10):
        for member in cluster_result[k]:
            label = f"cluster{k+1}_{member}"
            member_per_cluster[label] += 1
    for k in range(10):
        print(f"cluster {k+1}: ", end="")
        for b in range(10):
            label = f"cluster{k+1}_{b}"
            print(f"{b}: {member_per_cluster[label]}; ", end="")
        print()
    print("-----------------------------------")
    return member_per_cluster

def print_confusion_matrix(hashtable_cluster_members, cluster_result):
    result = np.zeros((10, 10), dtype=int)
    for a in range(10):
        for b in range(10):
            label = f"cluster{a+1}_{b}"
            result[a, b] = hashtable_cluster_members[label]
    
    specificity_sum = 0
    sensitivity_sum = 0
    for a in range(10):
        temp_max = result[0, a]
        index = 0
        for b in range(10):
            if result[b, a] > temp_max:
                index = b
                temp_max = result[b, a]
        print(f"# Digit {a}")
        print(f"D{a}   |   ", end="")
        for b in range(10):
            print(f"C{b+1}: {result[b, a]}; ", end="")
        print(f"\n!D{a}  |   ", end="")
        FP = len(cluster_result[index]) - result[index, a] if index < len(cluster_result) else 0
        for b in range(10):
            print(f"C{b+1}: {len(cluster_result[b]) - result[b, a]}; ", end="")
        sensitivity = temp_max / hashtable_label[str(a)] if hashtable_label[str(a)] != 0 else 0
        specificity = (len(binned_image_data) - hashtable_label[str(a)] - FP) / (len(binned_image_data) - hashtable_label[str(a)]) if (len(binned_image_data) - hashtable_label[str(a)]) != 0 else 0
        print(f"\nSensitivity: TP/Actual yes: {temp_max}/{hashtable_label[str(a)]}: {sensitivity}")
        print(f"Specificity: TN/Actual no: {len(binned_image_data) - FP}/{len(binned_image_data) - hashtable_label[str(a)]}: {specificity}")
        print(f"Digit {a} is clustered into cluster {index + 1}")
        print("*************************************")
        sensitivity_sum += sensitivity
        specificity_sum += specificity
    print(f"average sensitivity: {sensitivity_sum / 10}")
    print(f"average specificity: {specificity_sum / 10}")

def main():
    global binned_image_data, vec_label, cluster_result
    
    print("Reading MNIST dataset...")
    filename_image_test = "t10k-images.idx3-ubyte"
    vec_image_test = read_mnist(filename_image_test)
    print(f"Number of images: {len(vec_image_test)}")

    filename_label_test = "t10k-labels.idx1-ubyte"
    vec_label = read_mnist_label(filename_label_test)
    print(f"Number of labels: {len(vec_label)}")

    vec_image_test_preprocessed = preprocess_mnist(vec_image_test)
    binned_image_data = binning_data(vec_image_test_preprocessed)
    
    mixing_coef_init = initialize_mixing_coef()
    params_init = initialize_params()

    max_iter = 500
    tolerance = 0.0001
    for a in range(max_iter):
        joint_prob = calculate_joint_prob(binned_image_data, mixing_coef_init, params_init)
        cluster_result = evaluate_cluster_result(joint_prob)
        hashtable_cluster_members = evaluate_member_per_cluster(cluster_result, a + 1)
        responsibility = calculate_responsibility(joint_prob)
        params = update_params(binned_image_data, responsibility)
        delta = calculate_delta_params(params, params_init)
        
        if delta < tolerance or a == max_iter - 1:
            print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Converged! Print confusion matrix.")
            print_confusion_matrix(hashtable_cluster_members, cluster_result)
            break
        
        print(f"delta params: {delta}")
        mixing_coef_init = update_mixing_coef(responsibility)
        params_init = params

if __name__ == "__main__":
    main()