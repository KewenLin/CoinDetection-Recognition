import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os
import csv

def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * math.pi * area / (perimeter ** 2)

def region_std(gray_img, contour):
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    region_pixels = gray_img[mask == 255]
    return np.std(region_pixels)

def is_hollow_by_hierarchy(index, hierarchy, contours, area_threshold_ratio=0.4):
    child_idx = hierarchy[index][2]
    if child_idx == -1:
        return False

    parent_area = cv2.contourArea(contours[index])
    while child_idx != -1:
        child_area = cv2.contourArea(contours[child_idx])
        if parent_area == 0:
            return False
        if (child_area / parent_area) > area_threshold_ratio:
            return True
        child_idx = hierarchy[child_idx][0]
    return False

def contour_radius(contour):
    area = cv2.contourArea(contour)
    return math.sqrt(area / math.pi)

def get_orientation(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return 0
    angle = 0.5 * math.atan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
    return angle

def avg_intensity(gray_img, contour):
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return np.mean(gray_img[mask == 255])

def center_to_edge_intensity(gray_img, contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    radius = int(contour_radius(contour))

    h, w = gray_img.shape
    edge_y = np.clip(cy + radius, 0, h - 1)
    edge_x = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)
    cx = np.clip(cx, 0, w - 1)

    center_intensity = int(gray_img[cy, cx])
    edge_intensity = int(gray_img[edge_y, edge_x])
    return abs(center_intensity - edge_intensity)

def detect_edges(gray_img, contour):
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    edges = cv2.Canny(mask, 100, 200)
    return np.sum(edges)

def extract_color_features(color_img, contour):
    mask = np.zeros(color_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked_img = cv2.bitwise_and(color_img, color_img, mask=mask)

    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

    mean_hue = np.mean(hsv_img[:,:,0])
    mean_saturation = np.mean(hsv_img[:,:,1])
    mean_value = np.mean(hsv_img[:,:,2])

    return mean_hue, mean_saturation, mean_value

def detect_coins_white_bg(image_path, parameter, visualize=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)
    blurred = cv2.medianBlur(img, parameter['blur'])
    inverted = 255 - blurred
    binary = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    h, w = binary.shape
    k = parameter['k']
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and len(hierarchy.shape) == 3:
        hierarchy = hierarchy[0]

    min_radius = int(min(h, w) * parameter['sizePercent'])
    final_contours = []

    baseCircularity = parameter['circularity']
    baseRegionStd = parameter['std']

    for i, cnt in enumerate(contours):
        radius = contour_radius(cnt)
        if radius < min_radius:
            continue
        if circularity(cnt) < baseCircularity:
            continue
        if region_std(img, cnt) < baseRegionStd:
            continue
        if is_hollow_by_hierarchy(i, hierarchy, contours):
            continue
        final_contours.append(cnt)

    if visualize:
        color_output = color_img.copy()
        cv2.drawContours(color_output, final_contours, -1, (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return img, final_contours

def detect_coins_weird_bg(image_path, parameter, visualize=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)

    h, w = img.shape
    k = parameter['k']
    kernel = np.ones((k, k), np.uint8)

    blurred = cv2.medianBlur(img, parameter['blur'])
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    edges = cv2.Canny(blurred, 100, 200)

    combined = cv2.bitwise_or(binary, edges)

    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and len(hierarchy.shape) == 3:
        hierarchy = hierarchy[0]

    # if not contours or len(contours) == 0:
    #     print("No Contours")

    min_radius = int(min(h, w) * parameter['sizePercent'])
    final_contours = []

    baseCircularity = parameter['circularity']
    baseRegionStd = parameter['std']

    debug = False
    for i, cnt in enumerate(contours):
        radius = contour_radius(cnt)
        circ = circularity(cnt)
        std = region_std(img, cnt)
        hollow = is_hollow_by_hierarchy(i, hierarchy, contours)

        if radius < min_radius:
            if debug:
                print(f"Contour {i} rejected: radius too small ({radius:.2f} < {min_radius})")
            continue
        if circ < baseCircularity:
            if debug:
                print(f"Contour {i} rejected: low circularity ({circ:.2f})")
            continue
        if std < baseRegionStd:
            if debug:
                print(f"Contour {i} rejected: low stddev ({std:.2f})")
            continue
        if hollow:
            if debug:
                print(f"Contour {i} rejected: hollow")
            continue
        final_contours.append(cnt)

    if visualize:
        color_output = color_img.copy()
        cv2.drawContours(color_output, final_contours, -1, (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return img, final_contours

#Testing adaptability
def detect_coins_white_bg_scaled(image_path, parameter, visualize=True, scale = 1):
    circularity_sizes = {
        .5: .5,
        1: .4,
        1.25: .3,
        1.5: .2,
        1.8: .15,
        2: .1
    }

    kernel_sizes = {
        0.5: 7,  
        1.0: 11,
        1.25: 13, 
        1.5: 15,
        1.8: 21,
        2.0: 21  
    }

    kernel_size = kernel_sizes.get(scale,11)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)
    blurred = cv2.medianBlur(img, parameter['blur'])
    inverted = 255 - blurred
    binary = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    h, w = binary.shape
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and len(hierarchy.shape) == 3:
        hierarchy = hierarchy[0]

    min_radius = int(min(h, w) * parameter['sizePercent'])
    final_contours = []

    baseCircularity = circularity_sizes.get(scale, .3)
    baseRegionStd = parameter['std']

    for i, cnt in enumerate(contours):
        radius = contour_radius(cnt)
        if radius < min_radius:
            continue
        if circularity(cnt) < baseCircularity:
            continue
        if region_std(img, cnt) < baseRegionStd:
            continue
        if is_hollow_by_hierarchy(i, hierarchy, contours):
            continue
        final_contours.append(cnt)

    if visualize:
        color_output = color_img.copy()
        cv2.drawContours(color_output, final_contours, -1, (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return img, final_contours

def detect_coins(image_path, visualize=True, scale=1):
    circularity_sizes = {
        .5: .5,
        1: .4,
        1.25: .3,
        1.5: .2,
        2: .1
    }

    # distance_sizes = {
    #     .5: .01,
    #     1: .02,
    #     1.25 : .035,
    #     1.5 : .05,
    #     2 : .1,
    # }

    kernel_sizes = {
        0.5: 7,  
        1.0: 11,
        1.25: 13, 
        1.5: 15,
        2.0: 19  
    }

    kernel_size = kernel_sizes.get(scale,11)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)
    # blurred = cv2.GaussianBlur(img, (5,5), 0)
    blurred = cv2.medianBlur(img, 5)

    # tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    # _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and len(hierarchy.shape) == 3:
        hierarchy = hierarchy[0]

    # Check near contour
    # proximity_filtered = []

    # centers = []
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     centers.append((x + w//2, y + h//2))

    # tree = KDTree(centers)
    # used = [False] * len(contours)
    # proximity_filtered = []

    # for i, center in enumerate(centers):
    #     if used[i]:
    #         continue
    #     neighbors = tree.query_ball_point(center, r=contour_radius(contours[i]) * 2)
        
    #     best = contours[i]
    #     for j in neighbors:
    #         if used[j]:
    #             continue
    #         used[j] = True
    #         if cv2.contourArea(contours[j]) > cv2.contourArea(best):
    #             best = cv2.convexHull(np.concatenate([best, contours[j]]))

    #     proximity_filtered.append(best)

    # Filter
    h, w = img.shape
    min_radius = int(min(h, w) * .01) # 30 - 40 should work,

    final_contours = []
    debug = False
    for i, cnt in enumerate(contours):
        radius = contour_radius(cnt)
        circ = circularity(cnt)
        std = region_std(img, cnt)
        # hollow = is_hollow_by_hierarchy(i, hierarchy, proximity_filtered)

        if radius < min_radius:
            # if debug:
            #     print(f"Contour {i} rejected: radius too small ({radius:.2f} < {min_radius})")
            continue
        if circ < circularity_sizes.get(scale, .3):
            if debug:
                print(f"Contour {i} rejected: low circularity ({circ:.2f})")
            continue
        if std < 8:
            if debug:
                print(f"Contour {i} rejected: low stddev ({std:.2f})")
            continue
        # if hollow:
        #     if debug:
        #         print(f"Contour {i} rejected: hollow")
        #     continue
        if cv2.contourArea(cnt) < cv2.contourArea(cv2.convexHull(cnt)) * 0.6:
            if debug:
                print("Probably hollow")
            continue
        final_contours.append(cnt)
    
    print(f"{image_path} {len(final_contours)} contours")

    if visualize:
        color_output = color_img.copy()
        cv2.drawContours(color_output, final_contours, -1, (0, 255, 0), 3)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return img, final_contours

def extract_features(gray_img, color_img, contour):
    ori = get_orientation(contour)
    radius = contour_radius(contour)
    std = region_std(gray_img, contour)
    avg = avg_intensity(gray_img, contour)
    center_edge_diff = center_to_edge_intensity(gray_img, contour)
    edge_strength = detect_edges(gray_img, contour)
    hue, sat, val = extract_color_features(color_img, contour)
    COLOR_WEIGHT = 3.0
    return [ori, radius, std, avg, center_edge_diff, edge_strength, hue * COLOR_WEIGHT, sat * COLOR_WEIGHT, val * COLOR_WEIGHT]

def zoom_image(image, zoom_factor):
    if zoom_factor == 1 or zoom_factor <= 0:
        return image
    
    h, w = image.shape[:2]

    if zoom_factor > 1.0:
        # Zoom in: crop center, then resize for pixel
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = image[top:top+new_h, left:left+new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Zoom out: resize to smaller, then pad
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        zoomed = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

    return zoomed

def generate_scaled_and_rotated_features(filepath, label):
    print(f"Creating {filepath} rotated scaled dataset")

    X_aug = []
    y_aug = []

    original_img = cv2.imread(filepath)
    if original_img is None:
        return X_aug, y_aug

    # hs, ws = original_img.shape[:2]
    # center = (ws // 2, hs // 2)
    # for angle in range(0, 360, 15):
    #     rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     rotated_img = cv2.warpAffine(original_img, rot_mat, (ws, hs), flags=cv2.INTER_LINEAR)

    #     temp_path = f"temp_{filepath}_a{angle}.jpg"
    #     cv2.imwrite(temp_path, rotated_img)
    #     gray, contours = detect_coins_white_bg(temp_path, normalScaleWhiteBGParameter, False)
    #     os.remove(temp_path)

    #     if not contours or len(contours) == 0:
    #         print(f"No contours {filepath} angle : {angle}")
    #         continue

    #     cnt = max(contours, key=cv2.contourArea)
    #     feats = extract_features(gray, rotated_img, cnt)
    #     X_aug.append(feats)
    #     y_aug.append(label)

    for scale in scales:
        scaled_img = zoom_image(original_img, scale)
        hs, ws = scaled_img.shape[:2]
        center = (ws // 2, hs // 2)

        for angle in range(0, 360, 15):
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(scaled_img, rot_mat, (ws, hs), flags=cv2.INTER_LINEAR)

            temp_path = f"temp_{filepath}_s{scale:.2f}_a{angle}.jpg"
            cv2.imwrite(temp_path, rotated_img)
            gray, contours = detect_coins_white_bg(temp_path, scale_whiteBG_parameter, False)
            os.remove(temp_path)

            if not contours or len(contours) == 0:
                print(f"No contours {filepath} scale: {scale} angle : {angle}")
                continue

            cnt = max(contours, key=cv2.contourArea)
            feats = extract_features(gray, rotated_img, cnt)
            X_aug.append(feats)
            y_aug.append(label)

    return X_aug, y_aug

def loadDataset(dataset):
    X, y = [], []

    for path in dataset:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        print(f"Loading dataset from {path}")
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            for row in reader:
                *features, label = row
                features = [float(x) for x in features]
                X.append(features)
                y.append(label)
    
    return X, y

def createDataset(dataset_file="normalScaledWhiteBGDataset.csv"):
    X, y = [], []

    print("Create new dataset")
    for label, count in label_counts.items():
        for i in range(1, count + 1):
            filename = f"image\\scaled{label}{i}.jpg"
            if not os.path.exists(filename):
                print(f"{filename} not found.")
                continue
            print(f"Creating {filename} data")
            gray, contours = detect_coins_white_bg(filename, scale_whiteBG_parameter, visualize=False)
            color = cv2.imread(filename)
            for cnt in contours:
                feats = extract_features(gray, color, cnt)
                X.append(feats)
                y.append(label)

    for filepath in for_rotating_scalling:
        if not os.path.exists('image\\' + filepath):
            print(f"{filepath} not found.")
            continue

        X_aug, y_aug = generate_scaled_and_rotated_features(filepath, for_rotating_scalling[filepath])
        X.extend(X_aug)
        y.extend(y_aug)

    with open(dataset_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "orientation", "radius", "std_dev", "avg_intensity", 
            "center_edge_diff", "edge_strength", 
            "hue_weighted", "sat_weighted", "val_weighted", 
            "label"
        ])
        for feats, label in zip(X, y):
            writer.writerow(feats + [label])

    return X, y

def visualize_labeled_contours(image_path, labeled_results):
    img = cv2.imread(image_path)
    label_colors = {
        "penny": (0, 255, 0),
        "dime": (0, 0, 255),
        "nickel": (255, 0, 0),
        "quarter": (0, 255, 255),
        "unknown": (255, 255, 0)
    }

    for cnt, label, _ in labeled_results:
        if label == 'unknown' and not visualize_unknown:
            continue
        print(f'Detected {label}')
        color = label_colors.get(label, (255, 255, 255))
        cv2.drawContours(img, [cnt], -1, color, 3)

    # Draw legend in top-right corner
    legend_x = img.shape[1] - 250  
    legend_y = 30
    line_height = 70  

    for i, (label, color) in enumerate(label_colors.items()):
        y = legend_y + i * line_height
        cv2.rectangle(img, (legend_x, y), (legend_x + 60, y + 60), color, -1)
        cv2.putText(img, label, (legend_x + 70, y + 40),  
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{image_path} detected {len(labeled_results)}")
    plt.show()

# def classify_coins_at_scale(recognition_img_path, scale = 1):
#     X, y = loadDataset()

#     # Train KNN classifier
#     X, y = shuffle(X, y, random_state=42)
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(X, y)

#     threshold = 2
#     all_results = []

#     gray, contours = detect_coins(recognition_img_path, visualize=False, scale=scale)
#     color = cv2.imread(recognition_img_path)

#     for cnt in contours:
#         features = extract_features(gray, color, cnt)
#         features_scaled = scaler.transform([features])
#         dist, idx = knn.kneighbors(features_scaled)
#         label = y[idx[0][0]] if dist[0][0] < threshold else "unknown"
#         all_results.append((cnt, label, dist[0][0]))

#     visualize_labeled_contours(recognition_img_path, all_results)

def classify_coins_whiteBG(recognition_img_path, dataset, parameter):
    X, y = loadDataset(dataset)

    # Train KNN classifier
    X, y = shuffle(X, y, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    threshold = 2.5
    all_results = []

    gray, contours = detect_coins_white_bg(recognition_img_path, parameter, visualize=False)
    color = cv2.imread(recognition_img_path)

    for cnt in contours:
        features = extract_features(gray, color, cnt)
        features_scaled = scaler.transform([features])
        dist, idx = knn.kneighbors(features_scaled)
        label = y[idx[0][0]] if dist[0][0] < threshold else "unknown"
        all_results.append((cnt, label, dist[0][0]))

    visualize_labeled_contours(recognition_img_path, all_results)

def classify_coins_normal_scale_otherBG(recognition_img_path, dataset, parameter):
    X, y = loadDataset(dataset)

    # Train KNN classifier
    X, y = shuffle(X, y, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    threshold = 3
    all_results = []

    gray, contours = detect_coins_weird_bg(recognition_img_path, parameter, visualize=False)
    color = cv2.imread(recognition_img_path)

    for cnt in contours:
        features = extract_features(gray, color, cnt)
        features_scaled = scaler.transform([features])
        dist, idx = knn.kneighbors(features_scaled)
        label = y[idx[0][0]] if dist[0][0] < threshold else "unknown"
        all_results.append((cnt, label, dist[0][0]))

    visualize_labeled_contours(recognition_img_path, all_results)

scales = [1.8]
# scales = [1.5, 2.0]


label_counts = {
    # "penny": 7,
    # "dime": 7,
    # "nickel": 7,
    # "quarter": 7,
    'quarter' : 1,
    'dime' : 2,
    'nickel' : 2
}

for_rotating_scalling = {
    "image\\dimeHead.jpg" : 'dime',
    "image\\dimeTail.jpg" : 'dime',
    "image\\nickelHead.jpg" : 'nickel',
    "image\\nickelHead2.jpg" : 'nickel', 
    "image\\nickelTail.jpg" : 'nickel', # Maybe, could be repeated
    "image\\pennyHead.jpg" : 'penny', # 5 no contour
    "image\\pennyTail.jpg" : 'penny',
    "image\\pennyTail2.jpg" : 'penny',
    "image\\quarterHead.jpg" : 'quarter', # A bunch of no contours 
    "image\\quarterTail.jpg" : 'quarter', # 1 scale, 2 no contours, 
    # 'image\\quarterScaling.jpg' : 'quarter',
    # 'image\\nickelScaling.jpg' : 'nickel',
    # 'image\\dimeScaling.jpg' : 'dime',
}

normal_scale_whiteBG_dataset = [
    'dataset\\coin_features_dataset.csv',
    'dataset\\normalScaleWhiteBGDataset.csv',
]

scaled_whiteBG_dataset = [
    'dataset\\scaled_and_rotated_penny.csv',
    'dataset\\scaled_and_rotated_dime.csv',
    'dataset\\scaled_and_rotated_nickel.csv',
    'dataset\\scaled_and_rotated_quarter.csv',

    'dataset\\scaled_and_rotated_dime2.csv',
    'dataset\\scaled_and_rotated_nickel2.csv',
    'dataset\\scaled_and_rotated_quarter2.csv',

    'dataset\\scaledCoinFeatures.csv',
    'dataset\\scaled_and_rotated_dime3.csv',
    'dataset\\scaled_and_rotated_nickel3.csv',
    'dataset\\scaled_and_rotated_quarter3.csv',

    'dataset\\scaledCoinFeatures2.csv',
    'dataset\\scaledCoinFeatures3.csv',
    # "dataset\\scaled1.8CoinFeatures.csv",
]

otherBG_dataset = [
    "dataset\\rotatedHugeKernalSize.csv",
]

normal_scale_whiteBG_parameter = {
    'blur' : 5,
    "k" : 7,
    'sizePercent' : .03,
    'circularity' : .5,
    'std' : 10
}

scale_whiteBG_parameter = {
    'blur' : 7,
    "k" : 13,
    'sizePercent' : .05,
    'circularity' : .5,
    'std' : 10
}

normal_scale_otherBG_parameter = {
    'blur' : 13,
    "k" : 19,
    'sizePercent' : .02,
    'circularity' : .2,
    'std' : 5
}

visualize_unknown = True

# createDataset("dataset\\scaled1.8CoinFeatures.csv")

# Base scale with white background (Pretty reliable)
for i in range(1,6):
    classify_coins_whiteBG(f"image\\coinRecognition{i}.jpg", normal_scale_whiteBG_dataset, normal_scale_whiteBG_parameter)
classify_coins_whiteBG("image\\coinRecognition10.jpg", normal_scale_whiteBG_dataset, normal_scale_whiteBG_parameter)

# Testing scale on white background (Not reliable, dataset not set up for other scale except for 1.8 scale)
# Sort of random scaling for 6-9 and 14
for i in range(6,10):
    classify_coins_whiteBG(f"image\\coinRecognition{i}.jpg", scaled_whiteBG_dataset, scale_whiteBG_parameter)
classify_coins_whiteBG("image\\coinRecognition14.jpg", scaled_whiteBG_dataset, scale_whiteBG_parameter)

# Only coinRecognition15.jpg are actually 1.8x of normal scale, only reliable one
classify_coins_whiteBG("image\\coinRecognition15.jpg", scaled_whiteBG_dataset, scale_whiteBG_parameter)

# Testing OtherBG, still in detecting coin stage, dataset is just 1 coin image rotated and extracted features with
# normal_scale_otherBG_parameter which did not work. (Doesn't work)
for i in range(11, 14):
    classify_coins_whiteBG(f"image\\coinRecognition{i}.jpg", otherBG_dataset, normal_scale_otherBG_parameter)
