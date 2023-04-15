import numpy as np
import pandas as pd
import plotly.express as ex
import matplotlib.pyplot as plt
import cv2 as cv
import seaborn as sns
plt.rc('figure', figsize=(17, 11))
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
target = pd.get_dummies(train.label)
label = pd.DataFrame({'labels': train.pop('label')})
train[train < 210] = 0
train[train > 0] = 255
test[test < 210] = 0
test[test > 0] = 255
train.head(3)
train_images = [img.values.reshape(28, 28).astype(np.float64) for (_, img) in train.iterrows()]
test_images = [img.values.reshape(28, 28).astype(np.float64) for (_, img) in test.iterrows()]
N = 784
digit_pixel_prob = []
for i in range(0, 10):
    ones = train.loc[label.query(f'labels=={i}').index, :]
    ones = ones / 255
    pf = (ones.sum() / len(ones)).values
    ipf = pf.reshape((28, 28))
    digit_pixel_prob.append(ipf)
(fig, axes) = plt.subplots(nrows=2, ncols=5)
number = 0
for ax in axes.flat:
    ax.set_title(f' Pixels Probabilites Num: [{number}]')
    im = ax.imshow(digit_pixel_prob[number], cmap='coolwarm', vmax=1)
    number += 1
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)


def calculate_image_gradients(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = Kx.T
    Gx = cv.filter2D(img, -1, Kx)
    Gy = cv.filter2D(img, -1, Ky)
    return (Gx, Gy)
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
(Gx, Gy) = calculate_image_gradients(img)
plt.subplot(2, 3, 1)
plt.title('Gx', fontsize=16, fontweight='bold')
plt.imshow(Gx, cmap='jet')
plt.subplot(2, 3, 2)
plt.title('Gy', fontsize=16, fontweight='bold')
plt.imshow(Gy, cmap='jet')
plt.subplot(2, 3, 3)
plt.title('Original Image', fontsize=16, fontweight='bold')
plt.imshow(img, cmap='jet')
img = train_images[8].copy()
(Gx, Gy) = calculate_image_gradients(img)
plt.subplot(2, 3, 4)
plt.title('Gx', fontsize=16, fontweight='bold')
plt.imshow(Gx, cmap='jet')
plt.subplot(2, 3, 5)
plt.title('Gy', fontsize=16, fontweight='bold')
plt.imshow(Gy, cmap='jet')
plt.subplot(2, 3, 6)
plt.title('Original Image', fontsize=16, fontweight='bold')
plt.imshow(img, cmap='jet')
titles = ['$Gx^{2}$', '$Gy^{2}$', '$Gy\\cdot Gx}$']
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
(Gx, Gy) = calculate_image_gradients(img)
moments = [Gx ** 2, Gx * Gy, Gy ** 2]
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
for i in range(0, 3):
    plt.subplot(2, 3, i + 1)
    plt.title(titles[i], fontsize=16, fontweight='bold')
    plt.imshow(moments[i], cmap='jet')
img = train_images[8].copy()
(Gx, Gy) = calculate_image_gradients(img)
moments = [Gx ** 2, Gx * Gy, Gy ** 2]
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
for i in range(0, 3):
    plt.subplot(2, 3, 3 + (i + 1))
    plt.title(titles[i], fontsize=16, fontweight='bold')
    plt.imshow(moments[i], cmap='jet')
titles = ['$Gx^{2}$', '$Gy^{2}$', '$Gy\\cdot Gx}$']
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
(Gx, Gy) = calculate_image_gradients(img)
moments = [Gx ** 2, Gx * Gy, Gy ** 2]
moments = [cv.GaussianBlur(moments[i], (3, 3), 2) for i in range(0, 3)]
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
for i in range(0, 3):
    plt.subplot(2, 3, i + 1)
    plt.title('Filtered ' + titles[i], fontsize=16, fontweight='bold')
    plt.imshow(moments[i], cmap='jet')
img = train_images[8].copy()
(Gx, Gy) = calculate_image_gradients(img)
moments = [Gx ** 2, Gx * Gy, Gy ** 2]
moments = [cv.GaussianBlur(moments[i], (3, 3), 2) for i in range(0, 3)]
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
for i in range(0, 3):
    plt.subplot(2, 3, 3 + (i + 1))
    plt.title('Filtered ' + titles[i], fontsize=16, fontweight='bold')
    plt.imshow(moments[i], cmap='jet')
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
plt.subplot(2, 2, 1)
plt.title('Original Image', weight='bold')
plt.imshow(img, cmap='jet')
plt.subplot(2, 2, 2)
(Gx, Gy) = calculate_image_gradients(img)
M = [Gx ** 2, Gx * Gy, Gy ** 2]
M = [cv.GaussianBlur(Prod, (3, 3), 2) for Prod in M]
detM = M[0] * M[2] - M[1] ** 2
traceM = M[0] + M[2]
R_scores = detM - 0.06 * traceM ** 2
plt.title('R Score Matrix', weight='bold')
plt.imshow(R_scores, cmap='jet')
plt.colorbar()
img = train_images[8].copy()
plt.subplot(2, 2, 3)
plt.title('Original Image', weight='bold')
plt.imshow(img, cmap='jet')
plt.subplot(2, 2, 4)
(Gx, Gy) = calculate_image_gradients(img)
M = [Gx ** 2, Gx * Gy, Gy ** 2]
M = [cv.GaussianBlur(Prod, (3, 3), 2) for Prod in M]
detM = M[0] * M[2] - M[1] ** 2
traceM = M[0] + M[2]
R_scores = detM - 0.06 * traceM ** 2
plt.title('R Score Matrix', weight='bold')
plt.imshow(R_scores, cmap='jet')
plt.colorbar()

def get_R_scores(Gx, Gy, Sigma=2, alpha=0.06, tsh=0.35):
    M = [Gx ** 2, Gx * Gy, Gy ** 2]
    M = [cv.GaussianBlur(Prod, (3, 3), Sigma) for Prod in M]
    detM = M[0] * M[2] - M[1] ** 2
    traceM = M[0] + M[2]
    R_scores = detM - alpha * traceM ** 2
    cross_kernel = np.ones((3, 3), np.uint8)
    R_dilate = cv.dilate(R_scores, cross_kernel)
    R_th = R_scores > R_scores.max() * tsh
    R_nms = R_scores >= R_dilate
    R_final = R_th * R_nms
    return R_final.astype(np.int)
img = np.zeros((21, 21), dtype=np.float32)
img[5:-5, 5:-5] = 200
plt.subplot(2, 2, 1)
plt.title('Original Image', weight='bold')
plt.imshow(img, cmap='jet')
plt.subplot(2, 2, 2)
plt.title('R Score Matrix after NMS', weight='bold')
(Gx, Gy) = calculate_image_gradients(img)
plt.imshow(get_R_scores(Gx, Gy), cmap='jet')
plt.colorbar()
img = train_images[8].copy()
plt.subplot(2, 2, 3)
plt.title('Original Image', weight='bold')
plt.imshow(img, cmap='jet')
plt.subplot(2, 2, 4)
(Gx, Gy) = calculate_image_gradients(img)
plt.title('R Score Matrix', weight='bold')
plt.imshow(get_R_scores(Gx, Gy), cmap='jet')
plt.colorbar()

def Calculate_Harris_Corners(img, Sigma=0.3, alpha=0.06, tsh=0.1):
    (gx, gy) = calculate_image_gradients(img)
    return get_R_scores(gx, gy, Sigma, alpha, tsh)
for i in range(0, 6):
    plt.subplot(2, 3, i + 1)
    target = train_images[10 + i]
    plt.imshow(target, cmap='gray', label='Harris Corners')
    (y, x) = np.nonzero(Calculate_Harris_Corners(target))
    plt.plot(x, y, 'ro')
    plt.legend(['Corners'])
HARRIS_BLOCK_SIZE = 3
HARRIS_K_SIZE = 5
HARIS_ALPHA = 0.03
characteristic_R_Matrix = []
for number in range(0, 10):
    aux = np.zeros_like(train_images[0], np.float64)
    for zero_ind in label[label.labels == number].index:
        aux += cv.cornerHarris(np.float32(train_images[zero_ind]), HARRIS_BLOCK_SIZE, HARRIS_K_SIZE, HARIS_ALPHA)
    characteristic_R_Matrix.append(aux)
(fig, axes) = plt.subplots(nrows=2, ncols=5)
number = 0
for ax in axes.flat:
    ax.set_title(f'[{number}] Characteristic R Matrix')
    im = ax.imshow(characteristic_R_Matrix[number], cmap='jet')
    number += 1
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

for mat in characteristic_R_Matrix:
    mat[mat < mat.max() * 0.6] = 0
(fig, axes) = plt.subplots(nrows=2, ncols=5)
number = 0
for ax in axes.flat:
    ax.set_title(f'[{number}] Characteristic R Matrix')
    im = ax.imshow(characteristic_R_Matrix[number], cmap='jet')
    number += 1
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)


def Distances_From_Characteristic_Matrixs(R_Scores):
    distances = []
    for number in range(0, 10):
        distances.append(np.sum(characteristic_R_Matrix[number] * R_Scores))
    return distances / (max(distances) + 0.001)
Distances_From_Characteristic_Matrixs(Calculate_Harris_Corners(test_images[405]))
predictions = []
for number in test_images:
    predictions.append(np.argmax(Distances_From_Characteristic_Matrixs(cv.cornerHarris(np.float32(number), HARRIS_BLOCK_SIZE, HARRIS_K_SIZE, HARIS_ALPHA))))
p_df = pd.DataFrame({'Prediction': predictions})
sample = p_df.iloc[5:11]
sample
for n in range(0, 6):
    plt.subplot(2, 3, n + 1)
    plt.title('Predicted: {}'.format(sample.iloc[n, 0]))
    cor = cv.cornerHarris(np.array(test_images[sample.index[n]], dtype=np.float32), HARRIS_BLOCK_SIZE, HARRIS_K_SIZE, HARIS_ALPHA)
    cor[cor < cor.max() / 1.5] = 0
    p = test_images[sample.index[n]].copy()
    p = cv.cvtColor(p.astype(np.uint8), cv.COLOR_GRAY2RGB)
    p[cor > cor.max() / 3] = (255, 0, 0)
    plt.imshow(p)
