import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def pca(X):
    """Аналіз головних компонентів (PCA)
    Вхід: X — матриця (кількість зображень, кількість пікселів)
    Вихід: V — матриця проєкції, S — дисперсія, mean_X — середнє зображення
    """
    num_data, dim = X.shape

    # Центрування даних (віднімання середнього)
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # Компактний розрахунок коваріаційної матриці
        M = np.dot(X, X.T)  # коваріаційна матриця (100 x 100)
        e, EV = np.linalg.eigh(M)  # власні значення і вектори
        tmp = np.dot(X.T, EV).T  # проєкція на простір зображень
        V = tmp[::-1]  # розгортаємо за спаданням
        S = np.sqrt(np.maximum(e[::-1], 0))  # розрахунок дисперсії

        # Нормалізація кожного власного вектора без помилки індексації
        for i in range(V.shape[0]):
            if S[i] > 1e-10:
                V[i, :] /= S[i]
    else:
        # Звичайне розкладання SVD
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]

    return V, S, mean_X


# 1. Шлях до папки зі шрифтами
path = r"C:\Pyton\laba4\a_thumbs"

if not os.path.exists(path):
    raise FileNotFoundError(f"Вказану папку не знайдено: {path}")

# Отримання та сортування списку файлів зображень
imlist = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
imlist.sort()

# Завдання (Варіант 1): беремо перші 100 зображень
imlist = imlist[:100]
imnbr = len(imlist)

if imnbr == 0:
    raise ValueError(f"У папці {path} не знайдено графічних файлів!")

print(f"Завантажено {imnbr} зображень для аналізу (Варіант 1).")

# 2. Отримання розміру та формування матриці векторів
sample_im = Image.open(imlist[0]).convert("L")
m, n = np.array(sample_im).shape

immatrix = np.array(
    [np.array(Image.open(im).convert("L")).flatten() for im in imlist],
    dtype=np.float64,
)

# 3. Виконання PCA
print("Обчислення головних компонент...")
V, S, immean = pca(immatrix)

# 4. Збереження результатів за допомогою pickle (серіалізація)
output_pkl = os.path.join(path, "font_pca_modes.pkl")
with open(output_pkl, "wb") as f:
    pickle.dump(immean, f)
    pickle.dump(V, f)
print(f"Модель успішно збережено у: {output_pkl}")

# 5. Візуалізація середнього зображення та перших 7 головних компонент
plt.figure(figsize=(12, 6))
plt.gray()

# Середнє зображення
plt.subplot(2, 4, 1)
plt.imshow(immean.reshape(m, n))
plt.title("Середнє (Mean)")
plt.axis("off")

# Перші 7 компонент (моди найбільшої варіативності)
for i in range(7):
    plt.subplot(2, 4, i + 2)
    plt.imshow(V[i].reshape(m, n))
    plt.title(f"Компонента {i + 1}")
    plt.axis("off")

plt.tight_layout()

# Збереження графіка для звіту
output_plot = os.path.join(path, "lab4_pca_results.png")
plt.savefig(output_plot, dpi=300)
print(f"Графік для звіту збережено у: {output_plot}")

plt.show()