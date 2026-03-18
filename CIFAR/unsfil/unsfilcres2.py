# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 22:57:25 2025

@author: Kiselev_Mi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from ReadConvolutions import ReadConvolutions 

ConvolutionDimensions = [[22, 3, 3], [9, 4, 22]]
Stride1 = 1
PoolingSize = 2
Convolutions = ReadConvolutions("CIFAR10.filters.csv", ConvolutionDimensions)

nrows = 3

class CustomCNN(nn.Module):
    def __init__(self, first_conv_weights, second_conv_weights):
        """
        Args:
            first_conv_weights: numpy array shape (n, 3, 3, 3) - веса первой свертки
            second_conv_weights: numpy array shape (m, n, 4, 4) - веса второй свертки
        """
        super(CustomCNN, self).__init__()
        
        # Получаем размеры из входных тензоров
        n, _, _, _ = first_conv_weights.shape
        m, _, _, _ = second_conv_weights.shape
        
        # Первая сверточный слой: 3 входных канала, n выходных, ядро 3x3
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=n, 
            kernel_size=3, 
            stride=1, 
            padding=0  # padding=1 для сохранения размеров при kernel_size=3 и stride=1
        )
        
        # Вторая сверточный слой: n входных каналов, m выходных, ядро 4x4
        self.conv2 = nn.Conv2d(
            in_channels=n, 
            out_channels=m, 
            kernel_size=4, 
            stride=1, 
            padding=0  # padding=1 для kernel_size=4
        )
        
        # Активация и пулинг
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # mean pooling 2x2
        
        # Инициализация весов предзаданными значениями
        self._initialize_weights(first_conv_weights, second_conv_weights)
    
    def _initialize_weights(self, first_conv_weights, second_conv_weights):
        """Инициализация сверточных слоев предзаданными весами"""
        
        # Преобразуем numpy массивы в torch тензоры
        # Первая свертка: (n, 3, 3, 3) -> (n, 3, 3, 3) в PyTorch
        conv1_weights_tensor = torch.FloatTensor(first_conv_weights)
        # PyTorch ожидает веса в формате (out_channels, in_channels, height, width)
        # Если ваш массив уже в правильном формате, просто используем его
        
        # Вторая свертка: (m, n, 4, 4) -> (m, n, 4, 4)
        conv2_weights_tensor = torch.FloatTensor(second_conv_weights)
        
        # Устанавливаем веса
        self.conv1.weight.data = conv1_weights_tensor
        self.conv2.weight.data = conv2_weights_tensor
        
        # Инициализируем смещения нулями
        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()
    
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        
        # Первая свертка + ReLU
        x = self.conv1(x)  # shape: (batch_size, n, height, width)
        x = self.relu(x)
        
        # Mean pooling 2x2
        x = self.pool(x)  # shape: (batch_size, n, height/2, width/2)
        
        # Вторая свертка
        x = self.conv2(x)  # shape: (batch_size, m, (height/2)-3, (width/2)-3)
        # При kernel_size=4, stride=1 и padding=1 выходной размер уменьшается на 2
        
        return x
    
# Преобразуем веса в нужный формат.
first_conv_weights = np.transpose(Convolutions[0], (0, 3, 1, 2))  # (10, 3, 3, 3)
second_conv_weights = np.transpose(Convolutions[1], (0, 3, 1, 2))  # (30, 10, 4, 4)

# Создаем модель
model = CustomCNN(first_conv_weights, second_conv_weights)
model.eval()  # Переводим в режим оценки (выключены dropout, batchnorm статистика)

# Целевой слой и фильтр
input_size = first_conv_weights.shape[2] + (second_conv_weights.shape[2] * PoolingSize - 1) * Stride1
fig, ax = plt.subplots(nrows, len(second_conv_weights) // nrows, sharex=True, sharey=True, layout="constrained", figsize=(42, 10))
target_filter = 0
for r in range(nrows):
    for c in range(len(second_conv_weights) // nrows):
        print(f"Getting the prototype image for convolution {target_filter}")
        
        # Инициализация изображения (случайный шум)
        X = torch.empty(1, first_conv_weights.shape[1], input_size, input_size).uniform_(0, 256)
        X.requires_grad = True
        
        # Оптимизатор работает ТОЛЬКО с изображением X. Веса модели заморожены.
        optimizer = optim.Adam([X], lr=0.05)
        
        # Цикл оптимизации
        for iteration in range(10000):
            optimizer.zero_grad()
        
            # Прямой проход
            activations = []
            def hook_fn(module, input, output):
                # Сохраняем активацию нужного слоя
                activations.append(output)
            hook = model.conv2.register_forward_hook(hook_fn) # вешаем хук на 2-й слой
        
            output = model(X)
            hook.remove()  # Удаляем хук
        
            target_activation = activations[0]  # Активации второго слоя
            # Целевая функция: средняя активация target_filter по всей карте
            loss = -target_activation[0, target_filter, :, :].mean()  # Минус, потому что хотим максимизировать 
                                                                      # (минимизируем отрицание)
                                                                      # И на самом деле там только 1 значение...
        
            # Добавляем регуляризацию Total Variation (TV)
            tv_loss = torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:])) + \
                      torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :]))
            loss = loss + 0.0001 * tv_loss   # Регуляризация не слишком ли сильна?
        
            # Обратный проход
            loss.backward()
        
            # Шаг оптимизации
            optimizer.step()
        
            # (Опционально) Периодическое размытие
            if iteration % 100 == 0:
                X.data = transforms.GaussianBlur(kernel_size=3, sigma=0.5)(X.data)
        
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}')
                if iteration == 0:
                    former_loss = loss.item()
                else:
                    if former_loss < loss.item():
                        print('Loss stabilized')
                        break
                    former_loss = loss.item()
            
            with torch.no_grad():
                X.clamp_(0, 255)
        
        # Визуализация результата
        result_img = X.detach().squeeze().permute(1, 2, 0).cpu().numpy() / 256
        # Нормализация к диапазону [0, 1] для отображения
        # result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
        ax[r, c].imshow(result_img)
        ax[r, c].axis('off')
        target_filter += 1
plt.show()
