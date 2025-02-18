import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class AirPlumeModel:
    def __init__(self, domain_size_x: int, domain_size_y: int, num_points: int) -> None:
        """
        Инициализация модели распространения примесей в атмосфере.

        :param domain_size_x: Размер рассматривоемой области по оси X (в метрах).
        :param domain_size_y: Размер рассматривоемой области по оси Y (в метрах).
        :param num_points: Количество точек для построения сетки.
        """
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        self.num_points = num_points

    def create_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Создает сетку для расчетов.

        :return: Кортеж из двух массивов (x_grid, y_grid), представляющих сетку.
        """
        x_grid, y_grid = np.meshgrid(
            np.linspace(1e-20, self.domain_size_x, self.num_points), # Начинаем не с 0 что бы избежать деления на 0
            np.linspace(-self.domain_size_y, self.domain_size_y, 2 * self.num_points)
        )
        return x_grid, y_grid

    def plot(self, x_grid: np.ndarray, y_grid: np.ndarray, concentration: np.ndarray, min_concentration: float) -> None:
        """
        Отрисовывает график концентрации.

        :param x_grid: Сетка по оси X.
        :param y_grid: Сетка по оси Y.
        :param concentration: Массив концентраций.
        :param min_concentration: Минимальное значение концентрации для отображения.
        """
        concentration = np.where(concentration <= 0, 1e-20, concentration)
        levels = np.geomspace(min_concentration, concentration.max(), 20)  # Логарифмические уровни
        plt.contourf(
            x_grid, y_grid, concentration,
            levels=levels,
            cmap='cividis',  # Цветовая карта
            norm=LogNorm(vmin=min_concentration, vmax=concentration.max()),  # Логарифмическая нормализация
            linewidths=1  # Толщина линий
        )
        cbar = plt.colorbar()
        cbar.ax.set_yscale('log')
        return plt