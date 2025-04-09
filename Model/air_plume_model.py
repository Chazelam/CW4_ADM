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
        self._domain_size_x = domain_size_x
        self._domain_size_y = domain_size_y
        self._num_points = num_points

    def create_grid(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Создает сетку для расчетов.
        """
        self._x_grid, self._y_grid = np.meshgrid(
            np.linspace(1e-20, self._domain_size_x, self._num_points), # Начинаем не с 0 что бы избежать деления на 0
            np.linspace(-self._domain_size_y, self._domain_size_y, 2 * self._num_points)
        )

    def plot(self, concentration: np.ndarray[float], min_concentration: float = 1e-40, fill: bool = True) -> None:
        """
        Отрисовывает график концентрации.

        :param concentration: Массив концентраций.
        :param min_concentration: Минимальное значение концентрации для отображения.
        :param fill: Заливка графика, True по умолчанию.
        """
        concentration = np.where(concentration <= 0, 1e-50, concentration)
        levels = np.geomspace(min_concentration, concentration.max(), 20)  # Логарифмические уровни
        if fill:
            plt.contourf(
                self._x_grid, self._y_grid, concentration,
                levels=levels,
                cmap='cividis',  # Цветовая карта
                norm=LogNorm(vmin=min_concentration, vmax=concentration.max()),  # Логарифмическая нормализация
            )
        else:
            plt.contour(
                self._x_grid, self._y_grid, concentration,
                levels=levels,
                cmap='cividis',  # Цветовая карта
                norm=LogNorm(vmin=min_concentration, vmax=concentration.max()),  # Логарифмическая нормализация
                linewidths=1  # Толщина линий
            )
        cbar = plt.colorbar()
        cbar.ax.set_yscale('log')
        return plt
