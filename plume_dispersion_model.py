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


class GaussianPlumeModel(AirPlumeModel):
    # Коэффициенты дисперсии для классов атмосферной стабильности
    DISPERSION_COEFFICIENTS = {
        "B": {"sigma_y": (0.16, 0.0001), "sigma_z": (0.12, 0.0)},
        "C": {"sigma_y": (0.11, 0.0001), "sigma_z": (0.08, 0.0002)},
        "D": {"sigma_y": (0.08, 0.0001), "sigma_z": (0.06, 0.0015)},
    }

    def __init__(
        self,
        domain_size_x: int,
        domain_size_y: int,
        num_points: int,
        source_emission_rate: float,
        wind_speed: float,
        release_height: float,
        source_positions: list[tuple[int, int, int]]
    ) -> None:
        """
        Инициализация Стационарной Гауссовой модели рассеивания примеси.

        :param domain_size_x: Размер рассматривоемой области по оси X (в метрах).
        :param domain_size_y: Размер рассматривоемой области по оси Y (в метрах).
        :param num_points: Количество точек для построения сетки.
        :param source_emission_rate: мощность непрерывного точечного источника загрязнения (г/с).
        :param release_height: эффективная высота источника загрязнения (м).
        :param wind_speed: Скорость ветра на высоте release_height (м/с).
        :param source_positions: Список позиций источников загрязнения (x0, y0, z0).
        """
        super().__init__(domain_size_x, domain_size_y, num_points)

        self.source_emission_rate = source_emission_rate
        self.wind_speed = wind_speed
        self.release_height = release_height
        self.source_positions = source_positions

    @staticmethod
    def determine_atmospheric_stability(wind_speed: float) -> str:
        """
        Определяет класс атмосферной стабильности на основе скорости ветра.
        !!! Исключительно для ситуации АО «ПКС - Водоканал» !!!

        :param wind_speed: Скорость ветра (м/с).
        :return: Класс атмосферной стабильности ("B", "C" или "D").
        """
        if wind_speed < 2:
            return "B"
        elif wind_speed < 5:
            return "C"
        else:
            return "D"

    def calculate_plume_dispersion(self, x: float, stability_class: str) -> tuple[float, float]:
        """
        Рассчитывает коэффициенты горизонтальной (sigma_y) и вертикальной (sigma_z) дисперсии.

        :param x: Расстояние от источника (в метрах).
        :param stability_class: Класс атмосферной стабильности ("B", "C" или "D").
        :return: Кортеж (sigma_y, sigma_z) коэффициентов дисперсии.
        """
        coeff = self.DISPERSION_COEFFICIENTS.get(stability_class)
        if not coeff:
            raise ValueError(f"Unknown stability class: {stability_class}")

        a_y, b_y = coeff["sigma_y"]
        a_z, b_z = coeff["sigma_z"]

        sigma_y = a_y * x / np.sqrt(1 + b_y * x)
        sigma_z = a_z * x / np.sqrt(1 + b_z * x)

        return sigma_y, sigma_z

    def calculate_concentration(self, x: np.ndarray, y: np.ndarray, z: float, stability_class: str) -> np.ndarray:
        """
        Рассчитывает концентрацию загрязнителя с использованием Стационарной Гауссовой модели рассеивания примеси.

        :param x: Расстояние по направлению ветра от источника (м).
        :param y: Поперечное расстояние от центральной линии (м).
        :param z: Вертикальное расстояние от земли (м).
        :param stability_class: Класс атмосферной стабильности ("B", "C" или "D").
        :return: Массив концентраций.
        """

        if not self.source_positions:
            raise ValueError("source_positions cannot be empty")

        concentration = np.zeros(x.shape)
        for source in self.source_positions:
            # Маска для точек, которые находятся "после" источника по оси X
            mask = x >= source[0]
            
            # Расстояние от источника (только для точек, которые находятся "после" источника)
            distance = np.where(mask, x - source[0], 0)
            
            # Рассчитываем коэффициенты дисперсии только для точек, которые находятся "после" источника
            sigma_y, sigma_z = self.calculate_plume_dispersion(distance, stability_class)
            
            # Заменяем нулевые значения на очень маленькие положительные числа
            sigma_y = np.where(sigma_y <= 0, 1e-20, sigma_y)
            sigma_z = np.where(sigma_z <= 0, 1e-20, sigma_z)
            
            # Вычисляем концентрацию только для точек, которые находятся "после" источника
            term1 = self.source_emission_rate / (2 * np.pi * self.wind_speed * sigma_y * sigma_z)
            term2 = np.exp(-((y - source[1]) ** 2) / (2 * sigma_y ** 2))
            term3 = np.exp(-((z - self.release_height) ** 2) / (2 * sigma_z ** 2))
            term4 = np.exp(-((z + self.release_height) ** 2) / (2 * sigma_z ** 2))
            
            # Добавляем концентрацию только для точек, которые находятся "после" источника
            concentration += np.where(mask, term1 * term2 * (term3 + term4), 0)
        
        return concentration


# Основной скрипт
if __name__ == "__main__":
    # Константы
    SOURCE_EMISSION_RATE = 10  # Скорость выброса источника (кг/с)
    WIND_SPEED = 3  # Скорость ветра (м/с)
    RELEASE_HEIGHT = 0  # Высота выброса (м)
    MIN_CONCENTRATION = 5*10e-6 # Минимальный порог концентрации
    SOURCE_POSITIONS = [(100, 0, 0),   (180, 0, 0),
                        (100, 40, 0),  (180, 40, 0), (260, 40, 0), 
                                       (180, 80, 0), (260, 80, 0), 
                                                     (260, 120, 0)]  # Позиция источника (x0, y0, z0)

    # Параметры сетки
    DOMAIN_SIZE_X = 6000  # Размер области по x (м)
    DOMAIN_SIZE_Y = 1500  # Размер области по y (м)
    NUM_POINTS = 2000  # Количество точек для построения

    # Создание модели
    plume_model = GaussianPlumeModel(
        domain_size_x        = DOMAIN_SIZE_X,
        domain_size_y        = DOMAIN_SIZE_Y,
        num_points           = NUM_POINTS,
        source_emission_rate = SOURCE_EMISSION_RATE,
        wind_speed           = WIND_SPEED,
        release_height       = RELEASE_HEIGHT,
        source_positions     = SOURCE_POSITIONS
    )

    # Определение класса стабильности
    stability_class = plume_model.determine_atmospheric_stability(WIND_SPEED)

    # Создание сетки
    x_grid, y_grid = plume_model.create_grid()


    # Расчет концентрации
    concentration = plume_model.calculate_concentration(x_grid, y_grid, 1, stability_class)

    # Отрисовка графика
    plume_model.plot(x_grid, y_grid, concentration, MIN_CONCENTRATION)
    plt.show()