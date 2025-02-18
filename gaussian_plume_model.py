from air_plume_model import AirPlumeModel
import numpy as np

class GaussianPlumeModel(AirPlumeModel):
    def __init__(self,
                 domain_size_x: int,
                 domain_size_y: int,
                 num_points: int,
                 source_emission_rate: float,
                 wind_speed: float,
                 wind_direction: float,
                 release_height: float,
                 source_positions: list[tuple[int, int, int]]) -> None:
        """
        Инициализация Гауссовой модели рассеивания примеси.

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

        # Поворот координат источников в соответствии с направлением ветра
        if wind_direction % 360 != 0:
            rotated_source_positions = []
            for source in source_positions:
                x_rotated, y_rotated = self.rotate_coordinates(source[0], source[1], -wind_direction)
                rotated_source_positions.append((x_rotated, y_rotated, source[2]))

            # Обновляем позиции источников в модели
            self.source_positions = self.shift_coordinates(rotated_source_positions)
        else:
            self.source_positions = self.shift_coordinates(source_positions)

    @staticmethod
    def rotate_coordinates(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Поворачивает координаты (x, y) на заданный угол.

        :param x: Массив координат по оси X.
        :param y: Массив координат по оси Y.
        :param angle_deg: Угол поворота в градусах.
        :return: Кортеж (x_rotated, y_rotated) - повернутые координаты.
        """
        angle_rad = np.radians(angle_deg)  # Преобразуем угол в радианы
        x_rotated = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rotated = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        return x_rotated, y_rotated     

    @staticmethod
    def shift_coordinates(sources: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """
        Смещает координаты источников так, чтобы они не уходили в отрицательные значения.

        :param sources: Список координат источников (x, y, z).
        :return: Кортеж (смещенные координаты, (смещение по x, смещение по y)).
        """
        # Находим минимальные значения по x и y
        min_x = min(source[0] for source in sources)
        min_y = min(source[1] for source in sources)

        # Смещаем координаты
        shifted_sources = [
            (source[0] + abs(min_x), source[1] + abs(min_y), source[2])
            for source in sources
        ]

        return shifted_sources

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


class StationaryGaussianPlumeModel(GaussianPlumeModel):
    # Коэффициенты дисперсии для классов атмосферной стабильности
    DISPERSION_COEFFICIENTS = {
        "B": {"sigma_y": (0.16, 0.0001), "sigma_z": (0.12, 0.0)},
        "C": {"sigma_y": (0.11, 0.0001), "sigma_z": (0.08, 0.0002)},
        "D": {"sigma_y": (0.08, 0.0001), "sigma_z": (0.06, 0.0015)},
    }

    def __init__(self, domain_size_x, domain_size_y, num_points, source_emission_rate, wind_speed, wind_direction, release_height, source_positions):
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
        super().__init__(domain_size_x, domain_size_y, num_points, source_emission_rate, wind_speed, wind_direction, release_height, source_positions)

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
            
            # Заменяем нулевые значения на очень маленькие положительные числа для избежания деления на 0
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