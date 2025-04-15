from air_plume_model import AirPlumeModel
import numpy as np

class DifferentialTransportEquation(AirPlumeModel):
    def __init__(
        self,
        domain_size_x: int,
        domain_size_y: int,
        num_points: int,
        source_emission_rate: float,
        source_positions: list[tuple[float, float, float]],
        wind_speed_x: float,
        wind_speed_y: float,
        diffusion_coefficient: float,
        absorption_coefficient: float,
    ) -> None:
        """
        Модель расчета концентрации примеси на основе уравнения переноса.
        
        :param source_emission_rate: Мощность источника выбросов [кг/с]
        :param source_positions: Список координат источников
        :param wind_speed_x, wind_speed_y: Компоненты вектора скорости ветра [м/с]
        :param diffusion_coefficient: Коэффициент турбулентной диффузии [м²/с]
        :param absorption_coefficient: Коэффициент поглощения примеси [1/с]
        """
        super().__init__(domain_size_x, domain_size_y, num_points)
        self.source_emission_rate = source_emission_rate
        self.source_positions = source_positions
        self._wind_speed_x = wind_speed_x
        self._wind_speed_y = wind_speed_y
        self._absorption_coefficient = absorption_coefficient
        self._diffusion_coefficient = diffusion_coefficient

    def calculate_concentration(self) -> np.ndarray:
        """Основной метод расчета поля концентрации"""
        self.create_grid()
        concentration = np.zeros_like(self._x_grid)

        for x0, y0, z0 in self.source_positions:
            dx, dy, dr = self._calculate_distances(x0, y0)
            x_val = self._calculate_normalized_distance(dr)
            concentration_for_source = np.zeros_like(x_val)
            
            self._process_small_distance_case(dx, dy, x_val, concentration_for_source)
            self._process_large_distance_case(dx, dy, x_val, concentration_for_source)
            concentration += concentration_for_source


        return concentration

    def _calculate_distances(self, x0: float, y0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Рассчитывает расстояния от источника до точек сетки"""
        dx = self._x_grid - x0  # Расстояние по оси X
        dy = self._y_grid - y0  # Расстояние по оси Y
        dr = np.sqrt(dx**2 + dy**2) # Евклидово расстояние
        return dx, dy, dr

    def _calculate_normalized_distance(self, dr: np.ndarray) -> np.ndarray:
        """Вычисляет нормированное расстояние для модели"""
        beta = self._absorption_coefficient + (self._wind_speed_x**2 + self._wind_speed_y**2)/(4*self._diffusion_coefficient)
        return np.sqrt(beta/self._diffusion_coefficient) * dr

    def _process_small_distance_case(self, dx: np.ndarray, dy: np.ndarray, 
                                   x_val: np.ndarray, concentration: np.ndarray) -> None:
        """Обработка случая для малых расстояний (x_val < 2)"""
        mask = x_val < 2
        x_lt2 = x_val[mask]
        if x_lt2.size == 0:
            return

        # Коэффициенты полинома для модифицированной функции Бесселя
        alpha_coeffs = [1, 3.5156229, 3.0899424, 1.2067492, 
                        0.2659732, 0.0360768, 0.0045813]
        
        t = x_lt2 / 3.75
        alpha = self._evaluate_polynomial(t**2, alpha_coeffs)
        
        tilde_x = x_lt2 / 2
        log_term = np.log(tilde_x)
        k1_coeffs = [-0.5721566, 0.4227842, 0.23069756, 0.0348589,
                     0.00262698, 0.0001075, 0.000074]
        
        tilde_k1 = -alpha * log_term + self._evaluate_polynomial(tilde_x**2, k1_coeffs)
        exp_factor = np.exp((self._wind_speed_x*dx[mask] + self._wind_speed_y*dy[mask])/(2*self._diffusion_coefficient))
        
        concentration[mask] = (self.source_emission_rate/(2*np.pi*self._diffusion_coefficient)) * tilde_k1 * exp_factor

    def _process_large_distance_case(self, dx: np.ndarray, dy: np.ndarray, 
                                    x_val: np.ndarray, concentration: np.ndarray) -> None:
        """Обработка случая для больших расстояний (x_val >= 2)"""
        mask = x_val >= 2
        x_ge2 = x_val[mask]
        if x_ge2.size == 0:
            return

        # Коэффициенты асимптотического разложения
        k2_coeffs = [1.25331414, -0.07832358, 0.02189568, -0.01062446,
                     0.00587872, -0.0025154,  0.000532]
        
        tilde_x = 2 / x_ge2
        tilde_k2 = self._evaluate_polynomial(tilde_x, k2_coeffs)
        exp_factor = np.exp((self._wind_speed_x*dx[mask] + self._wind_speed_y*dy[mask])/(2*self._diffusion_coefficient) - x_ge2)
        
        concentration[mask] = (self.source_emission_rate/(2*x_ge2*np.pi*self._diffusion_coefficient)) * tilde_k2 * exp_factor

    def _evaluate_polynomial(self, x: np.ndarray, coeffs: list) -> np.ndarray:
        """Вычисляет стандартный полином заданной степени"""
        return sum(coeff * x**i for i, coeff in enumerate(coeffs))

if __name__ == "__main__":
    domain_size_x, domain_size_y = 2500, 600
    num_points = 1000

    source_emission_rate = 10   # Интенсивность источника
    u, v = 1, 0                 # Скорость ветра
    mu = 1                      # Коэффициент турбулентной диффузии
    sigma = 0.05                # Коэффициент поглощения

    source_positions = [(0, 0),      (29, -34.5),
                        (76, 57),    (105, 22.5),  (134, -12), (163, -46.5),
                        (195, 90.5), (224, 56),    (253, 21.5)]
    # source_positions = [(100, 0)]
    source_positions = [(x + 200, y, 0) for x, y in source_positions]

    # Создание модели
    disspersion_model = DifferentialTransportEquation(
        domain_size_x          = domain_size_x,
        domain_size_y          = domain_size_y,
        num_points             = num_points,
        source_emission_rate   = source_emission_rate,
        source_positions       = source_positions,
        wind_speed_x           = u,
        wind_speed_y           = v,
        diffusion_coefficient  = mu,
        absorption_coefficient = sigma
    )

    # Расчет концентрации
    concentration = disspersion_model.calculate_concentration()

    # Отрисовка графика
    plt = disspersion_model.plot(concentration, fill=False)
    plt.show()