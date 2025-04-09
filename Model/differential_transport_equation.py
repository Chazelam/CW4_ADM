from air_plume_model import AirPlumeModel
import numpy as np

class DifferentialTransportEquation(AirPlumeModel):
    def __init__(self, domain_size_x, domain_size_y, num_points, Q: float, x0: float, y0: float, u: float, 
                 v: float, mu: float, sigma: float):
        """
        Модель расчета концентрации примеси на основе уравнения переноса.
        
        :param Q: Мощность источника выбросов [кг/с]
        :param x0, y0: Координаты источника
        :param u, v: Компоненты вектора скорости ветра [м/с]
        :param mu: Коэффициент турбулентной диффузии [м²/с]
        :param sigma: Коэффициент поглощения примеси [1/с]
        """
        super().__init__(domain_size_x, domain_size_y, num_points)
        self.Q = Q
        self.x0, self.y0 = x0, y0
        self.u, self.v = u, v
        self.mu = mu
        self.sigma = sigma

    def calculate_concentration(self) -> np.ndarray:
        """Основной метод расчета поля концентрации"""
        self.create_grid()
        dx, dy, dr = self._calculate_distances()
        x_val = self._calculate_normalized_distance(dr)
        concentration = np.zeros_like(x_val)
        
        self._process_small_distance_case(dx, dy, x_val, concentration)
        self._process_large_distance_case(dx, dy, x_val, concentration)
        
        return concentration

    def _calculate_distances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Рассчитывает расстояния от источника до точек сетки"""
        dx = self.x_grid - self.x0  # Расстояние по оси X
        dy = self.y_grid - self.y0  # Расстояние по оси Y
        dr = np.sqrt(dx**2 + dy**2) # Евклидово расстояние
        return dx, dy, dr

    def _calculate_normalized_distance(self, dr: np.ndarray) -> np.ndarray:
        """Вычисляет нормированное расстояние для модели"""
        beta = self.sigma + (self.u**2 + self.v**2)/(4*self.mu)
        return np.sqrt(beta/self.mu) * dr

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
        exp_factor = np.exp((self.u*dx[mask] + self.v*dy[mask])/(2*self.mu))
        
        concentration[mask] = (self.Q/(2*np.pi*self.mu)) * tilde_k1 * exp_factor

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
        exp_factor = np.exp((self.u*dx[mask] + self.v*dy[mask])/(2*self.mu) - x_ge2)
        
        concentration[mask] = (self.Q/(2*x_ge2*np.pi*self.mu)) * tilde_k2 * exp_factor

    def _evaluate_polynomial(self, x: np.ndarray, coeffs: list) -> np.ndarray:
        """Вычисляет стандартный полином заданной степени"""
        return sum(coeff * x**i for i, coeff in enumerate(coeffs))

if __name__ == "__main__":
    domain_size_x, domain_size_y = 1000, 500
    num_points = 1000

    Q = 10          # Интенсивность источника
    x0, y0=100, 0   # Координаты источника
    u, v =1, 0      # Скорость ветра
    mu=1            # Коэффициент турбулентной диффузии
    sigma=0.05      # Коэффициент поглощения

    # Создание модели
    disspersion_model = DifferentialTransportEquation(
        domain_size_x        = domain_size_x,
        domain_size_y        = domain_size_y,
        num_points           = num_points,
        Q = Q, 
        x0 = x0, y0 = y0,
        u = u, v = v,
        mu = mu,
        sigma = sigma 
    )

    # Расчет концентрации
    concentration = disspersion_model.calculate_concentration()

    # Отрисовка графика
    plt = disspersion_model.plot(concentration)
    plt.show()