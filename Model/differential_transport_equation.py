from Model.air_plume_model import AirPlumeModel
import numpy as np

class DifferentialTransportEquation(AirPlumeModel):
    def __init__(self, domain_size_x, domain_size_y, num_points):
        super().__init__(domain_size_x, domain_size_y, num_points)

    def calculate_concentration(self, Q: float, x0: float, y0: float, u: float, 
                                v: float, mu: float, sigma: float) -> np.ndarray:
        """
        Вычисляет концентрацию примеси согласно аналитической модели основаной на полуэмпирическом дифференциальном уравнение переноса.

        :param Q: Интенсивность источника.
        :param x0, y0: Координаты источника.
        :param u, v: Компоненты скорости ветра.
        :param mu: Коэффициент турбулентной диффузии.
        :param sigma: Коэффициент поглощения.
        :return: Массив концентраций.
        """
        x_grid, y_grid = self.create_grid()
        dx = x_grid - x0
        dy = y_grid - y0
        dr = np.sqrt(dx**2 + dy**2)
        
        beta = sigma + (u**2 + v**2) / (4 * mu)
        x_val = np.sqrt(beta / mu) * dr
        
        uv_dot = u * dx + v * dy
        Phi = np.zeros_like(x_val)
        
        # Случай x_val < 2
        mask_lt2 = x_val < 2
        x_lt2 = x_val[mask_lt2]
        if x_lt2.size > 0:
            tilde_x1 = x_lt2 / 2
            t = x_lt2 / 3.75
            alpha = (1 + 3.5156229 * t**2 + 3.0899424 * t**4 +
                     1.2067492 * t**6 + 0.2659732 * t**8 +
                     0.0360768 * t**10 + 0.0045813 * t**12)
            ln_tilde_x1 = np.log(tilde_x1)
            tilde_k1 = (-alpha * ln_tilde_x1 - 0.5721566 +
                        0.4227842 * tilde_x1**2 + 0.23069756 * tilde_x1**4 +
                        0.0348589 * tilde_x1**6 + 0.00262698 * tilde_x1**8 +
                        0.0001075 * tilde_x1**10 + 0.000074 * tilde_x1**12)
            exp_factor = np.exp((u * dx[mask_lt2] + v * dy[mask_lt2]) / (2 * mu))
            Phi[mask_lt2] = (Q / (2 * np.pi * mu)) * tilde_k1 * exp_factor
        
        # Случай x_val >= 2
        mask_ge2 = ~mask_lt2
        x_ge2 = x_val[mask_ge2]
        if x_ge2.size > 0:
            tilde_x2 = 2 / x_ge2
            tilde_k2 = (1.25331414 - 0.07832358 * tilde_x2 +
                        0.02189568 * tilde_x2**2 - 0.01062446 * tilde_x2**3 +
                        0.00587872 * tilde_x2**4 - 0.0025154 * tilde_x2**5 +
                        0.000532 * tilde_x2**6)
            exp_factor_ge2 = np.exp((u * dx[mask_ge2] + v * dy[mask_ge2]) / (2 * mu) - x_ge2)
            Phi[mask_ge2] = (Q / (2 * x_ge2 * np.pi * mu)) * tilde_k2 * exp_factor_ge2
        
        return Phi