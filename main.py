from gaussian_plume_model import GaussianPlumeModel

# Основной скрипт
if __name__ == "__main__":
    # Константы
    SOURCE_EMISSION_RATE = 10  # Скорость выброса источника (кг/с)
    WIND_SPEED = 3  # Скорость ветра (м/с)
    WIND_DIRECTION = 135 # Направление ветра в градусах (Относительно оси X)
    RELEASE_HEIGHT = 2  # Высота выброса (м)
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
        wind_direction       = WIND_DIRECTION,
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
    plt = plume_model.plot(x_grid, y_grid, concentration, MIN_CONCENTRATION)
    plt.show()