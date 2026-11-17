"""Modelo cinemático del UAV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DroneModel:
    """Restricciones cinemáticas y energéticas del UAV.

    Se inyecta como parámetro a los algoritmos. Nunca se instancia dentro de
    un optimizador. Default values son configurables vía Hydra.

    Attributes:
        mass_kg: masa en kg.
        max_velocity_mps: velocidad máxima en m/s.
        cruise_velocity_mps: velocidad de crucero (≤ max_velocity_mps).
        max_acceleration_mps2: aceleración máxima en m/s².
        max_climb_angle_deg: ángulo máximo de ascenso/descenso (0-90).
        max_turn_angle_deg: ángulo máximo de giro entre segmentos (0-180).
        min_altitude_m: altitud mínima permitida.
        max_altitude_m: altitud máxima permitida.
        battery_capacity_wh: capacidad de batería en Wh.
        hover_power_w: potencia consumida en hover (W).
        enable_constraints: si False, las restricciones cinemáticas se ignoran
            (se usa para Config A/C del benchmark Fase 1).
    """

    mass_kg: float = 1.5
    max_velocity_mps: float = 10.0
    cruise_velocity_mps: float = 5.0
    max_acceleration_mps2: float = 3.0
    max_climb_angle_deg: float = 30.0
    max_turn_angle_deg: float = 45.0
    min_altitude_m: float = 2.0
    max_altitude_m: float = 120.0
    battery_capacity_wh: float = 100.0
    hover_power_w: float = 150.0
    enable_constraints: bool = True
    body_radius_m: float = 0.0
    payload_kg: float = 0.0
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0  # compass: 0=North(+Y), 90=East(+X), 180=South, 270=West
    max_range_m: float = 0.0  # max distance from start; 0 = unlimited
    endurance_min: float = 0.0  # max flight time in minutes; 0 = unlimited

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg debe ser positivo, recibido {self.mass_kg}")
        if self.max_velocity_mps <= 0.0:
            raise ValueError(
                f"max_velocity_mps debe ser positivo, recibido {self.max_velocity_mps}"
            )
        if self.cruise_velocity_mps > self.max_velocity_mps:
            raise ValueError("cruise_velocity_mps no puede exceder max_velocity_mps")
        if self.max_acceleration_mps2 <= 0.0:
            raise ValueError("max_acceleration_mps2 debe ser positivo")
        if not (0.0 < self.max_climb_angle_deg < 90.0):
            raise ValueError("max_climb_angle_deg fuera de rango (0, 90)")
        if not (0.0 < self.max_turn_angle_deg < 180.0):
            raise ValueError("max_turn_angle_deg fuera de rango (0, 180)")
        if self.min_altitude_m >= self.max_altitude_m:
            raise ValueError("min_altitude_m debe ser menor que max_altitude_m")
        if self.battery_capacity_wh <= 0.0:
            raise ValueError("battery_capacity_wh debe ser positivo")
        if self.hover_power_w <= 0.0:
            raise ValueError("hover_power_w debe ser positivo")
        if self.body_radius_m < 0.0:
            raise ValueError(f"body_radius_m debe ser >= 0, recibido {self.body_radius_m}")
        if self.payload_kg < 0.0:
            raise ValueError(f"payload_kg debe ser >= 0, recibido {self.payload_kg}")
        if self.wind_speed_mps < 0.0:
            raise ValueError(f"wind_speed_mps debe ser >= 0, recibido {self.wind_speed_mps}")
        if self.max_range_m < 0.0:
            raise ValueError(f"max_range_m debe ser >= 0, recibido {self.max_range_m}")
        if self.endurance_min < 0.0:
            raise ValueError(f"endurance_min debe ser >= 0, recibido {self.endurance_min}")

    @property
    def total_mass_kg(self) -> float:
        """Masa total: drone + payload."""
        return self.mass_kg + self.payload_kg

    @classmethod
    def dji_m210(cls) -> DroneModel:
        """DJI M210 RTK V2 — el aparato con el que Gong et al. validaron su modelo.

        `GongEnergyModel` usa los parámetros de rotor de su Tabla I: 4 rotores de
        0,261 m de radio (hélices de 20,6"). El `DroneModel` por defecto pesa
        1,5 kg, y un aparato de ese peso no lleva hélices de 20 pulgadas — la
        combinación describe algo físicamente imposible y, al darle demasiada
        área de disco, sale artificialmente eficiente.

        Usar este aparato hace que el modelo publicado se aplique sobre la
        instancia que sus autores validaron experimentalmente, de modo que esa
        validación se transfiere.

        **No se cambian los defaults de la clase**: la fórmula energética legacy
        toma `hover_power_w` y `total_mass_kg` de aquí, y es la que produjo
        `results/fase1/paper_v5`. Tocarlos invalidaría los datos de paper 1.

        Specs públicas: ~4,8 kg con baterías, 2x TB55 de 174 Wh, velocidad máxima
        22,3 m/s, ascenso 5 m/s, descenso 3 m/s.

        **`hover_power_w` sólo alimenta la fórmula legacy.** Bajo cualquier
        variante de Gong el hover lo calcula el modelo, no este campo. Los
        737,7 W que declara salen de la Tabla I sin corregir; el modelo
        congelado de Fase 2 en adelante (`gong_coherent`) predice 399,6 W
        aerodinámicos, o 696 W de batería con `POWERTRAIN_EFFICIENCY_M210`. El
        campo se deja como está porque cambiarlo movería silenciosamente
        cualquier corrida `legacy` con este aparato.
        """
        # Coherente con el modelo: P_hover sale de Gong para esta masa, no de un
        # valor hardcodeado (el default declara 150 W y el modelo predice 737,7).
        return cls(
            mass_kg=4.8,
            payload_kg=0.0,
            max_velocity_mps=22.3,
            cruise_velocity_mps=5.0,  # decisión experimental: vuelo lento de inspección
            max_acceleration_mps2=3.0,
            battery_capacity_wh=348.0,  # 2 x TB55
            hover_power_w=737.7,
            # Envolvente física ~0,9 m: diagonal entre ejes de 643 mm más las
            # hélices de 17". Sólo lo consumen `NumpyBackend` (línea PP) y el
            # offset por defecto del exportador SDF; la línea TSP planifica en
            # espacio de configuración y no infla nada con este valor.
            body_radius_m=0.45,
        )

    @property
    def wind_vector(self) -> np.ndarray:
        """Viento en coordenadas XYZ (compass → cartesiano)."""
        rad = np.radians(self.wind_direction_deg)
        return np.array(
            [
                self.wind_speed_mps * np.sin(rad),  # Este(+X) cuando dir=90
                self.wind_speed_mps * np.cos(rad),  # Norte(+Y) cuando dir=0
                0.0,
            ]
        )
