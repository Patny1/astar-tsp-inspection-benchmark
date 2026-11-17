"""Modelos de consumo energético del UAV."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from bioinspired.core.trajectory import Trajectory
from bioinspired.environment.drone_model import DroneModel


class BaseEnergyModel(ABC):
    """Interfaz para modelos de energía. Inyectada en PathPlanningProblem."""

    @abstractmethod
    def compute(self, trajectory: Trajectory, drone: DroneModel) -> float:
        """Consumo energético total de la trayectoria, en unidades del modelo."""
        ...


class NullEnergyModel(BaseEnergyModel):
    """Baseline sin física: la 'energía' es la distancia euclidiana 3D total.

    Sirve como referencia neutral para el benchmark Fase 1 (Config A/B).
    """

    def compute(self, trajectory: Trajectory, drone: DroneModel) -> float:
        diffs_arr = np.diff(trajectory.waypoints_arr, axis=0)
        return float(np.linalg.norm(diffs_arr, axis=1).sum())


class PhysicalEnergyModel(BaseEnergyModel):
    """Consumo energético basado en física simplificada (proxy heurístico).

    Aproximación trabajo-energía:
        E_segment = m·g·max(0, Δh) + (0.5·m·v²)·Δt + hover_power·Δt

    NOTA SOBRE LA SEMÁNTICA: el segundo término ``0.5·m·v²·Δt`` NO es energía
    cinética en el sentido estricto (que sería sólo ``0.5·m·v²`` única). Lo
    usamos como **proxy del coste sostenido de mantener la velocidad de
    crucero** durante un segmento — equivalente a tratar ``0.5·m·v²`` como
    una potencia constante en watts. Es conservador (sobrestima ligeramente
    el consumo, pero monótonamente con distancia/velocidad). Para análisis
    físicos precisos, reemplazar por ``drag_coefficient·v³·Δt`` (potencia
    aerodinámica real).

    Ignora regenerative braking en descensos (subir cuesta, bajar es 0).
    Asume velocidad constante = ``drone.cruise_velocity_mps``.
    Unidades: Joules. Convertir a Wh dividiendo por 3600.
    """

    GRAVITY_MPS2: ClassVar[float] = 9.81

    def compute(self, trajectory: Trajectory, drone: DroneModel) -> float:
        diffs_arr = np.diff(trajectory.waypoints_arr, axis=0)
        segment_lengths = np.linalg.norm(diffs_arr, axis=1)
        delta_z = diffs_arr[:, 2]
        v = drone.cruise_velocity_mps
        dt = segment_lengths / (v + 1e-12)
        m = drone.total_mass_kg  # includes payload

        e_pot = m * self.GRAVITY_MPS2 * np.maximum(0.0, delta_z)
        e_hov = drone.hover_power_w * dt

        # Wind: headwind increases effective airspeed needed to hold cruise ground speed.
        # Tailwind capped at zero benefit (conservative).
        if drone.wind_speed_mps > 0.0:
            seg_units = diffs_arr / (segment_lengths[:, np.newaxis] + 1e-12)
            headwind = np.clip(seg_units @ drone.wind_vector, 0.0, None)
            v_eff = v + headwind
        else:
            v_eff = np.full_like(segment_lengths, v)

        e_kin = 0.5 * m * v_eff**2 * dt
        return float((e_pot + e_kin + e_hov).sum())


@dataclass(frozen=True)
class RotorParameters:
    """Parámetros aerodinámicos del multirrotor (Gong et al., 2023, Tabla I).

    Valores por defecto = la Tabla I del paper tal como está publicada (columna
    "Simulation value", W = 20 N). El paper valida el modelo con un DJI M210 RTK V2
    (cuadrirotor) sobre >11 000 muestras de vuelo, pero W = 20 N no es el peso de
    ese aparato (~47 N): para el M210 usar ``dji_m210``.

    Referencia:
        Gong, H., Huang, B., Jia, B., & Dai, H. (2023). Modeling Power
        Consumptions for Multirotor UAVs. *IEEE Transactions on Aerospace and
        Electronic Systems*, 59(6), 7409-7422. doi:10.1109/TAES.2023.3288846
        (preprint: arXiv:2209.04128). Verificado contra la versión publicada:
        ecuaciones (1)-(19), su numeración y la Tabla I son idénticas al
        preprint; el publicado escribe V_h/V_v donde este módulo usa V∥/V⊥.
        Ver ``results/fase0/modelo_energetico/gong_preprint_vs_publicado.md``.

    Attributes:
        profile_drag_coefficient: δ, coeficiente de arrastre de perfil.
        air_density_kgpm3: rho, densidad del aire.
        rotor_solidity: s, solidez del rotor.
        rotor_disc_area_m2: A, área del disco de un rotor.
        induced_power_correction: k, corrección incremental de potencia inducida.
        thrust_coefficient: C_T, coeficiente de empuje.
        fuselage_area_horizontal_m2: SFP∥, área plana equivalente en horizontal.
        fuselage_area_vertical_m2: SFP⊥, área plana equivalente en vertical.
        n_rotors: n, número de rotores (el paper deriva sólo para n par).
        max_climb_rate_mps: tope de velocidad de ascenso validado experimentalmente.
        max_descent_rate_mps: tope de velocidad de descenso validado. **No subir
            este valor sin releer la Ec. (14)**: pasado cierto punto el
            coeficiente ``W/2 - n/4·SFP⊥·rho·V²`` cambia de signo, y más arriba
            el radicando se vuelve negativo — el modelo sale de su dominio de
            validez. Ambos umbrales dependen del **peso**, así que los valores
            históricos de este docstring (~4,8 y ~7,2 m/s) corresponden a los
            W = 20 N de la Tabla I, no al aparato de 4,8 kg que usa el repo:

            =========================  ==========  ==============
            instancia                  cambio de   radicando < 0
                                       signo
            =========================  ==========  ==============
            Tabla I, W = 20 N            4,8 m/s      7,2 m/s
            m210 (47,1 N), A = 0,214     7,3 m/s     11,1 m/s
            m210 (47,1 N), A = 0,146     7,3 m/s      9,4 m/s
            =========================  ==========  ==============

            En los tres casos el tope de 3 m/s queda holgadamente adentro.
    """

    profile_drag_coefficient: float = 0.011
    air_density_kgpm3: float = 1.168
    rotor_solidity: float = 0.045
    rotor_disc_area_m2: float = 0.214
    induced_power_correction: float = 0.11
    thrust_coefficient: float = 0.001195
    fuselage_area_horizontal_m2: float = 0.009
    fuselage_area_vertical_m2: float = 0.377
    n_rotors: int = 4
    max_climb_rate_mps: float = 5.0
    max_descent_rate_mps: float = 3.0

    def __post_init__(self) -> None:
        if self.n_rotors < 2 or self.n_rotors % 2 != 0:
            raise ValueError(f"n_rotors debe ser par y >= 2, recibido {self.n_rotors}")
        for field_name in (
            "profile_drag_coefficient",
            "air_density_kgpm3",
            "rotor_solidity",
            "rotor_disc_area_m2",
            "thrust_coefficient",
            "max_climb_rate_mps",
            "max_descent_rate_mps",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} debe ser positivo")
        if self.induced_power_correction < 0.0:
            raise ValueError("induced_power_correction debe ser >= 0")
        if self.fuselage_area_horizontal_m2 < 0.0 or self.fuselage_area_vertical_m2 < 0.0:
            raise ValueError("las áreas planas equivalentes deben ser >= 0")

    @classmethod
    def dji_m210(
        cls,
        weight_n: float = 4.8 * 9.81,
        tip_speed_hover_mps: float = 130.0,
    ) -> RotorParameters:
        """Tabla I con el **disco y el C_T corregidos** al rotor real del M210.

        Los defaults de la clase son la Tabla I del paper tal como está
        publicada, y ahí `A = 0,214 m²` corresponde a un rotor de R = 0,26 m
        —hélice de ~20,5"— mientras que el M200-series monta **1760S, de 17"**.
        Aplicados al aparato de 4,8 kg, esos valores implican un rotor
        imposible: la relación ``Omega = sqrt(T/(C_T·rho·A·R²))`` da 761 rad/s,
        es decir 7.264 RPM y una punta de pala de 198 m/s (Mach 0,58) sobre una
        hélice de 20 pulgadas. La consecuencia medible es que la potencia de
        perfil sale 1,91 veces la inducida —al revés de lo que da un
        multirrotor— y la figura de mérito implícita cae a 0,31.

        Esta parametrización mantiene **todas** las ecuaciones y los seis
        parámetros restantes de la Tabla I, y sólo corrige los dos que
        describen la geometría del rotor:

        - ``A`` = área de la 1760S (R = 0,2159 m) -> 0,1464 m².
        - ``C_T`` = ``W/(n·rho·A·v_punta²)``, el valor que hace que la punta de
          pala en hover valga ``tip_speed_hover_mps``.

        Resultado con los defaults: C_T = 0,004073, Omega = 602 rad/s
        (5.750 RPM), P_hover = 399,6 W **aerodinámicos**, P_perfil/P_inducida =
        0,30 y figura de mérito 0,69 — ya un multirrotor plausible. El nivel de
        batería sale de ``GongEnergyModel(powertrain_efficiency=...)``; ver
        ``POWERTRAIN_EFFICIENCY_M210``.

        La elección de la punta de pala es una hipótesis de ingeniería, no un
        dato del paper, pero es de segundo orden: entre 110 y 150 m/s el ratio
        subir/crucero que consume el optimizador se mueve de 3,11 a 2,87
        (contra 2,51 con la Tabla I sin corregir).

        Args:
            weight_n: peso del aparato, el que fija la condición de hover.
            tip_speed_hover_mps: punta de pala en hover a la que anclar ``C_T``.
        """
        if weight_n <= 0.0:
            raise ValueError(f"weight_n debe ser positivo, recibido {weight_n}")
        if tip_speed_hover_mps <= 0.0:
            raise ValueError(
                f"tip_speed_hover_mps debe ser positivo, recibido {tip_speed_hover_mps}"
            )
        base = cls()
        radius_m = 17.0 * 0.0254 / 2.0  # hélice 1760S: 17 pulgadas de diámetro
        area_m2 = math.pi * radius_m**2
        # De T_r = C_T·rho·A·(Omega·R)² con Omega·R = v_punta: no hace falta R.
        thrust_coefficient = weight_n / (
            base.n_rotors * base.air_density_kgpm3 * area_m2 * tip_speed_hover_mps**2
        )
        return replace(base, rotor_disc_area_m2=area_m2, thrust_coefficient=thrust_coefficient)


def leg_time_s(
    horizontal_m: float,
    climb_m: float,
    descent_m: float,
    v_cruise_mps: float,
    v_climb_mps: float,
    v_descent_mps: float,
) -> float:
    """Tiempo de vuelo (s) de un tramo en piernas: cada una a velocidad constante.

    ``horizontal / v_cruise + ascenso / v_climb + descenso / v_descent``. Es la
    misma descomposicion que ``GongEnergyModel.leg_energy_j`` usa por dentro
    (potencia x tiempo por pierna) y descarta; aqui se expone como magnitud
    propia para que el tiempo pueda ser un objetivo (Fase 3 en adelante). No
    modela hover en los targets ni aceleraciones: las matrices asumen crucero
    constante entre targets, igual que la energia.
    """
    t = 0.0
    if horizontal_m > 0.0 and v_cruise_mps > 0.0:
        t += horizontal_m / v_cruise_mps
    if climb_m > 0.0 and v_climb_mps > 0.0:
        t += climb_m / v_climb_mps
    if descent_m > 0.0 and v_descent_mps > 0.0:
        t += descent_m / v_descent_mps
    return t


POWERTRAIN_EFFICIENCY_M210: float = 0.574
"""Factor que lleva la potencia aerodinámica del M210 a potencia de batería.

Las ecuaciones de Gong entregan potencia **aerodinámica**. Con el rotor
corregido (`RotorParameters.dji_m210()`) el hover sale 399,6 W, que a batería
darían 87 min de autonomía — el M210 RTK V2 declara ~30 min con sus 2x TB55
(348 Wh). Este factor cierra esa brecha: 399,6 / 0,574 = 696 W -> 30,0 min.

**Es un factor empírico agrupado**, calibrado contra la autonomía publicada del
aparato: absorbe el rendimiento de motores y ESC, la aviónica, y el optimismo
de la figura de mérito del modelo (0,69, alta para hélices de esta carga). NO
es un rendimiento de motor y no debe presentarse como tal.

Es un **escalar**: multiplica toda la matriz de energía por igual, así que no
altera el orden de ningún tour — sólo devuelve al nivel correcto los valores
absolutos (Wh, % de batería, autonomía). Por eso el default de
`GongEnergyModel` es 1,0 —el paper tal cual— y este factor se pasa
explícitamente donde el nivel importa. La comparación contra la potencia de eje
de Gazebo va con eficiencia 1,0; las afirmaciones de batería, con ésta.
"""


class GongEnergyModel(BaseEnergyModel):
    """Modelo de potencia publicado para multirrotores (Gong et al., 2023).

    Implementa las cuatro expresiones en forma cerrada del paper:

    - ``power_hover_w``    → Ec. (5),  hover.
    - ``power_forward_w``  → Ec. (8),  vuelo horizontal a velocidad V∥.
    - ``power_vertical_w`` → Ec. (13) ascenso / Ec. (14) descenso, a |V⊥|.
    - ``power_total_w``    → Ec. (19), modelo genérico 3-D.

    Reemplaza el coeficiente de ascenso puesto a mano por un modelo citable.
    Tres consecuencias cualitativas respecto del proxy ``P_hover·d/v``:

    1. **Descender no es gratis.** El descenso exige empuje (Ec. 10) y consume
       una potencia del mismo orden que el ascenso dentro del rango validado,
       así que ``max(0, Δh)`` subestima gravemente el coste de bajar.
    2. **La potencia horizontal no es monótona.** La potencia inducida cae al
       aumentar V∥, de modo que ``P_forward`` tiene forma de U con un mínimo
       (≈ 8-11 m/s con los parámetros del M210) y llega a estar por debajo del
       hover — un proxy proporcional a la distancia no puede representarlo.
    3. **La asimetría ascenso/descenso la fija la velocidad vertical**, no una
       constante: se anula cuando V⊥ → 0 y crece con V⊥.

    Contabilidad de la trayectoria (``compute``): las piernas horizontal y
    vertical se integran por separado — la horizontal a velocidad de crucero,
    el ascenso y el descenso acumulados a su respectiva tasa vertical. Esta
    descomposición es la misma que usa el paper en sus experimentos 1-3, y
    evita dos artefactos de aplicar la Ec. (19) segmento a segmento sobre una
    ruta de A*: (a) las escaleras del vóxel introducen V⊥ espurias, y (b) la
    Ec. (18) es discontinua en V⊥ = 0 — el límite lateral vale
    ``(W/2)·√(2W/(n·rho·A))`` W, así que un Δz arbitrariamente pequeño cobraría
    igual una penalización finita. Para evaluar la Ec. (19) tal cual está
    publicada, usar ``power_total_w`` directamente.

    La tasa vertical se deriva de ``DroneModel``: ``cruise_velocity_mps ·
    sin(max_climb_angle_deg)``, topeada por los límites validados en
    ``RotorParameters``. Los overrides explícitos tienen prioridad.

    Unidades: Joules. Convertir a Wh dividiendo por 3600.
    """

    GRAVITY_MPS2: ClassVar[float] = 9.81

    def __init__(
        self,
        params: RotorParameters | None = None,
        climb_rate_mps: float | None = None,
        descent_rate_mps: float | None = None,
        continuous_at_hover: bool = False,
        powertrain_efficiency: float = 1.0,
    ) -> None:
        """Inicializa el modelo.

        Args:
            params: parámetros del rotor. Por defecto los del DJI M210 del paper.
            climb_rate_mps: tasa de ascenso a asumir. ``None`` → derivar del
                ``DroneModel`` recibido en ``compute``.
            descent_rate_mps: ídem para descenso.
            continuous_at_hover: ver ``power_vertical_w``. ``False`` (por
                defecto) reproduce las Ecs. (13)/(14) tal como están
                publicadas; ``True`` aplica la corrección de continuidad.
            powertrain_efficiency: divisor aplicado a toda potencia devuelta.
                ``1.0`` (default) = potencia **aerodinámica**, el paper tal
                cual — es lo que se compara contra la potencia de eje de un
                simulador. Un valor < 1 lleva la salida a potencia de
                **batería**; ver ``POWERTRAIN_EFFICIENCY_M210``. Al ser un
                escalar no altera el orden de ningún tour.
        """
        if not 0.0 < powertrain_efficiency <= 1.0:
            raise ValueError(
                f"powertrain_efficiency debe estar en (0, 1], recibido {powertrain_efficiency}"
            )
        self.powertrain_efficiency = powertrain_efficiency
        self.params = params if params is not None else RotorParameters()
        if climb_rate_mps is not None and climb_rate_mps <= 0.0:
            raise ValueError("climb_rate_mps debe ser positivo")
        if descent_rate_mps is not None and descent_rate_mps <= 0.0:
            raise ValueError("descent_rate_mps debe ser positivo")
        self.climb_rate_mps = climb_rate_mps
        self.descent_rate_mps = descent_rate_mps
        self.continuous_at_hover = continuous_at_hover

    # ------------------------------------------------------------------
    # Potencias en forma cerrada
    # ------------------------------------------------------------------

    def _induced_velocity_mps(self, weight_n: float) -> float:
        """v0 por rotor en hover: ``√(W_r / (2·rho·A))`` con ``W_r = W/n``."""
        p = self.params
        return math.sqrt(weight_n / (p.n_rotors * 2.0 * p.air_density_kgpm3 * p.rotor_disc_area_m2))

    def _blade_power_w(self, weight_n: float) -> float:
        """P_bl de la Ec. (5): potencia de perfil de pala en hover."""
        p = self.params
        return float(
            (p.profile_drag_coefficient * p.rotor_solidity / 8.0)
            * weight_n**1.5
            / math.sqrt(p.n_rotors * p.air_density_kgpm3 * p.rotor_disc_area_m2)
            * p.thrust_coefficient**-1.5
        )

    def _induced_power_w(self, weight_n: float) -> float:
        """P_in de la Ec. (5): potencia inducida en hover."""
        p = self.params
        return float(
            (1.0 + p.induced_power_correction)
            * weight_n**1.5
            / math.sqrt(2.0 * p.n_rotors * p.air_density_kgpm3 * p.rotor_disc_area_m2)
        )

    def _hover_aero_w(self, weight_n: float) -> float:
        """Ec. (5) **sin** aplicar ``powertrain_efficiency``.

        Los métodos públicos aplican el factor una única vez al final; los
        internos trabajan siempre en potencia aerodinámica para que no se
        aplique dos veces al componer términos.
        """
        return self._blade_power_w(weight_n) + self._induced_power_w(weight_n)

    def power_hover_w(self, weight_n: float) -> float:
        """Ec. (5): potencia en hover ``P_mh = P_bl + P_in``."""
        return self._hover_aero_w(weight_n) / self.powertrain_efficiency

    def power_forward_w(self, velocity_mps: float, weight_n: float) -> float:
        """Ec. (8): potencia en vuelo horizontal a velocidad ``V∥``."""
        p = self.params
        v = abs(float(velocity_mps))
        v0 = self._induced_velocity_mps(weight_n)
        blade = self._blade_power_w(weight_n) + (3.0 / 8.0) * p.profile_drag_coefficient * (
            p.rotor_solidity
            * v**2
            * math.sqrt(
                weight_n
                * p.n_rotors
                * p.air_density_kgpm3
                * p.rotor_disc_area_m2
                / p.thrust_coefficient
            )
        )
        induced = self._induced_power_w(weight_n) * math.sqrt(
            math.sqrt(1.0 + v**4 / (4.0 * v0**4)) - v**2 / (2.0 * v0**2)
        )
        parasite = (p.n_rotors / 2.0) * p.fuselage_area_horizontal_m2 * p.air_density_kgpm3 * v**3
        return (blade + induced + parasite) / self.powertrain_efficiency

    def power_vertical_w(self, velocity_mps: float, weight_n: float) -> float:
        """Ec. (13)/(14): potencia en vuelo vertical.

        **Las ecuaciones publicadas no convergen al hover cuando V⊥ → 0.** La
        Ec. (3), heredada de Yang et al. [27], suma ``P1`` (que ya contiene la
        potencia inducida de hover ``P_in``) al término de empuje
        ``(T/2)·√(V² + 2T/(rho·A))``, cuyo límite en ``V = 0`` es exactamente
        ``T·√(T/(2·rho·A))``, es decir ``P_in/(1+k)``. La potencia inducida queda
        así contada dos veces y el modelo vertical arranca ``(W/2)·√(2W/(n·rho·A))``
        W por encima del hover (≈ 63 W con los parámetros de la Tabla I y
        W = 20 N). Consecuencia práctica: dentro del rango de descenso validado
        (≤ 3 m/s) la Ec. (14) predice **más** potencia que el hover, y un Δz
        arbitrariamente pequeño paga igual el escalón completo.

        Con ``continuous_at_hover=True`` el término de hover duplicado se
        elimina — se conserva ``P_bl`` y se aplica ``(1+k)`` al término de
        empuje — de modo que el modelo vertical converge exactamente a la
        Ec. (5) cuando ``V⊥ → 0``. El comportamiento cualitativo (ascenso más
        caro que descenso, brecha creciente con V⊥) se mantiene en ambos casos.

        Args:
            velocity_mps: velocidad vertical con signo. ``> 0`` ascenso (Ec. 13),
                ``< 0`` descenso (Ec. 14), ``0`` devuelve el hover.
            weight_n: peso del UAV en Newton.

        Raises:
            ValueError: si la velocidad excede el dominio de validez del modelo.
        """
        p = self.params
        if velocity_mps == 0.0:
            return self.power_hover_w(weight_n)
        sign = 1.0 if velocity_mps > 0.0 else -1.0
        v = abs(float(velocity_mps))
        limit = p.max_climb_rate_mps if sign > 0 else p.max_descent_rate_mps
        if v > limit:
            raise ValueError(
                f"velocidad vertical {v} m/s fuera del dominio validado (<= {limit} m/s)"
            )
        rho, area, n = p.air_density_kgpm3, p.rotor_disc_area_m2, p.n_rotors
        sfp = p.fuselage_area_vertical_m2
        radicand = (1.0 + sign * sfp / area) * v**2 + 2.0 * weight_n / (n * rho * area)
        if radicand < 0.0:  # pragma: no cover - bloqueado por el tope de velocidad
            raise ValueError("Ec. (14) fuera de dominio: radicando negativo")
        thrust_term = weight_n / 2.0 + sign * (n / 4.0) * sfp * rho * v**2

        if self.continuous_at_hover:
            baseline = self._blade_power_w(weight_n)
            thrust_term *= 1.0 + p.induced_power_correction
        else:
            baseline = self._hover_aero_w(weight_n)

        return (
            baseline
            + 0.5 * weight_n * v
            + sign * (n / 4.0) * sfp * rho * v**3
            + thrust_term * math.sqrt(radicand)
        ) / self.powertrain_efficiency

    def power_total_w(
        self, v_horizontal_mps: float, v_vertical_mps: float, weight_n: float
    ) -> float:
        """Ec. (19): potencia genérica 3-D ``P_mh + ΔP∥(V∥) + ΔP⊥(V⊥)``.

        Discontinua en ``V⊥ = 0`` por construcción — ver la nota de clase.
        """
        hover = self.power_hover_w(weight_n)
        delta_par = self.power_forward_w(v_horizontal_mps, weight_n) - hover
        delta_perp = (
            0.0
            if v_vertical_mps == 0.0
            else self.power_vertical_w(v_vertical_mps, weight_n) - hover
        )
        return hover + delta_par + delta_perp

    # ------------------------------------------------------------------
    # Integración sobre la trayectoria
    # ------------------------------------------------------------------

    def vertical_rates_mps(self, drone: DroneModel) -> tuple[float, float]:
        """Tasas (ascenso, descenso) a asumir para este dron, ya topeadas."""
        p = self.params
        nominal = drone.cruise_velocity_mps * math.sin(math.radians(drone.max_climb_angle_deg))
        climb = self.climb_rate_mps if self.climb_rate_mps is not None else nominal
        descent = self.descent_rate_mps if self.descent_rate_mps is not None else nominal
        return min(climb, p.max_climb_rate_mps), min(descent, p.max_descent_rate_mps)

    def leg_time_s(
        self,
        horizontal_m: float,
        climb_m: float,
        descent_m: float,
        drone: DroneModel,
    ) -> float:
        """Tiempo de vuelo (s) de un tramo dado en las mismas piernas que ``leg_energy_j``.

        Es exactamente el tiempo por el que ``leg_energy_j`` multiplica cada
        potencia: horizontal a crucero, ascenso y descenso a las tasas de
        ``vertical_rates_mps``. Objetivo de Fase 3 en adelante. Nota:
        ``metrics.path.kinematic_violations`` estima el tiempo como
        ``distancia 3D / crucero`` (una sola velocidad); es otra convencion y se
        deja como esta.
        """
        v_climb, v_descent = self.vertical_rates_mps(drone)
        return leg_time_s(
            horizontal_m, climb_m, descent_m, drone.cruise_velocity_mps, v_climb, v_descent
        )

    def leg_energy_j(
        self,
        horizontal_m: float,
        climb_m: float,
        descent_m: float,
        drone: DroneModel,
    ) -> float:
        """Energía (J) de un tramo dado en piernas horizontal / ascenso / descenso.

        Args:
            horizontal_m: distancia horizontal recorrida (>= 0).
            climb_m: ascenso acumulado, ``Σ max(0, Δz)`` (>= 0).
            descent_m: descenso acumulado, ``Σ max(0, -Δz)`` (>= 0).
            drone: dron, del que se toman peso y velocidades.
        """
        weight_n = drone.total_mass_kg * self.GRAVITY_MPS2
        v_cruise = drone.cruise_velocity_mps
        v_climb, v_descent = self.vertical_rates_mps(drone)

        energy = 0.0
        if horizontal_m > 0.0 and v_cruise > 0.0:
            energy += self.power_forward_w(v_cruise, weight_n) * horizontal_m / v_cruise
        if climb_m > 0.0:
            energy += self.power_vertical_w(v_climb, weight_n) * climb_m / v_climb
        if descent_m > 0.0:
            energy += self.power_vertical_w(-v_descent, weight_n) * descent_m / v_descent
        return energy

    def compute(self, trajectory: Trajectory, drone: DroneModel) -> float:
        """Energía total (J) de la trayectoria."""
        diffs_arr = np.diff(trajectory.waypoints_arr, axis=0)
        if diffs_arr.size == 0:
            return 0.0
        delta_z_arr = diffs_arr[:, 2]
        horizontal_m = float(np.linalg.norm(diffs_arr[:, :2], axis=1).sum())
        climb_m = float(np.maximum(delta_z_arr, 0.0).sum())
        descent_m = float(np.maximum(-delta_z_arr, 0.0).sum())
        return self.leg_energy_j(horizontal_m, climb_m, descent_m, drone)
