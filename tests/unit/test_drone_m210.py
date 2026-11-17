"""Aparato coherente para el modelo energético homologado.

`GongEnergyModel` usa los parámetros de rotor de la Tabla I de Gong et al., que
son de un DJI M210 RTK V2: 4 rotores de 0,261 m de radio (hélices de 20,6"). El
`DroneModel` por defecto pesa 1,5 kg — un aparato de ese peso no lleva hélices de
20 pulgadas, así que la combinación describe algo físicamente imposible y sale
artificialmente eficiente.

`dji_m210()` cierra esa brecha: usa el modelo publicado sobre el aparato con el
que sus autores lo validaron, de modo que la validación empírica se transfiere.
"""

import math

from bioinspired.environment.drone_model import DroneModel
from bioinspired.environment.energy_model import GongEnergyModel

G = 9.81


def test_el_default_no_cambia() -> None:
    """Guarda de reproducibilidad: `paper_v5`/`paper_v6` usan estos valores.

    La fórmula legacy (`tsp_problem.py`) toma `hover_power_w` y `total_mass_kg`
    del dron, así que tocar los defaults invalidaría los datos de paper 1.
    """
    d = DroneModel()
    assert d.mass_kg == 1.5
    assert d.hover_power_w == 150.0
    assert d.battery_capacity_wh == 100.0


def test_m210_es_coherente_con_el_modelo_energetico() -> None:
    """`hover_power_w` debe salir del modelo, no estar hardcodeado.

    Es la incoherencia concreta que motiva este aparato: el default declara
    150 W mientras Gong predice 737,7 W para esa geometría de rotor.
    """
    m210 = DroneModel.dji_m210()
    predicho = GongEnergyModel().power_hover_w(m210.total_mass_kg * G)
    assert math.isclose(
        m210.hover_power_w, predicho, rel_tol=0.02
    ), f"hover_power_w={m210.hover_power_w} no coincide con el modelo ({predicho:.1f} W)"


def test_m210_tiene_las_specs_publicadas() -> None:
    m210 = DroneModel.dji_m210()
    assert 4.5 <= m210.mass_kg <= 5.0, "M210 RTK V2 pesa ~4,8 kg con baterías"
    # 2x TB55 de 174 Wh
    assert math.isclose(m210.battery_capacity_wh, 348.0, rel_tol=0.01)
    assert m210.max_velocity_mps > DroneModel().max_velocity_mps


def test_el_default_es_incoherente_y_por_eso_existe_este_aparato() -> None:
    """Documenta el problema que motiva el cambio, para que no se revierta.

    Si algún día el default se vuelve coherente, este test falla y hay que
    revisar si `dji_m210()` sigue haciendo falta.
    """
    d = DroneModel()
    predicho = GongEnergyModel().power_hover_w(d.total_mass_kg * G)
    assert not math.isclose(
        d.hover_power_w, predicho, rel_tol=0.02
    ), "el default dejó de ser incoherente: revisar si dji_m210() sigue justificándose"


def test_m210_declara_su_radio_fisico() -> None:
    """Envolvente ~0,9 m (diagonal 643 mm + helices de 17"): radio 0,45 m.

    Es el valor del que sale el `--offset` por defecto del exportador SDF
    (radio + margen). El default de la clase sigue en 0,0: la linea TSP planifica
    en espacio de configuracion y no infla nada por su cuenta.
    """
    assert DroneModel.dji_m210().body_radius_m == 0.45
    assert DroneModel().body_radius_m == 0.0
