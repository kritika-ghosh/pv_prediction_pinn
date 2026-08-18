import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

# ==========================================
# 1. LOAD & CLEAN WEATHER DATA
# ==========================================
file_path = "open-meteo-23.23N77.36E525m.csv"

# Extract metadata (lat, lon, elevation) from header
meta_df = pd.read_csv(file_path, nrows=1)
latitude = float(meta_df["latitude"].iloc[0])
longitude = float(meta_df["longitude"].iloc[0])
altitude = (
    float(meta_df["elevation"].iloc[0])
    if "elevation" in meta_df.columns
    else 525.0
)

# Load full CSV skipping metadata lines
df = pd.read_csv(file_path, skiprows=2)

# Filter out non-datetime header rows (e.g. daily summary blocks)
df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
df = df.dropna(subset=["time"]).copy()
df.set_index("time", inplace=True)

# Identify required columns dynamically
temp_col = [c for c in df.columns if "temperature_2m" in c][0]
cloud_col = [c for c in df.columns if "cloud_cover" in c and "low" not in c][0]
wind_col = [c for c in df.columns if "wind_speed_10m" in c][0]

# Cast numeric values
df["temp_air"] = pd.to_numeric(df[temp_col], errors="coerce")
df["cloud_cover"] = pd.to_numeric(df[cloud_col], errors="coerce")
df["wind_speed"] = (
    pd.to_numeric(df[wind_col], errors="coerce") / 3.6
)  # km/h to m/s

df.dropna(subset=["temp_air", "cloud_cover", "wind_speed"], inplace=True)

# ==========================================
# 2. IRRADIANCE DECOMPOSITION & POA
# ==========================================
location = Location(
    latitude=latitude,
    longitude=longitude,
    tz="UTC",
    altitude=altitude,
    name="Site",
)

# Solar Position
solar_pos = location.get_solarposition(times=df.index)

# Clear-sky baseline adjusted by cloud attenuation
clearsky = location.get_clearsky(times=df.index, model="ineichen")
cloud_attenuation = 1.0 - 0.75 * (df["cloud_cover"] / 100.0) ** 3.4
ghi = clearsky["ghi"] * cloud_attenuation

# Decompose GHI into DNI & DHI via Erbs model
dni_dhi = pvlib.irradiance.erbs(
    ghi=ghi,
    zenith=solar_pos["apparent_zenith"],
    datetime_or_doy=df.index.dayofyear,
)

df["ghi"] = ghi
df["dni"] = pd.Series(dni_dhi["dni"], index=df.index).fillna(0.0)
df["dhi"] = pd.Series(dni_dhi["dhi"], index=df.index).fillna(0.0)

# Plane of Array (POA) Irradiance (Surface tilt = latitude, Azimuth = 180° South)
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt=latitude,
    surface_azimuth=180.0,
    solar_zenith=solar_pos["apparent_zenith"],
    solar_azimuth=solar_pos["azimuth"],
    dni=df["dni"],
    ghi=df["ghi"],
    dhi=df["dhi"],
)
df["poa_global"] = pd.Series(
    poa_irradiance["poa_global"], index=df.index
).fillna(0.0)

# ==========================================
# 3. THERMAL DYNAMICS (Cell Temperature)
# ==========================================
# SAPM open rack thermal model
temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_polymer"
]
df["temp_cell"] = pvlib.temperature.sapm_cell(
    poa_global=df["poa_global"],
    temp_air=df["temp_air"],
    wind_speed=df["wind_speed"],
    a=temp_params["a"],
    b=temp_params["b"],
    deltaT=temp_params["deltaT"],
)

# ==========================================
# 4. ELECTRICAL SINGLE-DIODE MODEL (MPPT)
# ==========================================
# Canadian Solar CS6P-250P reference parameters
module_params = {
    "alpha_sc": 0.0048,  # A/°C
    "a_ref": 1.6234,  # Diode ideality modifier (V)
    "I_L_ref": 8.87,  # Light current at STC (A)
    "I_o_ref": 2.1e-9,  # Saturation current at STC (A)
    "R_s": 0.35,  # Series resistance (Ohm)
    "R_sh_ref": 412.0,  # Shunt resistance at STC (Ohm)
    "EgRef": 1.121,  # Bandgap energy (eV)
    "dEgdT": -0.0002677,
}

# 5-parameter De Soto translation for operational POA and T_cell
photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
    pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=df["poa_global"],
        temp_cell=df["temp_cell"],
        alpha_sc=module_params["alpha_sc"],
        a_ref=module_params["a_ref"],
        I_L_ref=module_params["I_L_ref"],
        I_o_ref=module_params["I_o_ref"],
        R_sh_ref=module_params["R_sh_ref"],
        R_s=module_params["R_s"],
        EgRef=module_params["EgRef"],
        dEgdT=module_params["dEgdT"],
    )
)

# Standard singlediode solver (returns dataframe of i_sc, v_oc, i_mp, v_mp, p_mp)
sd_results = pvlib.pvsystem.singlediode(
    photocurrent=photocurrent,
    saturation_current=saturation_current,
    resistance_series=resistance_series,
    resistance_shunt=resistance_shunt,
    nNsVth=nNsVth,
    method="lambertw",
)

# Mask nighttime/low irradiance entries to zero
is_day = df["poa_global"] > 10.0
df["I_mp"] = np.where(is_day, sd_results["i_mp"], 0.0)
df["V_mp"] = np.where(is_day, sd_results["v_mp"], 0.0)
df["P_mp"] = np.where(is_day, sd_results["p_mp"], 0.0)

# ==========================================
# 5. ARRHENIUS AGING DYNAMICS & EXPORT
# ==========================================
k_B = 8.617333262145e-5  # eV/K
E_a = 0.85  # eV[cite: 1]
T_cell_k = df["temp_cell"] + 273.15
df["thermal_stress_rate"] = np.exp(-E_a / (k_B * T_cell_k))

output_cols = [
    "poa_global",
    "temp_air",
    "wind_speed",
    "temp_cell",
    "I_mp",
    "V_mp",
    "P_mp",
    "thermal_stress_rate",
]
pinn_dataset = df[output_cols].copy()
pinn_dataset.to_csv("pinn_training_dataset.csv")

print(f"Generated PINN dataset with {len(pinn_dataset)} records.")
print(pinn_dataset.head())