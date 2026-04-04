import random
import pandas as pd

# -----------------------------
# 1) TEXTURE ENCODING
# -----------------------------
TEXTURE_MAP = {
    "sand": 1,
    "loamy sand": 2,
    "sandy loam": 3,
    "loam": 4,
    "silt loam": 5,
    "clay loam": 6,
    "clay": 7,
}

ALL_TEXTURES = list(TEXTURE_MAP.keys())

def normalize_texture(raw):
    if raw is None:
        return None
    s = raw.strip().lower()

    if "sandy loam" in s: return "sandy loam"
    if "clay loam" in s: return "clay loam"
    if "silt loam" in s: return "silt loam"
    if "loamy sand" in s: return "loamy sand"
    if "sand" in s and "loam" not in s: return "sand"
    if "clay" in s and "loam" not in s: return "clay"
    if "loam" in s: return "loam"

    return None

def texture_score(texture_value, preferred_list, tol_steps=1):
    if texture_value is None:
        return 0.5

    texture_value = normalize_texture(texture_value) or texture_value
    if texture_value not in TEXTURE_MAP:
        return 0.5

    v = TEXTURE_MAP[texture_value]

    pref_codes = []
    for p in preferred_list:
        p2 = normalize_texture(p) or p
        if p2 in TEXTURE_MAP:
            pref_codes.append(TEXTURE_MAP[p2])

    if not pref_codes:
        return 0.5

    d = min(abs(v - pc) for pc in pref_codes)

    if d == 0: return 1.0
    if d <= tol_steps: return 0.8
    if d <= 2 * tol_steps: return 0.4
    return 0.0


# -----------------------------
# 2) NUMERIC SCORE
# -----------------------------
def score_range(value, min_val, max_val):
    if value < min_val:
        return max(0, 1 - (min_val - value) / (min_val if min_val != 0 else 1))
    elif value > max_val:
        return max(0, 1 - (value - max_val) / (max_val if max_val != 0 else 1))
    return 1.0

CROP_REQUIREMENTS = {
    "Brinjal": {
        "temperature": (25, 32),
        "rainfall": (375, 750),
        "ph": (5.5, 5.8),
        "oc": (1.2, 2.5),
        "cec": (12, 25),
        "awc": (0.015, 0.03),
        "bulk_density": (1.1, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Luffa": {
        "temperature": (20, 30),
        "rainfall": (1200, 1500),
        "ph": (5.5, 7.5),
        "oc": (1.5, 3.0),
        "cec": (15, 30),
        "awc": (0.018, 0.035),
        "bulk_density": (1.0, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam", "loam"],
        "texture_tol": 1,
    },

    "Okra": {
        "temperature": (20, 35),
        "rainfall": (250, 550),
        "ph": (6.0, 7.5),
        "oc": (1.0, 2.2),
        "cec": (10, 22),
        "awc": (0.012, 0.028),
        "bulk_density": (1.1, 1.5),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Cucumber": {
        "temperature": (20, 30),
        "rainfall": (350, 550),
        "ph": (5.5, 7.5),
        "oc": (1.2, 2.8),
        "cec": (12, 25),
        "awc": (0.015, 0.030),
        "bulk_density": (1.0, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Snake Gourd": {
        "temperature": (24, 27),
        "rainfall": (400, 600),
        "ph": (5.5, 7.5),
        "oc": (1.3, 2.8),
        "cec": (14, 26),
        "awc": (0.015, 0.030),
        "bulk_density": (1.0, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Bitter Gourd": {
        "temperature": (20, 35),
        "rainfall": (450, 500),
        "ph": (5.5, 7.5),
        "oc": (1.2, 2.5),
        "cec": (12, 24),
        "awc": (0.014, 0.030),
        "bulk_density": (1.0, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Radish": {
        "temperature": (18, 25),
        "rainfall": (125, 400),
        "ph": (6.0, 7.5),
        "oc": (1.0, 2.0),
        "cec": (10, 20),
        "awc": (0.012, 0.028),
        "bulk_density": (1.1, 1.5),
        "sunshine_hours": (5, 8),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 1,
    },

    "Capsicum": {
        "temperature": (21, 28),
        "rainfall": (300, 500),
        "ph": (5.5, 6.8),
        "oc": (1.5, 3.0),
        "cec": (15, 30),
        "awc": (0.018, 0.035),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (6, 9),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 1,
    },

    "Yard Long Bean": {
        "temperature": (15, 28),
        "rainfall": (360, 600),
        "ph": (5.5, 7.5),
        "oc": (1.0, 2.2),
        "cec": (10, 22),
        "awc": (0.015, 0.030),
        "bulk_density": (1.1, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam", "loam"],
        "texture_tol": 2,
    },

    "Banana": {
        "temperature": (26, 38),
        "rainfall": (1250, 1900),
        "ph": (5.5, 7.0),
        "oc": (1.8, 3.5),
        "cec": (18, 35),
        "awc": (0.020, 0.040),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam", "clay loam"],
        "texture_tol": 2,
    },

    "Pineapple": {
        "temperature": (22, 32),
        "rainfall": (1500, 3000),
        "ph": (4.5, 6.5),
        "oc": (1.2, 2.5),
        "cec": (10, 22),
        "awc": (0.015, 0.030),
        "bulk_density": (1.0, 1.4),
        "sunshine_hours": (6, 8),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 1,
    },

    "Ginger": {
        "temperature": (18, 30),
        "rainfall": (1300, 1500),
        "ph": (6.0, 6.5),
        "oc": (1.5, 3.0),
        "cec": (15, 30),
        "awc": (0.018, 0.035),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (5, 7),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Papaya": {
        "temperature": (25, 35),
        "rainfall": (1000, 1500),
        "ph": (5.5, 6.5),
        "oc": (1.2, 2.5),
        "cec": (12, 25),
        "awc": (0.015, 0.030),
        "bulk_density": (1.1, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 1,
    },

    "Passion Fruit": {
        "temperature": (25, 35),
        "rainfall": (1500, 2000),
        "ph": (6.0, 7.5),
        "oc": (1.3, 2.8),
        "cec": (12, 26),
        "awc": (0.018, 0.035),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (6, 9),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Sweet Potato": {
        "temperature": (25, 35),
        "rainfall": (700, 800),
        "ph": (5.5, 6.5),
        "oc": (1.0, 2.0),
        "cec": (10, 20),
        "awc": (0.015, 0.030),
        "bulk_density": (1.1, 1.5),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 1,
    },

    "Rambutan": {
        "temperature": (25, 35),
        "rainfall": (1500, 2000),
        "ph": (5.5, 6.5),
        "oc": (1.5, 3.0),
        "cec": (15, 30),
        "awc": (0.018, 0.035),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 1,
    },

    "Mangosteen": {
        "temperature": (25, 32),
        "rainfall": (1500, 2000),
        "ph": (5.0, 6.5),
        "oc": (1.5, 3.0),
        "cec": (18, 35),
        "awc": (0.020, 0.040),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (4, 8),
        "preferred_textures": ["clay loam", "clay"],
        "texture_tol": 2,
    },

    "Manioc": {
        "temperature": (25, 29),
        "rainfall": (700, 3000),
        "ph": (5.5, 7.5),
        "oc": (0.8, 2.0),
        "cec": (8, 18),
        "awc": (0.015, 0.030),
        "bulk_density": (1.2, 1.6),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["sandy loam"],
        "texture_tol": 2,
    },

    "Kiriala": {
        "temperature": (21, 30),
        "rainfall": (700, 1000),
        "ph": (5.5, 6.5),
        "oc": (1.0, 2.2),
        "cec": (10, 22),
        "awc": (0.015, 0.030),
        "bulk_density": (1.1, 1.5),
        "sunshine_hours": (3, 6),
        "preferred_textures": ["clay loam"],
        "texture_tol": 1,
    },

    "Yams": {
        "temperature": (25, 30),
        "rainfall": (1000, 1500),
        "ph": (5.5, 7.5),
        "oc": (1.2, 2.5),
        "cec": (12, 25),
        "awc": (0.018, 0.035),
        "bulk_density": (1.1, 1.4),
        "sunshine_hours": (7, 10),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },

    "Turmeric": {
        "temperature": (20, 30),
        "rainfall": (1000, 2000),
        "ph": (5.0, 7.5),
        "oc": (1.5, 3.0),
        "cec": (15, 30),
        "awc": (0.018, 0.035),
        "bulk_density": (1.0, 1.3),
        "sunshine_hours": (3, 6),
        "preferred_textures": ["loam"],
        "texture_tol": 1,
    },
}


# -----------------------------
# 5) GROUP WEIGHTS (your style)
# -----------------------------
CROP_WEIGHTS = {
    "Brinjal": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
    "Luffa": {"climate": 0.50, "physical": 0.25, "chemical": 0.25},
    "Okra": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
    "Cucumber": {"climate": 0.50, "physical": 0.25, "chemical": 0.25},
    "Snake Gourd": {"climate": 0.50, "physical": 0.25, "chemical": 0.25},
    "Bitter Gourd": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
    "Radish": {"climate": 0.35, "physical": 0.40, "chemical": 0.25},
    "Capsicum": {"climate": 0.45, "physical": 0.25, "chemical": 0.30},
    "Yard Long Bean": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},

    "Banana": {"climate": 0.50, "physical": 0.30, "chemical": 0.20},
    "Pineapple": {"climate": 0.45, "physical": 0.25, "chemical": 0.30},
    "Papaya": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
    "Passion Fruit": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
    "Rambutan": {"climate": 0.40, "physical": 0.30, "chemical": 0.30},
    "Mangosteen":{"climate": 0.45, "physical": 0.30, "chemical": 0.25},

    "Sweet Potato": {"climate": 0.40, "physical": 0.40, "chemical": 0.20},
    "Manioc": {"climate": 0.45, "physical": 0.35, "chemical": 0.20},
    "Kiriala": {"climate": 0.40, "physical": 0.40, "chemical": 0.20},
    "Yams": {"climate": 0.40, "physical": 0.40, "chemical": 0.20},
    "Ginger": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
    "Turmeric": {"climate": 0.45, "physical": 0.30, "chemical": 0.25},
}


ROWS_PER_CROP = 100
rows = []

crop_names = list(CROP_REQUIREMENTS.keys())

for crop in crop_names:
    req = CROP_REQUIREMENTS[crop]
    w = CROP_WEIGHTS[crop]

    for _ in range(ROWS_PER_CROP):

        # -----------------------------
        # GENERATE RANDOM ENVIRONMENT
        # -----------------------------
        temperature = random.uniform(15, 40)
        rainfall = random.uniform(200, 3000)
        sunshine_hours = random.uniform(3, 10)

        ph = random.uniform(4.5, 8.0)
        oc = random.uniform(0.5, 4.0)
        cec = random.uniform(5, 40)

        awc = random.uniform(0.010, 0.045)
        bulk_density = random.uniform(0.9, 1.7)

        texture = random.choice(ALL_TEXTURES)
        texture_code = TEXTURE_MAP[texture]

        # -----------------------------
        # CLIMATE SCORE
        # -----------------------------
        climate_score = (
            score_range(temperature, *req["temperature"]) +
            score_range(rainfall, *req["rainfall"]) +
            score_range(sunshine_hours, *req["sunshine_hours"])
        ) / 3 * w["climate"]

        # -----------------------------
        # PHYSICAL SCORE (UPDATED)
        # Removed rooting depth
        # -----------------------------
        t_score = texture_score(texture, req["preferred_textures"], req["texture_tol"])

        physical_score = (
            score_range(awc, *req["awc"]) +
            score_range(bulk_density, *req["bulk_density"]) +
            t_score
        ) / 3 * w["physical"]

        # -----------------------------
        # CHEMICAL SCORE
        # -----------------------------
        chemical_score = (
            score_range(ph, *req["ph"]) +
            score_range(oc, *req["oc"]) +
            score_range(cec, *req["cec"])
        ) / 3 * w["chemical"]

        # -----------------------------
        # FINAL SCORE
        # -----------------------------
        total = climate_score + physical_score + chemical_score

        label = "Suitable" if total >= 0.75 else "Unsuitable"

        # -----------------------------
        # STORE ROW
        # -----------------------------
        rows.append({
            "crop": crop,
            "temperature": temperature,
            "rainfall": rainfall,
            "sunshine_hours": sunshine_hours,
            "ph": ph,
            "organic_carbon": oc,
            "cec": cec,
            "awc": awc,
            "bulk_density": bulk_density,
            "texture": texture,
            "texture_code": texture_code,
            "suitability": round(total, 3),
            "suitability_class": label
        })

# -----------------------------
# SAVE DATASET
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv("Crop_training_data.csv", index=False)

print("Saved:", df.shape, "-> Crop_training_data.csv")
print(df["suitability_class"].value_counts())