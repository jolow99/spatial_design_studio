LIGHT_GREY = [200, 200, 200]

ET_COLOR_MAPPING = {
    0: LIGHT_GREY,
    1: [0, 255, 0],
    2: [255, 255, 0],
    3: [255, 165, 0],
    4: [255, 0, 0]
}

ET_CLASS_LABELS = {
    0: "No Attention",
    1: "Very Low",
    2: "Low",
    3: "Medium",
    4: "High"
}

EEG_COLOR_MAPPING = {
    0: [255, 0, 0],       # Red
    1: [255, 165, 0],     # Orange
    2: LIGHT_GREY,        # Light Grey
    3: [0, 255, 0],       # Green
    4: [0, 100, 0]        # Dark Green
}

EEG_CLASS_LABELS = {
    0: "Very Negative",
    1: "Slightly Negative",
    2: "No Attention",
    3: "Slightly Positive",
    4: "Very Positive"
}