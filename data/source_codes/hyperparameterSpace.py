SVMHyperParameterSpace = {
    "kernel":{ "options": ["linear","poly","rbf","sigmoid"], "type": "categorical"},
    "C": {"range": [0.1, 10.0], "type": "continuous"},
    "coef0": {"range": [0.0, 1.0], "type": "continuous"},
    "gamma": {"options": ["scale", "auto"], "type": "categorical"}
}
