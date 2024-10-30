SVMHyperParameterSpace = {
    "kernel":{ "options": ["linear","poly","rbf","sigmoid"], "type": "categorical"},
    "C": {"range": [0.1, 10.0], "type": "continuous"},
    "coef0": {"range": [0.0, 1.0], "type": "continuous"},
    "gamma": {"options": ["scale", "auto"], "type": "categorical"}
}


BOHyperParameterSpace = {
    "acquisition_function": {
        "options": ["expected_improvement", "probability_of_improvement", "upper_confidence_bound"],
        "type": "categorical",
        "expected_improvement": {
            "xi": {"range": [0.0, 1.0], "type": "continuous"}
        },
        "probability_of_improvement": {
            "xi": {"range": [0.0, 1.0], "type": "continuous"}
        },
        "upper_confidence_bound": {
            "kappa": {"range": [0.5, 10.0], "type": "continuous"}
        }
    },
    "GPHyperParameter": {
        "noise_level": {"range": [1e-5, 1.0], "type": "continuous"},
        "amplitude": {"range": [0.1, 5.0], "type": "continuous"},
        "kernel": {
            "options": ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared"],
            "type": "categorical",
            "kernel_specific": {
                "RBF": {
                    "length_scale": {"range": [0.1, 10.0], "type": "continuous"}
                },
                "Matern": {
                    "length_scale": {"range": [0.1, 10.0], "type": "continuous"},
                    "nu": {"options": [0.5, 1.5, 2.5], "type": "categorical"}
                },
                "RationalQuadratic": {
                    "length_scale": {"range": [0.1, 10.0], "type": "continuous"},
                    "alpha": {"range": [0.1, 2.0], "type": "continuous"}
                },
                "ExpSineSquared": {
                    "length_scale": {"range": [0.1, 10.0], "type": "continuous"},
                    "periodicity": {"range": [0.1, 2.0], "type": "continuous"}
                }
            }
        }
    }
}

