import torch
import gpytorch

class PI:
    def __init__(self, xi=0.1):
        self.xi = xi

    def compute(self, X_candidates, model, y_best):
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = model(X_candidates)
            mean = preds.mean
            std = preds.variance.sqrt()

        Z = (mean - y_best - self.xi) / (std + 1e-9) 
        pi = torch.distributions.Normal(0, 1).cdf(Z)
        return pi.numpy()
