from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class LRCV:
    def __init__(self, Cs=10, l1_ratios=None, cv=5, scoring="recall", solver="lbfgs",
                tol=1e-4, max_iter=2000, scaler=StandardScaler()):
        """
        NOTE: l1_ratios=(0.0,) or None is L2, l1_ratios=(1.0,) is L1, elasticnet is tuple with any values
        between 0.0 and 1.0
        """
        assert scoring == "recall" or scoring == "f1", f"scoring must be either recall or f1. Got: {scoring}"
        assert (
            (l1_ratios == None and solver == "lbfgs") or
            (any(r > 0.0 for r in l1_ratios) and solver == "saga")
        ), "Use solver='lbfgs' for L2 only; use solver='saga' for L1 or Elastic Net"
        self.model = Pipeline([
            ("scaler", scaler),
            ("logreg", LogisticRegressionCV(
                Cs=Cs,
                l1_ratios=l1_ratios,
                cv=cv,
                scoring=scoring,
                solver=solver,
                tol=tol,
                max_iter=max_iter,
                class_weight="balanced",
                n_jobs=-2,
                random_state=42,
            ))
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def best_Cs(self):
        return self.model.named_steps["logreg"].C_