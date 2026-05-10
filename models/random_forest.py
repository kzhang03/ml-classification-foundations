from sklearn.ensemble import RandomForestClassifier

class RFC:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)