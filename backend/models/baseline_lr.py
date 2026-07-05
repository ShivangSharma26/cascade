from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_baseline_model(class_weight=None):
    """
    Returns a Logistic Regression baseline model with standard scaling and imputation.
    """
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42))
    ])
    return model

if __name__ == '__main__':
    model = get_baseline_model()
    print("Baseline model defined.")
