import numpy as np
from dfi.explainers import DFIExplainer
def simple_model(X): return X.sum(axis=1)
data = np.random.randn(100, 5) 
X_test = np.random.randn(10, 5) 
explainer = DFIExplainer(simple_model, data)
results = explainer(X_test)
print("DFI Explanation Results:")
for key, value in results.items():
	print(f"{key}: {value}")