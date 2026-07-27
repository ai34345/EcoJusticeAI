from violations.ml.models import build_model
import numpy as np
model = build_model()
X = np.random.rand(2, 60, 19).astype(np.float32)
y = model.predict(X, verbose=0)
assert y.shape == (2, 1), "Wrong output shape!"
assert (y >= 0).all() and (y <= 1).all(), "Output should be 0-1!"
print("✅ Model works")