"""train.py と export_cv.py で共有する LightGBM のパラメータ"""
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
