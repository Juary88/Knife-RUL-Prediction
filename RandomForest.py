from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2,
                    random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0,
                          n_estimators=100)
regr.fit(X, y)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
       max_features='auto', max_leaf_nodes=None,
       min_impurity_decrease=0.0, min_impurity_split=None,
       min_samples_leaf=1, min_samples_split=2,
       min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
       oob_score=False, random_state=0, verbose=0, warm_start=False)
print("X:..............................")
print(X)
print("y:..............................")
print(y)
#  print(regr.feature_importances_)
# [0.18146984 0.81473937 0.00145312 0.00233767]
#  print(regr.predict([[0, 0, 0, 0]]))
# [-8.32987858]