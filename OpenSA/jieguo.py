from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 创建一个虚拟数据集
X, y = make_classification(n_samples=1000, n_features=40, n_classes=2, n_clusters_per_class=2, random_state=42)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义ColumnTransformer，用于将不同类型的特征进行不同的处理
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
        ]), slice(0, 40)),  # 数值特征的处理
        ('categorical_low', Pipeline([
            ('imputer', SimpleImputer(fill_value='missing', strategy='constant')),
            ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), []),
        ('categorical_high', Pipeline([
            ('imputer', SimpleImputer(fill_value='missing', strategy='constant')),
            ('encoding', OrdinalEncoder())
        ]), [])
    ]
)

# 定义SVC分类器
svm_classifier = SVC(random_state=42)

# 定义XGBoost分类器
xgb_classifier = XGBClassifier(
    objective='multi:softprob',  # 多分类问题
    random_state=42
)

# 定义完整的Pipeline，包括数据预处理和分类器
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm_classifier)
])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_classifier)
])

# 训练SVM模型
svm_pipeline.fit(X_train, y_train)

# 训练XGBoost模型
# xgb_pipeline.fit(X_train, y_train)

# 在测试集上进行预测并评估模型性能
svm_y_pred = svm_pipeline.predict(X_test)
# xgb_y_pred = xgb_pipeline.predict(X_test)

print("SVM分类器性能报告:")
print(classification_report(y_test, svm_y_pred))

# print("XGBoost分类器性能报告:")
# print(classification_report(y_test, xgb_y_pred))
