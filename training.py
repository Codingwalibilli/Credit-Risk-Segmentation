import pandas as pd
import numpy as np
import time
import joblib
from tqdm import tqdm
from kmodes.kprototypes import KPrototypes

K = 5
MODEL_PATH = "kprototypes_model.pkl"
df = pd.read_csv("final_dataset.csv")

float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype('float32')

categorical_cols = [
    'Gender', 'Existing Customer', 'State', 'City', 
    'Employment Profile', 'Occupation', 'Is_Max_Loan', 'Is_Max_Profile', 
    'Is_Max_LTV', 'Is_Max_Loan_Amount', 'Is_Max_Profile_Score', 
    'Is_Min_LTV', 'Is_Min_Credit_Score', 'Is_Max_Credit_Score'
]
numerical_cols = df.select_dtypes(include=['float32']).columns.tolist()

cat_idx = [df.columns.get_loc(col) for col in categorical_cols]

print(f"Categorical Columns: {categorical_cols}")
print(f"Numerical Columns: {numerical_cols}")

for col in categorical_cols:
    df[col] = df[col].astype(str)
X = df.values

print("\nTraining K-Prototypes...")

start_time = time.time()

kp = KPrototypes(
    n_clusters=K,
    init='Huang',
    n_init=1,         
    max_iter=20,
    verbose=2
)
clusters = kp.fit_predict(X, categorical=cat_idx)

end_time = time.time()

print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")

model_bundle = {
    "model": kp,
    "categorical_cols": categorical_cols,
    "column_order": df.columns.tolist()
}
joblib.dump(model_bundle, MODEL_PATH, compress=3)

print(f"Model saved to {MODEL_PATH}")

df_og = pd.read_csv('credit_data.csv')
df_og["cluster"] = clusters.astype('int8')
df_og.to_csv("result.csv", index=False)