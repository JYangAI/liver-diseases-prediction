import matplotlib.pyplot as plt
import pandas as pd

from tools.common_evaluate_calibration import evaluate_model
from tools.dnn_evaluate_calibration import dnn_evaluate_model
from tools.scaler import scaler

plt.figure(figsize=(10, 8))

# Hepatic steatosis
# ≥ Mild hepatic steatosis
nhanes_df = pd.read_excel('../datasets/NHANES/NHANES_2021_8_2023_8 test data for mild or greater hepatic steatosis.xlsx')
nhanes_df = scaler(nhanes_df, '≥ Mild hepatic steatosis')
nhanes_x = nhanes_df.drop(columns='≥ Mild hepatic steatosis')
nhanes_y = nhanes_df['≥ Mild hepatic steatosis']
print(dnn_evaluate_model('../models/steatosis/dnn_classifier_model_mild_or_greater.pth', nhanes_x, nhanes_y, name='NHANES external test'))

# ≥ moderate hepatic steatosis
# nhanes_df = pd.read_excel('../datasets/NHANES/NHANES_2021_8_2023_8 test data for moderate or greater hepatic steatosis.xlsx')
# nhanes_df = scaler(nhanes_df, '≥ Moderate hepatic steatosis')
# nhanes_x = nhanes_df.drop(columns='≥ Moderate hepatic steatosis')
# nhanes_y = nhanes_df['≥ Moderate hepatic steatosis']
# print(dnn_evaluate_model('../models/steatosis/dnn_classifier_model_moderate_or_greater.pth', nhanes_x, nhanes_y, name='NHANES external test'))

# = Severe hepatic steatosis
# nhanes_df = pd.read_excel('../datasets/NHANES/NHANES_2021_8_2023_8 test data for severe hepatic steatosis.xlsx')
# nhanes_df = scaler(nhanes_df, 'Severe hepatic steatosis')
# nhanes_x = nhanes_df.drop(columns='Severe hepatic steatosis')
# nhanes_y = nhanes_df['Severe hepatic steatosis']
# print(dnn_evaluate_model('../models/steatosis/dnn_classifier_model_severe.pth', nhanes_x, nhanes_y, name='NHANES external test'))

# Liver fibrosis
# ≥ F1 liver fibrosis
# nhanes_df = pd.read_excel('../datasets/NHANES/NHANES_2021_8_2023_8 test data for F1 or greater liver fibrosis.xlsx')
# nhanes_df = scaler(nhanes_df, '≥ F1 liver fibrosis')
# nhanes_x = nhanes_df.drop(columns='≥ F1 liver fibrosis')
# nhanes_y = nhanes_df['≥ F1 liver fibrosis']
# print(evaluate_model('../models/fibrosis/xgboost_classifier_model_F1_or_greater.joblib', nhanes_x, nhanes_y, name='NHANES external test'))

# ≥ F2 liver fibrosis
# nhanes_df = pd.read_excel('../datasets/NHANES/NHANES_2021_8_2023_8 test data for F2 or greater liver fibrosis.xlsx')
# nhanes_df = scaler(nhanes_df, '≥ F2 liver fibrosis')
# nhanes_x = nhanes_df.drop(columns='≥ F2 liver fibrosis')
# nhanes_y = nhanes_df['≥ F2 liver fibrosis']
# print(evaluate_model('../models/fibrosis/xgboost_classifier_model_F2_or_greater.joblib', nhanes_x, nhanes_y, name='NHANES external test'))

# ≥ F3 liver fibrosis
# nhanes_df = pd.read_excel('../datasets/NHANES/NHANES_2021_8_2023_8 test data for F3 or greater liver fibrosis.xlsx')
# nhanes_df = scaler(nhanes_df, '≥ F3 liver fibrosis')
# nhanes_x = nhanes_df.drop(columns='≥ F3 liver fibrosis')
# nhanes_y = nhanes_df['≥ F3 liver fibrosis']
# print(evaluate_model('../models/fibrosis/xgboost_classifier_model_F3_or_greater.joblib', nhanes_x, nhanes_y, name='NHANES external test'))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('Predicted probability', font={'family': 'Times New Roman'})
plt.ylabel('True probability', font={'family': 'Times New Roman'})
plt.title('Test calibration curve (≥ mild hepatic steatosis)', font={'family': 'Times New Roman'})
# plt.title('Test calibration curve (≥ moderate hepatic steatosis)', font={'family': 'Times New Roman'})
# plt.title('Test calibration curve (= severe hepatic steatosis)', font={'family': 'Times New Roman'})
# plt.title('Test calibration curve (≥ F1 liver fibrosis)', font={'family': 'Times New Roman'})
# plt.title('Test calibration curve (≥ F2 liver fibrosis)', font={'family': 'Times New Roman'})
# plt.title('Test calibration curve (≥ F3 liver fibrosis)', font={'family': 'Times New Roman'})
plt.yticks(fontproperties={'family': 'Times New Roman'})
plt.xticks(fontproperties={'family': 'Times New Roman'})
plt.legend(loc='lower right')
plt.grid(False)
plt.show()
