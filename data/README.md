# Data

`raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` is the IBM HR Analytics Employee Attrition & Performance dataset
(1,470 employees, 35 columns, 237 leavers).

Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

`Age` and `Gender` are never used as model inputs. They only define the fairness groups
(gender; age bands <30, 30-39, 40+). `EmployeeNumber` and the constant columns `EmployeeCount`, `Over18`
and `StandardHours` are dropped.
