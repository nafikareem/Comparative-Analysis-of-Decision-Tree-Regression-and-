# Comparative Analysis of Decision Tree Regression and Random Forest Regression Models for Prediction of NOx Gas Emissions from Backhoe Machines

One source of NOx emissions comes from the use of heavy construction equipment such as backhoe machines used in construction and mining projects. This study aims to compare the performance of two machine learning algorithms, namely Decision Tree Regression (DTR) and Random Forest Regression (RFR), in predicting NOx gas emissions from backhoe machines

The data used in this study is secondary data from 37528 backhoe engine operational data which includes variables such as engine type (Backhoe), engine speed (RPM), engine power (HP), Manifold Absolute Pressure (MAP), engine age (Age), engine tier technology type (Engine_Tier), engine temperature (TEMP[C]), and NOx gas emissions produced (NOx[g/s]). The data was processed through several stages, including data cleaning, Pearson correlation test, Z-score standardization, and K-FoldCross-Validation. Modeling was performed using the DTR and RFR algorithms with hyperparameter optimization using Grid Search, while model performance evaluation was performed with regression performance indicators, namely RMSE, MAPE, and R². Experiments were conducted by comparing the performance of the two models based on RMSE, MAPE, and R² metrics.

The results show that the RFR model provides more accurate prediction results than DTR. Based on testing, the RFR model achieved an RMSE value of 0.0065 both before and after tuning, a decrease in MAPE of 14.1041% to 14% after tuning, and a fairly stable R2 of 0.8911 before tuning and 0.8894 after tuning. While the DTR model experienced a decrease in RMSE value of 0.0085 to 0.0075 after tuning, a decrease in MAPE of 17.1751% to 15.8783% after tuning, and an increase in R2 of 0.81490 to 0.8539 after tuning. These results indicate that RFR is more suitable for use in the prediction of NOx emissions from backhoe engines due to its higher accuracy and ability to handle complex variables.

## Run steamlit app
```
streamlit run dashboard.py
```
![image](https://github.com/user-attachments/assets/598aa3b6-054d-4abb-b28c-72e83426bfc7)
