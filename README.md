# Forecasting Sales

Problem Statement:
	Generate monthly sales forecast of each store for a given department on M5 store-department data. You are required to use Spark MLIB library for building the ML pipeline. The pipeline should have following modules: 

1.	Preprocessing: 
Write following Spark MLIB transformers in this module. You need to write unit tests for these transformers as well. 
o	DataAggregation Store-department level (Transformer) 
o	MarkZero/NegativeSales
o	ImputeMean (store) 
o	Train_Test Split

2.	Feature Engineering
Write following Spark MLIB transformers for features in this module. You need to write unit tests for these transformers as well.
o	LagFeature
o	LogTransformation
o	AntiLogTranformation

3.	Model Training 
Write custom Spark MLIB estimator and model for following ML models in this module. 
o	Random Forrest
o	XGBoost (xgboost library)
Estimator will do:
•	Hyper parameter tunning using Hyperopt 
•	Re-training using best parameters
Model will generate prediction on test data.




4.	Evaluator
Write custom Spark MLIB evaluator for MAPE evaluation matrix. You need to write unit tests for your evaluator as well.

5.	Model Selection
Compare the results of all available models using MAPE on test data. You need to generate the sales forecast for next 24 months using the best performing models. The forecast should be stored as csv and should have store id, month, year, original sales, forecast columns. 
