import numpy as np 
import pandas as pd
import math as math
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from pandas_profiling import ProfileReport


import warnings
from dinosay import dinoprint, DINO_TYPE,Dino
from sklearn.impute import SimpleImputer
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
        
class mod_selection:

	def __init__(self):


		self.dinos = ["tyrannosaurus", "dimetrodon", "ankylosaur", "hypsilophodon","stegosaurus", "deinonychus", "pterodactyl", "archaeopteryx", "maiasaur", "pleisiosaur", "brachiosaur", "corythosaur", "parasaurolophus","triceratops"]
		self.behav = ["normal", "happy", 'joking', 'lazy', 'tired', 'nerd', 'cyborg', 'dead', 'trance', 'stoned']

	def read_me(self):
		self.readme = """
WELCOME!

caremico.mod_selection() is a small library meant to aide in  model selection, using an Information-Theoretic framework, and building predictive models using the latest machine learning algorithms, all in the same place.

The most important aspect about using this library, and about AIC-based model selection, is building a solid a priori model set based on the your knowledge and expertise on the study system/species. Read and think deeply on the subject, then think some more and seek inspiration in order to produce relevant competing hypotheses!

The overall pipeline used throughout is: data cleaning, data exporation, model/feature selection, cross-validation, model hyperparametrization, and making predictions. Always look for grouping variables and fit mixed models (eg. LMM, GLMM) to produce more accurate estimates for your explanatory parameters of interest, especially. if your goal is to explain rather than predict.

The main functions and their arguments are:

- caremico.instructions() -> produces step-by-step instructions. Dinos are cool, don't lie.

- caremico.test_dataset(n_samples, n_features, n_informative, random_state, regression) -> produces a test dataset for exercises. For regression exersices set the "regression" argument to True. Same as sklearn make_classification() and make_regression()

- caremico.check_data(data) -> takes one argument, which is the dataframe object where the data is stored. Cxhecks for any missing values in the dataset.

- caremico.impute_data(strategy) -> imputes missing data using the SimpleImputer() for categorical try 'most_frequent',if you only have numerical values try mean, median, etc.

- caremico.explore(data, cmap) -> returns a heatmap of the different explanatory variables (features) with your outcome variable (target). It takes two arguments: the dataframe where the data are stored and the color map (try 'rainbow', 'seismic', 'hsv' or 'plasma').

- caremico.aictable(model_set, model_names) -> returns a dataframe showing each model ranked from best to worst. The function takes 2 arguments: a list containing each model (e.g., model_set = [sex, age]) and a list containing the names of each model in model_set (e.g., model_names = ['sex', 'age']).

- caremico.best_fit() -> returns the name and corresponding statistics for a single best-fit model (i.e., AIC weight > 0.90). If no single model is identified, use caremico.best_ranked() for multi-model inference.

- caremico.best_ranked() -> returns the name and corresponding statistics for the best-ranked models (i.e., AIC cummulative weight > 0.95). Use caremico.mod_avg() to return model-averaged estimates for each parameter in the best-ranked models.

- cafemico.mod_avg() -> returns model-averaged estimates for each parameter in the best-ranked models.

- caremico.cross_val(X, y, classificaion) -> takes 3 arguments and returns the accuracy of the model containing the explanatory variables (features) proxided in the X argument) to the outcome variable in the y argument (target), as well as the best hyperparameters. If your outcome variable is a categorical variable, make sure to set the classification argument to True.

- caremico.hyper(model) -> takes the name of the most accurate model from the list provided by caremico.cross_val() (eg, 'ExtraTreeRegressor()' and tunes it's hyperparameters with GridSearchCV.

- caremico.best_predictions(new_data) -> uses best-fit and best-hyperparameterized (most accurate) model to the dataset (new_data dataframe) provided and adds predictions to new_data.

        """
		print(self.readme)

	def instructions(self):

		dinoprint('FIRST: you must clean the data and make sure\nthat there are no missing values.\nUse caremico.check_data() for a quick check\nand caremico.impute_data() to impute missing data.', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		dinoprint('SECOND: you must create a model set based on your\nknowledge of the study system/species.\n Use caremico.explore() to help build your model set.\nThen create a list with each model\n(e.g., model_set = [sex, temperature, age_sex)\nand a similar list with the names\n(e.g.,model_names = ["Sex","Temp", "Age_and_sex")', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		dinoprint('THIRD: use caremico.aictable() for AIC-based model selection. caremico.best_fit()\nreturns the single best model if one is identified,\notherwise use caremico.best_ranked() for multi-model inferences.', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		dinoprint('FOURTH, use caremico.cross_val() to test\nthe accuracy of the best-fit models\n and caremic.hyper() to tune their hyperparameters.', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		dinoprint('FINALLY, use caremico.best_predictions()\nto predict new data with the best-fit- and\nbest-hyperparametrized models.', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))

	def test_dataset(self, n_samples, n_features, n_informative, random_state, regression):
		self.regression = regression
		if bool(self.regression) == False:
			X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state)
			self.df = pd.DataFrame(X,y)
			self.df.columns = ["F"+str(i) for i in range(1,n_features+1)]
			self.df = self.df.reset_index().rename(columns={"index": 'target'})
			print(self.df.describe())
			return self.df
		else: 
			X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state)
			self.df = pd.DataFrame(X,y)
			self.df.columns = ["F"+str(i) for i in range(1,n_features+1)]
			self.df = self.df.reset_index().rename(columns={"index": 'target'})
			print(self.df.describe())
			return self.df
        
	def check_data(self, data):
		self.df = data
		self.describe = self.df.describe()
		self.counts = self.describe.loc['count', :]
		self.drop = self.counts.sort_values(ascending = True)
		self.balanced = (self.counts[0] == self.counts).all()
		self.to_drop = str(self.drop.index[0])
		if bool(self.balanced) == True:
			dinoprint('Well done! Your dataset has no missing values.\nTo continue the model-selection process print\nthe caremico.aictable(). You can use cremico.best_fit() to print\nthe single best-fitmodel, if one is identified, or use\ncaremico.best_ranked() to use the 95% best ranked models.\n', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		else:
			dinoprint("Your sample is not balanced.\nPlease consider dropping rows and columns with missing data.\nFor example, {} has the lowest sample size.\nIf this is an important parameter, keep all columns but drop the necessary rows\nwith missing data. Otherwise drop the {} column completely.\nAlternatively, try imputing the missing data with caremico.impute_missing().".format(self.to_drop, self.to_drop, wrap=True), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))

	def explore(self, df, name):
		self.df = df
		self.profile = ProfileReport(self.df, title = name)
		return self.profile.to_notebook_iframe()

	def impute(self, data, strategy):
		self.df = data
		self.imp_most = SimpleImputer(missing_values=np.nan, strategy= strategy)
		self.imp_data = pd.DataFrame(self.imp_most.fit_transform(self.df))
		self.imp_data.columns =self.df.columns
		print(self.imp_data.describe(include = 'all'))
		return self.imp_data

	def drop_missing(self, data):
		self.df = data
		self.drop_data = self.df.dropna(axis = 0)
		print(self.drop_data.describe(include = 'all'))
		return self.drop_data
    
	def aictable(self, model_names, model_set):
		self.names = [list(models.params.index[1:]) for models in model_set]
		self.model_set = model_set
		self.model_estimates= [models.params for models in self.model_set] 
		self.bic = [models.bic for models in model_set] 
		self.delta_bic = [x-min(self.bic) for x in self.bic]
		
		self.aic = [models.aic for models in model_set] 
		self.delta_aic = [x-min(self.aic) for x in self.aic]
		self.exp_delta_aic = [math.exp(-0.5 * x) for x in self.delta_aic]
		self.aic_weight = [x/np.nansum(self.exp_delta_aic) for x in self.exp_delta_aic]
		self.aic_ev_ratio = [max(self.aic_weight)/x for x in self.aic_weight]	
		
		self.log_likelihood= [models.llf for models in model_set] 
		self.k = [(len(models.params)-1) for models in model_set]
		self.n = len(self.df.index)
		
		self.aicc = self.aic+(2*np.square(self.k)+np.multiply(self.k, 2))/(np.subtract(self.n, self.k)-1)
		self.delta_aicc = [x-min(self.aicc) for x in self.aicc]
		self.exp_delta_aicc = [math.exp(-0.5 * x) for x in self.delta_aicc]
		self.aicc_weight = [x/np.nansum(self.exp_delta_aicc) for x in self.exp_delta_aicc]
		self.aicc_ev_ratio = [max(self.aicc_weight)/x for x in self.aicc_weight]	

		self.aictab = pd.DataFrame(index = model_names)
		self.aictab['Mod_Parameters']= self.names
		self.aictab['K']= self.k
		# self.aictab['Deviance'] = [models.deviance for models in model_set] 
		self.aictab['BIC'] = self.bic
		self.aictab['\u0394_BIC'] = self.delta_bic
		
		self.aictab['AICc'] = self.aicc
		self.aictab['\u0394_AICc'] = self.delta_aicc
		self.aictab["exp(-0.5 * \u0394_AICc)"] = self.exp_delta_aicc
		self.aictab['AICc_Weight'] = self.aicc_weight  
		self.aictab['AICc_Evidence_Ratios'] = self.aic_ev_ratio
		# self.aictab = self.aictab.sort_values(by=['\u0394_AICc'], ascending = True)

		self.aictab['AIC'] = self.aic
		self.aictab['\u0394_AIC'] = self.delta_aic
		self.aictab['Log_Likelihood'] = self.log_likelihood
		self.aictab["exp(-0.5 * \u0394_AIC)"] = self.exp_delta_aic
		self.aictab['AIC_Weight'] = self.aic_weight  
		self.aictab['AIC_Evidence_Ratios'] = self.aic_ev_ratio
		self.aictab = self.aictab.sort_values(by=['\u0394_AIC'], ascending = True)
		self.aictab['AIC_Cum_Weight'] = self.aictab['AIC_Weight'].cumsum(skipna = True)
		self.aictab['AICc_Cum_Weight'] = self.aictab['AICc_Weight'].cumsum(skipna = True)

		self.aictable = self.aictab.reset_index()
		self.aictable = self.aictable.rename(columns={'index': "Model_Name"})
		return self.aictable

	def best_fit(self):
		self.best = self.aictab[self.aictab['AIC_Weight'] >= 0.90]
		if self.best.empty:
			dinoprint('Several models in your set best explained the data.\nYou can print caremico.best_ranked() to identify the the 95% best ranked models and then\ncaremico.model_averaging() to produce model-averaged parameter estimates.\n', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		else:
			dinoprint('{} was identified as being the single best-fit model.\nPrint the model summary (model.summary()) and\nuse the estimates to make inferences.\n'.format(self.best_name,wrap =True), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
	def best_ranked(self):
		self.best = self.aictable
		self.best_95 = self.best[self.best['AIC_Cum_Weight'] <= 0.95]
		if self.best_95.empty:
			dinoprint('A single model in your set best explained the data.\nYou can print caremico.best_fit() to\nidentify the model and print the model.summary() for more details.', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		else:
			return self.best_95

	def modavg(self): # get estimates for best ranked onky and return averages
		return pd.DataFrame(caremico.model_estimates).mean()

	def cross_val(self, X, y, regression):
		dinoprint('Crossvalidations are happening!\nMeanwhile, pet your pet!', DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))

		self.regression = regression
		self.X = self.df.loc[:, X]
		self.y = self.df.loc[:, y]

		# Preprocessing the data done by user
		self.enc = preprocessing.OrdinalEncoder()
		self.X = self.enc.fit_transform(self.X)
		self.y = self.enc.fit_transform(self.y)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
		if bool(self.regression) == False:
			# Logistic regression
			self.logistic_classifier = LogisticRegression(random_state = 0)
			self.logist=self.logistic_classifier.fit(self.X_train,self.y_train)
			self.logist_pred = self.logist.predict(self.X_test)
			self.logist_accuracy = accuracy_score(self.y_test, self.logist_pred)
			print("LOGREG: {}".format(self.logist_accuracy))
			#Naive Bayes
			self.nb_classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior =None)
			self.nb = self.nb_classifier.fit(self.X_train, self.y_train)
			self.nb_pred = self.nb.predict(self.X_test)
			self.nb_accuracy = accuracy_score(self.y_test, self.nb_pred)
			print("NB: {}".format(self.nb_accuracy))

			# K-nearest neighbors
			self.knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
			self.knn = self.knn_classifier.fit(self.X_train,self.y_train)
			self.knn_pred = self.knn.predict(self.X_test)
			self.knn_accuracy = accuracy_score(self.y_test, self.knn_pred)
			print("KNN: {}".format(self.knn_accuracy))
            		
			 # random forest classifier
			self.randforest_classifier = RandomForestClassifier()
			self.randforest =self.randforest_classifier.fit(self.X_train,self.y_train)
			self.randforest_pred = self.randforest.predict(self.X_test)
			self.randforest_accuracy = accuracy_score(self.y_test, self.randforest_pred)
			print("RANDFOR: {}".format(self.randforest_accuracy))

			# Support vector machines
			self.svm_classifier = SVC()
			self.svm =self.svm_classifier.fit(self.X_train,self.y_train)
			self.svm_pred = self.svm.predict(self.X_test)
			self.svm_accuracy = accuracy_score(self.y_test, self.svm_pred)
			print("SVM: {}".format(self.svm_accuracy))

           		# XGBoostClassifier()
			self.xgbclass = XGBClassifier()
			self.xgbc = self.xgbclass.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
			self.xgbc_pred = self.xgbc.predict(self.X_test)
			self.xgbc_accuracy = accuracy_score(self.y_test, self.xgbc_pred)
			
			classifier_name = ['LogisticRegression()', 'MultinomialNB()','KNeighborsClassifier()', 'SVC()', 'RandomForestClassifier()', 'XGBClassifier()']
			accuracies  = [self.logist_accuracy, self.nb_accuracy, self.knn_accuracy, self.svm_accuracy, self.randforest_accuracy, self.xgbc_accuracy  ]

			self.accuracy = pd.DataFrame({"Sklearn Classifier": classifier_name, "Accuracy Score": accuracies})
			self.accuracy = self.accuracy.sort_values(by = ['Accuracy Score'], ascending = False).reset_index(drop=True)
			print(self.accuracy)
            
		else: 
			# Ordinary Least Squares (simple) regression
			self.ols_reg =  linear_model.LinearRegression()
			self.ols= self.ols_reg.fit(self.X_train,self.y_train)
			self.ols_pred = self.ols.predict(self.X_test)
			self.ols_accuracy = math.sqrt(mean_absolute_error(self.y_test, self.ols_pred))
			
			# Ridge regression
			self.ridge_reg =  linear_model.Ridge()
			self.ridge= self.ridge_reg.fit(self.X_train,self.y_train)
			self.ridge_pred = self.ridge.predict(self.X_test)
			self.ridge_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.ridge_pred))
			
			# RidgeCV regression
			self.ridgecv_reg =  linear_model.RidgeCV()
			self.ridgecv= self.ridgecv_reg.fit(self.X_train,self.y_train)
			self.ridgecv_pred = self.ridgecv.predict(self.X_test)
			self.ridgecv_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.ridgecv_pred))

			# Elastic Net Regression
			self.elasticnetcv_reg =  linear_model.ElasticNetCV()
			self.elasticnetcv= self.elasticnetcv_reg.fit(self.X_train,self.y_train)
			self.elasticnetcv_pred = self.elasticnetcv.predict(self.X_test)
			self.elasticnetcv_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.elasticnetcv_pred))

			# Lasso Regression
			self.lasso_reg =  linear_model.Lasso()
			self.lasso= self.lasso_reg.fit(self.X_train,self.y_train)
			self.lasso_pred = self.lasso.predict(self.X_test)
			self.lasso_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.lasso_pred))

			# Extra tree Regression
			self.extratree_reg =  ExtraTreesRegressor()
			self.extratree= self.extratree_reg.fit(self.X_train,self.y_train)
			self.extratree_pred = self.extratree.predict(self.X_test)
			self.extratree_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.extratree_pred))  

			# SVR
			self.svr_reg =  SVR()
			self.svr= self.svr_reg.fit(self.X_train,self.y_train)
			self.svr_pred = self.svr.predict(self.X_test)
			self.svr_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.svr_pred))
            
           		# XGBoostRegressor()
			self.xgbreg = XGBRegressor()
			self.xgbr = self.xgbreg.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
			self.xgbr_pred = self.xgbr.predict(self.X_test)
			self.xgbr_accuracy = np.sqrt(mean_absolute_error(self.y_test, self.xgbr_pred)).round(3)

			regression_name = ['LinearRegression()', 'Ridge()', 'RidgeCV()', 'ElasticNetCV()', 'Lasso()', 'ExtraTreesRegressor()', 'SVR()',
                              'XGBoostRegressor()']
			regression_accuracies  = [self.ols_accuracy, self.ridge_accuracy, self.ridgecv_accuracy, self.elasticnetcv_accuracy, 
                                      self.lasso_accuracy, self.extratree_accuracy,self.svr_accuracy, self.xgbr_accuracy]

			self.accuracy = pd.DataFrame({"Model": regression_name, "MAE": regression_accuracies})
			self.accuracy = self.accuracy.sort_values(by = ['MAE'], ascending = True).reset_index(drop=True)
			print(self.accuracy)

	def hyper(self, model):
		dinoprint('Tuning {} to increase accuracy!\nGo get some coffee/tea and call your mom.'.format(model), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))

		if model == 'ExtraTreesRegressor()':
			etr = ExtraTreesRegressor(n_estimators=100, n_jobs=4, max_features = 150, min_samples_split=25,
                            min_samples_leaf=35, criterion = 'absolute_error')
			parameters= {'max_depth' : range (1, 10, 1), 'max_features':range(10, 150, 10)}
			self.grid_search = GridSearchCV(estimator=etr,param_grid=parameters, verbose = 1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			print(self.grid_search.best_estimator_)
			self.etrscore = str(self.grid_search.best_score_.round(2))
			dinoprint('The MAE for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model,self.etrscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))

		if model == 'XGBoostRegressor()':
			xgb1 = XGBRegressor()
			parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
			'objective':['reg:linear'],
			'learning_rate': [.03, 0.05, .07], #so called `eta` value
			'max_depth': [5, 6, 7],
			'min_child_weight': [4],
			'silent': [1],
			'subsample': [0.7],
			'colsample_bytree': [0.7],
			'n_estimators': [10, 50, 200, 500]}
			self.grid_search = GridSearchCV(xgb1,
			parameters,
			cv = 2,
			n_jobs = 5,
			verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.xgbrscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The MAE for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.xgbrscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
            
		if model == 'LinearRegression()':
			dinoprint('{} has no hyperparameters to tune. If this is the\nmost accurate model you can use to make predictons\nby providing new data to caremico.best_predictions()'.format(model), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
            
		if model == 'RidgeCV()':
			ridge = linear_model.RidgeCV()
			ridge_params = {'alphas':[550, 580, 600, 620, 650]}            
			self.grid_search = GridSearchCV(ridge,
			ridge_params,
			cv = 2,
			n_jobs = 5,
			verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.ridgescore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The MAE for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.ridgescore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
            
		if model == 'Lasso()':
			lasso = linear_model.Lasso()
			lasso_params = {'alpha':[0.005, 0.02, 0.03, 0.05, 0.06]}
			self.grid_search = GridSearchCV(lasso,
			lasso_params,
			cv = 2,
			n_jobs = 5,
			verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.lassoscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The MAE for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.lassoscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
            
		if model == 'ElasticNetCV()':
			en = linear_model.ElasticNetCV()
			en_params = {"alphas": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
			self.grid_search = GridSearchCV(en,
			en_params,
			cv = 2,
			n_jobs = 5,
			verbose=1)
			self.grid_search.fit(self.X, self.y)
			self.enscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The MAE for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.enscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
            
		if model == 'SVR()':
			svr = SVR()
			c_range = np.logspace(-0, 4, 8)
			gamma_range = np.logspace(-4, 0, 8)
			tuned_parameters = {'C': c_range,'gamma':gamma_range}
			self.grid_search = GridSearchCV(svr,param_grid=tuned_parameters,
                   scoring='explained_variance',cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.svrscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The MAE for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.svrscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		if model == 'LogisticRegression()':
			logreg = LogisticRegression()
			grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
			self.grid_search = GridSearchCV(logreg,param_grid=grid_values,
                   	cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.logregscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The accuracy score for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.logregscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		if model == 'SVC()':
			svc = SVC()
			param_grid = {'C': [0.1,1, 10, 100, 200], 'gamma': [1,0.1,0.01,0.001]}
			self.grid_search = GridSearchCV(svc,param_grid=param_grid,cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.svcscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The accuracy score for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.svcscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		if model == 'RandomForestClassifier()':
			rantree = RandomForestClassifier()
			param_grid = {'n_estimators': [10,50,100,200, 500],'criterion' :['gini', 'entropy'], 'min_samples_split': [2,4],
						 'min_samples_leaf': [1,2]}
			self.grid_search = GridSearchCV(rantree,param_grid=param_grid,cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.rentrscore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The accuracy score for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.rentrscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		if model == 'MultinomialNB()':
			mnb = MultinomialNB()
			mnbparam_grid = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  }  
			self.grid_search = GridSearchCV(mnb,param_grid=mnbparam_grid,cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.mnbscore = str(self.grid_search.best_score_.round(2))
			print(self.mnbscore)

			print(self.grid_search.best_estimator_)
			dinoprint('The accuracy score for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.mnbscore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		if model == 'XGBClassifier()':
			xgbc = XGBClassifier()
			parameters = {'nthread':[3,4,5], #when use hyperthread, xgboost may become slower
			              'objective':['binary:logistic'],
			              'learning_rate': [0.05,.03, 0.05, .07, 0.1], #so called `eta` value
			              'max_depth': [3,5,6,7],
			              'min_child_weight': [1, 4, 6, 10,11],
			              'subsample': [0.8, 1],
			              'colsample_bytree': [0.7],
			              'n_estimators': [100, 200, 500],
			              'base_score': [0.1,  0.5]}			
			self.grid_search = GridSearchCV(xgbc,param_grid=parameters,cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.xgbccore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The accuracy score for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.xgbccore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		
		if model == 'KNeighborsClassifier()':
			knnc = KNeighborsClassifier()
			k_range = list(range(1, 31))
			param_grid = dict(n_neighbors=k_range)
			self.grid_search = GridSearchCV(knnc,param_grid=param_grid,cv = 2, n_jobs = 5, verbose=1)
			self.grid_search.fit(self.X, np.ravel(self.y))
			self.knnccore = str(self.grid_search.best_score_.round(2))
			print(self.grid_search.best_estimator_)
			dinoprint('The accuracy score for the {} with\ntuned hyperparameters (shown above) is {}. To make predictions\nwith this tuned model just provide new data to caremico.best_predictions().'.format(model, self.knnccore), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))
		else:
			pass
	def best_predictions(self, new_data):
		dinoprint('Making predictions!\nGo get some coffee/tea and call your mom.'.format(model), DINO_TYPE[random.choice(self.dinos)], behavior=random.choice(self.behav))

		self.new_data = new_data
		self.enc = preprocessing.OrdinalEncoder()
		self.X = self.enc.fit_transform(self.new_data)
		self.new_preds = self.grid_search.predict(self.X)
		self.new_data['Predictions'] = self.new_preds
		return self.new_data
