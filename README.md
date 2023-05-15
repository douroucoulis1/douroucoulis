# douroucoulis

Description: douroucoulis is a library that synergizes Information-Theory model selection, the latest Machine Learning algorithms, and numerous Data Science tools in Python.

WELCOME!

mod_selection() is Python class with a small library of methods meant to aide in  model selection, using an Information-Theoretic framework (& other feature-reducing approaches), to build predictive models using the latest machine learning algorithms, all in the same place.

The most important aspect about using this library, and about AIC-based model selection, is building a solid a priori model set based on the your knowledge and expertise on the study system/species. Read and think deeply on the subject, then think some more and seek inspiration in order to produce relevant competing hypotheses!

The overall pipeline used throughout is: data cleaning, data exporation, model/feature selection/reduction, cross-validation, model hyperparametrization, and making predictions. Always look for grouping variables and fit mixed models (eg. LMM, GLMM) to produce more accurate estimates for your explanatory parameters of interest, especially. if your goal is to explain rather than predict.

The main methods and their arguments are:

- instructions() -> produces step-by-step instructions. Dinos are cool, don't lie.

- test_dataset(n_samples, n_features, n_informative, random_state, regression) -> produces a test dataset for exercises. For regression exersices set the "regression" argument to True. Same as sklearn make_classification() and make_regression()

- check_data(data) -> takes one argument, which is the dataframe object where the data is stored. Cxhecks for any missing values in the dataset.

- impute_data(strategy) -> imputes missing data using the SimpleImputer() for categorical try 'most_frequent',if you only have numerical values try mean, median, etc.

- explore(data, cmap) -> returns a heatmap of the different explanatory variables (features) with your outcome variable (target). It takes two arguments: the dataframe where the data are stored and the color map (try 'rainbow', 'seismic', 'hsv' or 'plasma').

- aictable(model_set, model_names) -> returns a dataframe showing each model ranked from best to worst. The function takes 2 arguments: a list containing each model (e.g., model_set = [sex, age]) and a list containing the names of each model in model_set (e.g., model_names = ['sex', 'age']).

- best_fit() -> returns the name and corresponding statistics for a single best-fit model (i.e., AIC weight > 0.90). If no single model is identified, use best_ranked() for multi-model inference.

- best_ranked() -> returns the name and corresponding statistics for the best-ranked models (i.e., AIC cummulative weight > 0.95). Use caremico.mod_avg() to return model-averaged estimates for each parameter in the best-ranked models.

- mod_avg() -> returns model-averaged estimates for each parameter in the best-ranked models.

- cross_val(X, y, classificaion) -> takes 3 arguments and returns the accuracy of the model containing the explanatory variables (features) proxided in the X argument) to the outcome variable in the y argument (target), as well as the best hyperparameters. If your outcome variable is a categorical variable, make sure to set the classification argument to True.

- hyper(model) -> takes the name of the most accurate model from the list provided cross_val() (eg, 'ExtraTreeRegressor()' and tunes it's hyperparameters with GridSearchCV.

- best_predictions(new_data) -> uses best-fit and best-hyperparameterized (most accurate) model to the dataset (new_data dataframe) provided and adds predictions to new_data.
