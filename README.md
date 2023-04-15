# Automatically Standardizing Data Preparation Programs

In addition to our code, we use this repo as an appendix. 

## Experimental 

### Prompt to GPT-models

```
"I have a Python script that I would like you to edit. " \
    + 'The modified script should be able to run and maintain similar semantics to that of the unmodified version.' \
    + f"The script is from Kaggle's '{competition_full_name}' competition." \
    + "The goal is to make the script more standard regarding other scripts in the same competition. That means it should use common libraries, common functions, etc." \
    + "You should only return the code." \
    + "I am sending it as an attachment now. \n```python \n"
```

### Detailed description of Kaggle datasets
We crawled the Jupyter notebooks associated with given topics. Notebooks of a given topic use the same set of publicly available input files and have the same performance goal. Therefore, they are considered codes drawn from the same latent space. Statistics of notebooks with each topic are shown in Table 2. The execution of downloaded scripts can fail for various reasons, including missing packages in the local environment due to the under-specification of Python library dependency in notebooks and incorrect input file paths only applicable to the notebook creator's local file system. We implement a tool that can parse error traces produced from the execution of a script and automatically attempt to fix the error accordingly. Scripts that cannot be fixed after multiple attempts are removed.
* **Titanic** is created from the ***Titanic*** competition whose goal is to create a model that predicts which passengers survived the Titanic shipwreck.
* **Sales** is created from the ***Predict Future Sale*** competition whose goal is to predict total sales for every product and store in the next month.
* **House** is created from the ***House Prices*** competition whose goal is to use 79 variables describing aspects of residential homes to predict the final price of each house.
* **NLP** is created from the ***NLP with Disaster Tweets*** competition whose goal is to build a model that predicts which Tweets are about real disasters and which ones are not.
* **Spaceship** is created from the ***Spaceship Titanic*** competition whose goal is to predict which passengers were transported from an imaginary spaceship using records recovered from its damaged computer system.
* **Medical** is created from notebooks that use the ***Pima Indians Diabetes Database*** dataset with the goal of diagnostically predicting whether or not a patient has diabetes.

### Input data files
* **Titanic**: https://www.kaggle.com/c/titanic
* **Sales**: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales
* **House**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
* **NLP**: https://www.kaggle.com/competitions/nlp-getting-started
* **Spaceship**: https://www.kaggle.com/competitions/spaceship-titanic
* **Medical**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database 
