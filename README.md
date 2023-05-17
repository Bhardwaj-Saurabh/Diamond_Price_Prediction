# Diamond Price Prediction
A data science project to predict the price of the Diamond.

<img src="https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction/blob/master/reports/figures/diamond-diamond.jpeg">

## 1.0 Business Problem
The business problem is to develop a model that accurately predicts diamond prices based on various attributes such as carat size, cut quality, color, clarity, and other factors. By solving this problem, businesses in the diamond industry, such as diamond retailers or wholesalers, can optimize their pricing strategy and make informed decisions when buying or selling diamonds.

The objective is to leverage data science and predictive modeling techniques to create a model that can estimate diamond prices with a high degree of accuracy. This model can help businesses determine the optimal price for a given diamond based on its characteristics, market trends, and customer preferences.

By solving this business problem, companies can make data-driven decisions, improve pricing strategies, and enhance their competitiveness in the market. It can also enable them to identify pricing anomalies, detect potential opportunities for profit maximization, and provide valuable insights for inventory management and sales forecasting.

## 2.0 Data Description
**The dataset** The goal is to predict `price` of given diamond (Regression Analysis).
There are 10 independent variables (including `id`):

* `id` : unique identifier of each diamond
* `carat` : Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
* `cut` : Quality of Diamond Cut
* `color` : Color of Diamond
* `clarity` : Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
* `depth` : The depth of diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface)
* `table` : A diamond's table is the facet which can be seen when the stone is viewed face up.
* `x` : Diamond X dimension
* `y` : Diamond Y dimension
* `x` : Diamond Z dimension

Target variable:
* `price`: Price of the given Diamond.

Dataset Source Link :
[Gemstone Price Dataset](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

## 3.0 Business Assumptions
Followings assumptions have been made to define the solution strategy for diamind price prediction.

1. Diamond Prices Reflect Market Demand: The assumption is that diamond prices are influenced by market demand. Factors such as carat size, cut quality, color, clarity, and other characteristics affect the price. It is assumed that customers are willing to pay more for diamonds with desirable attributes.

2. Price Consistency: It is assumed that there is a level of consistency in diamond pricing across different regions and markets. While there may be variations due to local factors and market dynamics, the assumption is that overall, diamond prices follow similar trends and patterns.

3. Economic Factors: It is assumed that economic factors, such as inflation rates, consumer purchasing power, and overall economic conditions, have an impact on diamond prices. Changes in these factors may influence the pricing trends for diamonds.

## 4.0 Product Structure

    DiamondPricePrediction
    ├── README.md
    ├── application.py
    ├── artifacts
    │   ├── model.pkl
    │   ├── preprocessor.pkl
    │   ├── raw.csv
    │   └── train.csv
    ├── logs
    │   ├── 05_17_2023_12_55_12.log
    │   └── 05_17_2023_12_55_14.log
    ├── notebooks
    │   ├── EDA.ipynb
    │   ├── Model Training.ipynb
    │   └── data
    │       └── gemstone.csv
    ├── reports
    │   └── figures
    │       ├── api.png
    │       ├── barplot.png
    │       ├── diamond-diamond.jpeg
    │       ├── heatmap.png
    │       ├── pricehist.png
    │       └── tablehist.png
    ├── requirements.txt
    ├── setup.py
    ├── src
    │   ├── components
    │   │   ├── __pycache__
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   └── model_trainer.py
    │   ├── exception.py
    │   ├── logger.py
    │   ├── pipelines
    │   │   ├── __pycache__
    │   │   ├── prediction_pipeline.py
    │   │   └── training_pipeline.py
    │   └── utils.py
    ├── templates
    │   ├── form.html
    │   └── index.html
    
## 5.0 Solution Strategy
My solution to solve this problem will be the development of a data science project. This project will have a machine learning model which can predict the price of a diamond based on provided features.

**Step 01. Data Description:** The missing values will be threated or removed. Finally, a initial data description will carried out to know the data. Therefore some calculations of descriptive statistics will be made, such as skewness, mean, median and standard desviation.

**Step 02. Feature Engineering:** In this section, a mind map will be created to assist the creation of the hypothesis and the creation of new features. These assumptions will help in exploratory data analysis and may improve the model scores.

**Step 03. Data Filtering:** Data filtering is used to remove columns or rows that are not part of the business. For example, columns with customer ID, hash code or rows with age that does not consist of human age.

**Step 04. Exploratory Data Analysis:** The exploratory data analysis section consists of univariate analysis, bivariate analysis and multivariate analysis to assist in understanding of the database.

**Step 05. Data Preparation:** In this section, the data will be prepared for machine learning modeling.

**Step 06. Feature Selection:** Here, I will select the best columns to be used for the training of the machine learning model. This reduces the dimensionality of the database and decreases the chances of overfiting.

**Step 07. Machine Learning Modeling:** Here, the aim to train the machine learning algorithms and how they can predict the data. For validation the model is trained, validated and applied to cross validation to know the learning capacity of the model.

**Step 08. Conclusions:** This is a conclusion stage which the generation capacity model is tested using unseen data. In addition, some business questions are answered to show the applicability of the model in the business context.

**Step 10. Model Deploy:** This is the final step of the data science project. So, in this step the flask api is created and the model and the functions are saved to be implemented in the api.

## 6.0 Top Data Insights

<img src="https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction/blob/master/reports/figures/pricehist.png">

Most diamonds falls under below 5000 price in the dataset and it is heavily sckewed on the right. 

<img src="https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction/blob/master/reports/figures/scatter.png">

The price are heavily correlated with price but table value has a random effect on the price. 

<img src="https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction/blob/master/reports/figures/heatmap.png">

x, y, and z variable are highly correlated with each other and with carat variable which indicate multicolinearity of the variable.  

<img src="https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction/blob/master/reports/figures/barplot.png">

Catergorical values distribution is not normal which might also be problematic especially incase of linear regression as the algorithms assume the normality of the data.

## 7.0 Machine Learning Applied
Here's all the results of the machine learning models with their default parameters.

| Model      | LinearRegression | Lasso     | Ridge     | Elasticnet |
| :---:      | :---:            | :---:     | :---:     | :---:      |   
| RMSE       | 1013.90          | 1013.87   | 1013.90   | 1533.41    |
| MAE        | 674.02           | 675.07    | 674.05    | 1060.73    |
| R2 score   | 93.68            | 93.68     | 93.68     | 85.56      |

The finaL model will be selected by program based on best R2_Score. Below there's a table with the capacity of the model to learn.

## 8.0 Run Application 

**Step-1:** Create a new envionment

    conda create -p venv python==3.8

**Step-2:** Install the necessary libraries

    pip install -r requirements.txt

**Step-3:** Clone the repository

    git clone https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction.git

**Step-4:** cd in to cloned folder

    cd Diamond_Price_Prediction

**Step-5:** Run application

    python application.py 

<img src="https://github.com/Bhardwaj-Saurabh/Diamond_Price_Prediction/blob/master/reports/figures/api.png">

## 9.0 Conclusions
The conclusion of the Diamond Price Prediction project showcases my expertise in data science and machine learning. Through this project, I successfully developed a predictive model that accurately estimates diamond prices based on various attributes such as carat size, cut quality, color, clarity, and more.

By leveraging advanced data analysis techniques, I was able to extract meaningful insights from the dataset and build a robust predictive model. The model demonstrated high accuracy in predicting diamond prices, enabling businesses in the diamond industry to make informed pricing decisions.

This project highlights my proficiency in data preprocessing, feature engineering, and model development using popular machine learning algorithms. I showcased strong skills in Python programming, data manipulation with libraries like pandas and NumPy, and implementing machine learning models with scikit-learn.

Furthermore, this project demonstrates my ability to work on real-world data science problems and deliver valuable solutions. It showcases my analytical thinking, problem-solving skills, and attention to detail. The successful outcome of the project strengthens my credentials and positions me as a capable data scientist ready to tackle complex business challenges.

# 10.0 Next Step
- Implement the api on the heroku plataform.
- Try to deploy the model in cloud platform.
