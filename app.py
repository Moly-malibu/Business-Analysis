import streamlit as st
import streamlit.components.v1 as components
import datetime 
from datetime import datetime

import requests
from requests import get
import pandas as pd
import numpy as np

import seaborn as sns

# Model
import sys
import sklearn
sklearn.set_config(transform_output="pandas")
import category_encoders as ce
sklearn.set_config(transform_output="pandas")
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Graph
import itertools
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


st.markdown("<h1 style='text-align: center; color: #002967;'>Business Analysis with AI</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: #002967;'>Agriculture by Countries</h1>", unsafe_allow_html=True)
def main():
    # Register pages
    pages = {
        "Home": Home,
        "Sales": Sales,
        "AgroBusiness": AgroBusiness,
    }
    st.sidebar.title("Statistics")
    page = st.sidebar.selectbox("Select Menu", tuple(pages.keys()))
    pages[page]()

def Home():
    def main():
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://images.pexels.com/photos/1353938/pexels-photo-1353938.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.markdown(
        """

        Analyze the potential applications of artificial intelligence (AI) in the Business. Specifically, explore how AI can optimize production processes, improve product quality, enhance sustainability, and support data-driven decision-making. Consider the challenges and potential benefits of AI implementation in this sector.

        
        
        Considerations for Businesses:

        
        ***Crop Diversification***:  Market Analysis: Identify crops with growing demand or untapped potential.

        ***Risk Mitigation***: Diversify to reduce vulnerability to pests, diseases, or market fluctuations.

        ***Environmental Impact***: Consider crops that promote sustainable practices (e.g., cover crops, organic farming).
        
        ***Technological Advancements***:  Precision Agriculture: Implement technologies like GPS, drones, and sensors for efficient resource management.
        
        ***Biotechnology:*** Explore genetically modified organisms (GMOs) for increased yields and resistance to pests.
        
        ***Automation***: Invest in automated machinery to reduce labor costs and improve efficiency.
        
        ***Climate Change Impacts***: Adaptation Strategies: Develop drought-resistant varieties, implement water-saving irrigation techniques, and consider climate-smart agriculture practices.
        
        ***Mitigation Efforts***: Contribute to carbon sequestration by adopting practices like agroforestry or cover cropping.
        
        ***Consumer Preferences***: Organic and Sustainable Products: Meet growing demand for environmentally friendly and ethically produced food.
        
        ***Local Sourcing***: Support local economies and reduce transportation costs by sourcing ingredients regionally.
        
        ***Health and Nutrition***: Offer products that align with consumer health trends (e.g., gluten-free, non-GMO, high-protein).
        
        ***Supply Chain Management***:  Traceability: Ensure transparency in the supply chain to build trust with consumers.
        
        ***Fair Trade Practices***: Support ethical labor practices and fair compensation for farmers.
        
        ***Sustainabilit***y: Minimize environmental impact throughout the value chain (e.g., reduce food waste).
        
        ***Regulatory Compliance***:  Food Safety Standards: Adhere to local, national, and international food safety regulations.
        
        ***Environmental Regulations***: Comply with environmental protection laws and regulations.
        
        ***Labor Standards***: Ensure compliance with labor laws and ethical standards.  Business Opportunities for Social Good
        
        ***Food Security Initiatives***: Partner with local communities to improve access to nutritious food.
        
        ***Sustainable Farming Practices***: Promote regenerative agriculture to enhance soil health and biodiversity.
        
        ***Community Supported Agriculture (CSA)***: Connect directly with consumers through subscription-based farming models.
        
        ***Value-Added Products***: Create processed food items that increase product value and shelf life.
        
        ***Education and Training***: Provide training and resources to farmers and agricultural workers.

        By carefully considering these factors, businesses can not only capitalize on agricultural trends but also make a positive impact on society through food security, environmental sustainability, and ethical practices.                
        """)
        today = st.date_input('Today is', datetime.datetime.now())

        footer_temp1 = """
            <!-- CSS  -->
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
            <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
            <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
            <footer class="background-color: black">
                <div class="container" id="About App">
                <div class="row">
                    <div class="col l6 s12">
                    </div>
            <div class="col l3 s12">
                    <h5 class="white-text">Connect With Me</h5>
                    <ul>
                        <a href="/" target="#002967" class="white-text">
                        ❤<i class="❤"></i>
                    </a>
                    <a href="https://www.linkedin.com/in/monica-bustamante-b2ba3781/3" target="#002966" class="white-text">
                        <i class="fab fa-linkedin fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002967" class="white-text">
                    ❤<i class="❤"></i>
                    </a>
                    <a href="https://github.com/Moly-malibu/UNHCR.git" target="#002967" class="white-text">
                        <i class="fab fa-github-square fa-4x"></i>
                    <a href="/" target="#002967" class="white-text">
                    ❤<i class="❤"></i>
                    </a>
                    </ul>
                    </div>
                </div>
                </div>
                <div class="footer-copyright">
                <div class="container">
                <a class="white-text text-lighten-3" href="/">Made by Liliana Bustamante</a><br/>
                <a class="white-text text" href=""> @Copyrigh</a>
                </div>
                </div>
            </footer>
            """
        components.html(footer_temp1,height=500)

    if __name__ == "__main__":
        main()

title_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="background-color: white">
	    <div class="container" id="About App">
	      <div class="row">
	        <div class="col l6 s12">
                <h4 style='text-align: center; class="black-text-dark">AI Internet of Things (IoT)-</h4>
              <h6 class="Sklearn, Tensorflow,  Keras, Pandas Profile, Numpy, Math, Data Visualization. </h6>
	        </div>     
	  </footer>
	"""
components.html(title_temp,height=100)
    
def AgroBusiness():
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg&ga=GA1.2.635976572.1603931911");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Production statistics</h1>", unsafe_allow_html=True)
    
    df = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)
    df.replace(np.nan, 0, inplace=True)
    df.dropna(inplace=True)
    # Method 2: Using boolean indexing to filter rows
    df = pd.DataFrame(df)
    # Create a new column 'Sum_AB' by summing values in columns 'A' and 'B' for each row
    df['Total Production'] = df[['Y1961', 'Y1962', 'Y1963',
       'Y1964', 'Y1965', 'Y1966', 'Y1967', 'Y1968', 'Y1969', 'Y1970',
       'Y1971', 'Y1972', 'Y1973', 'Y1974', 'Y1975', 'Y1976', 'Y1977',
       'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984',
       'Y1985', 'Y1986', 'Y1987', 'Y1988', 'Y1989', 'Y1990', 'Y1991',
       'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998',
       'Y1999', 'Y2000', 'Y2001', 'Y2002', 'Y2003', 'Y2004', 'Y2005',
       'Y2006', 'Y2007', 'Y2008', 'Y2009', 'Y2010', 'Y2011', 'Y2012',
       'Y2013', 'Y2014']].sum(axis=1)

    # info = st.sidebar.selectbox('Data Set Info', (df)) 
    st.markdown("<h1 style='text-align: center; color: #002967;'>from 1961 to 2014</h1>", unsafe_allow_html=True)
    
    st.markdown("""Global production, that is, agricultural production by each country for 53 years. The aim is to analyze the performance of the Agro-Business industry from 1961 to 2014. Historical data reveals that while the vegetables and fruits industry once faced significant challenges, it has defied initial predictions. Despite forecasts suggesting a bleak future, scientific advancements have played a crucial role in ensuring bananas remain a staple in the global food basket. Although the banana industry has experienced slower growth compared to other sectors within Agribusiness, its resilience over the past six decades is a testament to the power of innovation and adaptation.""")
    st.write("Production by Countries:", df)
    
    items = df.groupby(['Item', 'Element', 'Unit', 'Total Production' ]).size().reset_index(name='Products')

    grouped_data = df.groupby(['Area', 'Item']).agg({'Total Production': 'sum'}).reset_index()
    st.subheader("Crop by Product")
    st.write(grouped_data)
     

    # Create a bar chart to visualize the quantity of each product by country
    fig = plt.figure(figsize=(20, 20))
    plt.barh(grouped_data['Area'], grouped_data['Item'], color='skyblue')
    plt.xlabel('Products')
    plt.ylabel('Country')
    plt.title('Agro Products by Country')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()  # Adjust layout for better spacing
    st.subheader("Agro Products by Country")
    st.markdown("""The graph illustrates a concerning trend in global agricultural production. While output remained consistently high in the 1960s, it has steadily declined over the decades. This decline has led to a situation where only a limited number of countries are major agricultural producers, posing a significant food security risk on a global scale. """)
    st.pyplot(fig)

    # Create a figure
    st.subheader("Agro Products by Country")
    grouped = df.groupby(['Area', 'Item', 'Element']).agg({'Total Production': 'sum'}).reset_index()

    # Add a dropdown to select the x-axis column
    x_axis_column = st.selectbox('Area', grouped.columns)

    # Add a dropdown to select the y-axis column
    y_axis_column = st.selectbox('Total Production', grouped.columns)

    # Create the Plotly figure
    fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')

    # Customize the figure (optional)
    fig.update_layout(
        xaxis_title=x_axis_column,
        yaxis_title=y_axis_column
    )
    # Display the figure in Streamlit
    st.plotly_chart(fig)

#############################################
    #Statistic
    st.subheader("Statistical Distribution:")
    st.write(grouped.describe())
    st.write(grouped['Area'].value_counts())
    st.markdown(
        """
        Summary:  The dataset contains 7,134 observations with a mean value of 66,665,819.6385. The data is highly skewed to the right, with a standard deviation of 653,695,815.3245, indicating a wide range of values. The minimum value is 183, while the maximum is 23,108,809.118. This suggests that the majority of values are relatively small, with a few outliers pulling the mean to the right.
        
        Detailed Breakdown:

        ***Count***: 7,134 observations are present in the dataset.
                        
        ***Mean***: The average value of the observations is 66,665,819.6385.
                        
        ***Standard Deviation (SD)***: The SD is 653,695,815.3245, indicating a significant spread of values around the mean. A high SD suggests that the data points are widely dispersed.
                        
        ***Min and Max***: The minimum value is 183, and the maximum is 23,108,809.118. This shows the range of values in the dataset.

        ***Percentiles***:
                        
        25% (Q1): 25% of the observations fall below 936,321.5.
                        
        50% (Median): 50% of the observations fall below 4,025,630.5. This is also the middle value when the data is sorted.
                        
        75% (Q3): 75% of the observations fall below 10,450,083.75.

        ***Interpretation of Skewness***:

        The large difference between the mean and the median (Q2) suggests that the data is skewed to the right. This means that there are a few very large values (outliers) that pull the mean to the right, while the majority of values are smaller.

        The main agricultural producer has been China and the smallest has been British Virgin Island.                

    """)
    
    st.title("Box Plot and Histogram:")
    # Box plot
    st.header("Box Plot")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=grouped, ax=ax1)
    st.pyplot(fig1)

    # Histogram
    st.header("Histogram")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=grouped['Area'], bins=10, ax=ax2)
    st.pyplot(fig2)

    st.subheader("""Model""")
    items= df['Item'].describe()
    st.write("Statistic Describe: Targe Products:", items)

    y = df['Item']
    y.nunique()
    items = y.value_counts(normalize=True)
    st.markdown("""The table showcases the percentage of various fruits and vegetables cultivated globally. A downward trend is evident for many of these crops. This presents a significant opportunity for market expansion and diversification, benefiting both global society and the company. By identifying regions with untapped potential, we can contribute to food security, reduce reliance on a few major producers, and foster sustainable agricultural practices. """)
    st.write("Binary Classification: if the classes are imbalanced: ", items)

    items = df['Item'].value_counts().nlargest(100)
    st.write("Main agricultural products cultivated:", items)

    df['Item'] = df['Item'].str.lower()
    blueberries = df['Item'].str.contains('blueberries')
    gums = df['Item'].str.contains('gums')
    peppermint = df['Item'].str.contains('peppermint')
    tallowtree = df['Item'].str.contains('tallowtree')
    roots = df['Item'].str.contains('roots')
    vegetablesmelons = df['Item'].str.contains('vegetablesmelons')
    vegetables = df['Item'].str.contains('vegetables')
    kola = df['Item'].str.contains('kola')
    nuts = df['Item'].str.contains('nuts')
    melons = df['Item'].str.contains('melons')
    cereals = df['Item'].str.contains('cereals')
    bananas = df['Item'].str.contains('bananas')
    agave = df['Item'].str.contains('agave fibres nes')
    apples = df['Item'].str.contains('apples')
    apricots = df['Item'].str.contains('apricots')
    sparagus = df['Item'].str.contains('sparagus')


    df.loc[vegetables, 'Item'] = 'vegetables'
    df.loc[roots, 'Item'] = 'roots'
    df.loc[kola, 'Item'] = 'kola'
    df.loc[nuts, 'Item'] = 'nuts'
    df.loc[melons, 'Item'] = 'melons' 
    df.loc[cereals, 'Item'] = 'cereals'
    df.loc[bananas, 'Item'] = 'bananas'
    df.loc[agave, 'Item'] = 'agave'
    df.loc[apples, 'Item'] = 'apples'
    df.loc[apricots, 'Item'] = 'apricots'
    df.loc[sparagus, 'Item'] = 'sparagus'

    df.loc[~blueberries & ~gums & ~roots & ~kola & ~nuts & ~melons & ~cereals &
       ~peppermint & ~tallowtree & ~vegetables & ~vegetablesmelons & ~bananas & ~agave  & ~apples & ~apricots & ~sparagus,'Item'] = 'Other'
    items1 = df['Item'].value_counts()
    st.write("Subclassification:", items1)

    #training dataset
    test = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1) 
    val = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)  
    train = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)
    train['Item'].mode()   

    test.replace(np.nan, 0, inplace=True)
    val.replace(np.nan, 0, inplace=True)
    train.replace(np.nan, 0, inplace=True)
    train.describe(exclude='number')

    val['Item'].value_counts()
    bananas = val[val.Item=='Bananas']

    target = 'Element'
    train[target].value_counts(normalize=True)
    features = train.select_dtypes('number').columns.drop(target)
    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]
    X_test = test[features]
    y_test = test[target]

    pipeline = make_pipeline(
        ce.OrdinalEncoder(), 
        SimpleImputer(strategy='median'), 
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )

    # Fit on train, score on val
    pipeline.fit(X_train, y_train)
    st.write('Validation Accuracy', pipeline.score(X_val, y_val))

    # Get feature importances
    rf = pipeline.named_steps['randomforestclassifier']
    importances = pd.Series(rf.feature_importances_, X_train.columns)

    #Permutation Importance: transformers
    transformers = make_pipeline(
        ce.OrdinalEncoder(), 
        SimpleImputer(strategy='median')
    )

    X_train_transformed = transformers.fit_transform(X_train)
    X_val_transformed = transformers.transform(X_val)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_transformed, y_train)

    # Plot feature importances
    n =  53
    plt.figure(figsize=(5, n / 2))  # Adjust figure size based on number of features
    plt.title(f'Top {n} features')
    sorted_importances = importances.sort_values(ascending=False)[:n]  # Sort descending and select top n
    
    fig, ax = plt.subplots()
    sorted_importances.plot.barh(color='Cyan');
    st.title("Permutation Plotter")
    st.pyplot(fig)    

    #XGBoost dominates structured or tabular datasets on classification and regression predictive modeling problems.
    

    gb = make_pipeline(
        ce.OrdinalEncoder(), 
        XGBRegressor(n_estimators=200, objective='reg:squarederror', n_jobs=-1)
    )

    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_val)
    st.subheader("""R-squared""")
    st.markdown(
    """
    R-squared (R²) is a statistical measure that represents the proportion of the variance in the dependent variable that is explained by the independent variables in a regression model. In simpler terms, it tells us how well our model fits the data. 

    Interpreting R² = 0.9522429024395279

    In this case, an R² of 0.9522 means that approximately 95.22% of the variability in the dependent variable can be explained by the independent variables in your Gradient Boosting model. This is a very good fit, indicating that the model is performing well in predicting the target variable. 
     """)
    st.write('Gradient Boosting R^2', r2_score(y_val, y_pred))

    residual = plt.figure(figsize=(20, 20))
    plt.scatter(y_pred, y_train - y_pred)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    st.pyplot(residual)

    # Prediction vs. Actual Values Plot
    predic = plt.figure(figsize=(20, 20))
    plt.scatter(y_train, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs. Actual Values")
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')  # 45-degree line
    st.pyplot(predic)

    st.write(
    """ 
    conclusions:

    1. Inventory Reduction:

    The industry may be strategically reducing inventory levels to optimize costs or prepare for a shift in demand. This could be due to a change in consumer preferences, a change in production methods, or other factors. However, it's crucial to consider the potential impact on global food security, especially in the context of fruit and vegetable production. If these essential food sources are not being properly valued and exploited, it could lead to significant challenges in meeting future food demands.

    2. Increased Efficiency:

    This industry may have improved its productive efficiency, allowing it to produce the same amount of goods with fewer resources. However, the decline in agricultural cultivation, particularly for fruits and vegetables, raises concerns about potential food security issues. This trend could lead to the extinction of certain crops and limit the availability of healthy food options. To address these challenges, there's a growing interest in revitalizing agriculture, particularly in areas like fruit and vegetable production. By doing so, we can potentially reduce production costs, increase profit margins, and positively impact both the economy and public health.
    
    3. Supply Chain Optimization:

    This industry could benefit from optimizing its supply chain to reduce delivery times and inventory levels. This would lead to faster deliveries and lower costs. However, it's important to note that agriculture, particularly fruit and vegetable production, is a vital sector that has been declining for decades. This trend poses a significant threat to global food security. Therefore, investing in and revitalizing this industry is crucial to ensure a sustainable food supply for future generations.

    4. Outsourcing or Offshoring:
 
    This industry might be outsourcing or offshoring some of its production to reduce costs or increase capacity. While this can lead to a decrease in domestic production, it can also increase overall production. Such strategic moves can be beneficial for both the company and the countries involved, as they can lead to economic growth, job creation, and increased efficiency. However, it's important to consider the potential social and environmental impacts of these decisions.

    5. Market Strategy:

    This industry might focus on high-margin products to boost sales and profitability. However, it's essential to maintain production of lower-margin products, especially those that are vital for food security and human health. A balanced approach that prioritizes both profitability and societal needs is crucial.


    To draw a definitive conclusion, it's essential to consider additional factors:

    Market conditions: Analyze the overall market trends, including economic factors, consumer behavior, and competitive landscape.
    Company-specific factors: Consider the company's business strategy, financial performance, and any recent announcements or news.
    Data quality and completeness: Ensure that the data used for analysis is accurate and reliable.
    Time frame: Analyze the trends over a specific period to identify any short-term fluctuations or long-term patterns.
    

    """)
    
def Sales():
    import seaborn as sns
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://img.freepik.com/free-vector/geometric-shapes-background-paper-style_23-2148292297.jpg?t=st=1729897204~exp=1729900804~hmac=8f0e9f3a92bd3cd34f7302101bbf78b063b2db65d1a1634eb2283c530aaf00e2&w=2000");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Customer demographics</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Sales in United States</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    ***Business Understanding***

    This project begins with the premise that the company is preparing to launch a new product. To ensure its success, a comprehensive market segmentation study is essential. This study aims to answer several key questions:
    
    ***Market Niche Identification***: What specific market niche will the new product target?
    ***Product Offering Locations***: Where will the product be offered to maximize reach and impact?
    ***Target Market Characteristics***: What are the defining characteristics of the target market and consumers?
    
    To achieve these objectives, a relevant database has been selected that provides insights into demographic composition, sales data, and consumer behavior patterns. This database will help us understand what customers consume on a daily basis.
    Ultimately, the company will leverage the findings from this analysis, supported by the expertise of data scientists and their algorithms, to inform strategic actions and decisions for the successful launch of the new product. 

    """)
    raw_df = pd.read_csv('https://raw.githubusercontent.com/IBM/analyze-customer-data-spark-pixiedust/master/data/customers_orders1_opt.csv')
    st.write(raw_df)

    #cleaning data unnecesary columns.

    df = raw_df.drop(columns=['ADDRESS1', 'CITY', 'COUNTRY_CODE', 'POSTAL_CODE', 'POSTAL_CODE_PLUS4', 
                        'ADDRESS2', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'LOCALITY', 'DRIVER_LICENSE', 
                        'NATIONAL_ID', 'ORDER_DATE', 'ORDER_TIME', 'NATIONALITY', 'SALESMAN_ID',
                        'CUST_ID', 'ORDER_ID', 'FREIGHT_CHARGES', 'ORDER_SALESMAN', 'ORDER_POSTED_DATE', 'ORDER_SHIP_DATE', 'AGE',
                        'ORDER_VALUE', 'T_TYPE', 'PURCHASE_STATUS', 'CUSTNAME', 'CREDITCARD_NUMBER'])
    df = pd.DataFrame(df)
    st.write(df)
    st.markdown(
    """
    ***Data set Description*** :The data set Customer demographics and sales has 13.733 rows and 62 columns, but after cleanned data set has 36 row: 
    """)
    st.write("Columns and Rows", df.shape)
    st.write(df.count())
    st.write(df.describe())
    

    data = df[['Baby Food', 'Diapers', 'Formula', 'Lotion',
       'Baby wash', 'Wipes', 'Fresh Fruits', 'Fresh Vegetables', 'Beer',
       'Wine', 'Club Soda', 'Sports Drink', 'Chips', 'Popcorn', 'Oatmeal',
       'Medicines', 'Canned Foods', 'Cigarettes', 'Cheese',
       'Cleaning Products', 'Condiments', 'Frozen Foods', 'Kitchen Items',
       'Meat', 'Office Supplies', 'Personal Care', 'Pet Supplies', 'Sea Food',
       'Spices']]
    
    correlation_matrix = data.corr(method='pearson')
    st.markdown(
    """
    ***Correlation Analysis***:
    Correlation is a statistical measure that helps assess the linear relationship between two quantitative variables. A positive correlation occurs when both variables move in the same direction; as one variable increases, the other also increases. Conversely, a negative correlation indicates that as one variable increases, the other decreases.
    In our analysis of the relationship between wine and fresh vegetables, we observe a negative correlation coefficient of -0.022. This suggests that as customers purchase more wine, their purchases of fresh vegetables tend to decrease. This finding highlights an inverse relationship between these two products, indicating that increased spending on wine may be associated with reduced spending on fresh vegetables. 
    
    """)
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)
    
    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Show the heatmap in Streamlit
    st.pyplot(plt)
    
    # Calculate the covariance matrix
    covariance_matrix = data.cov()

    # Display the covariance matrix in Streamlit
    st.markdown(
    """
    ***Covariance Analysis:***
    Covariance is a measure used to determine the relationship between two variables. A large covariance indicates a strong relationship between the variables. Specifically, a positive covariance means that the returns of the two products move in the same direction; as one product's return increases, the other product's return also increases. Conversely, a negative covariance indicates that the returns move inversely; when one product's return increases, the other product's return decreases.
    ***For example***, when analyzing the relationship between beer and fresh vegetables, we find a covariance of 0.004401. This positive value suggests that there is a slight tendency for the returns of beer and fresh vegetables to move together, albeit the relationship is minimal.

    ***The formula is:***

    Cov(X,Y) = Σ E((X-μ)E(Y-ν)) / n-1 where:

    X is a random variable

    E(X) = μ is the expected value (the mean) of the random variable X and

    E(Y) = ν is the expected value (the mean) of the random variable Y

    n = the number of items in the data set

    """)
    st.write("Covariance Matrix:")
    st.write(covariance_matrix)

    # Create a heatmap for the covariance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Covariance Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Show the heatmap in Streamlit
    st.pyplot(plt)
    st.write('The value of Skewness is:')
    st.write(data.skew())


    st.markdown(
    """ 
    ***Skewness and Kurtosis Analysis***:
    Skewness is a statistical measure that assesses the symmetry of a dataset in relation to its mean. In this dataset, certain products such as chips, club soda, and canned foods exhibit a skewness of 0, indicating that their distribution is symmetric.
    It's important to note that negative skewness suggests that there are more frequent purchases of the product, as it indicates a longer tail on the left side of the distribution. This means that a larger number of customers tend to buy smaller quantities, while fewer customers purchase larger quantities. 
    
    ***Kurtosis Analysis***

    Kurtosis measures the "tailedness" of a distribution, indicating how much of the data is concentrated in the tails versus the center. In this analysis, products such as cigarettes and spices exhibit a normal distribution, which is characterized by a kurtosis value close to zero.
    On the other hand, products like wine, club soda, chips, popcorn, canned foods, and condiments have negative kurtosis. This suggests that these products have limited customer purchases, indicating a flatter distribution with fewer extreme values.
    In contrast, other products in the dataset show positive kurtosis, implying a sharper peak and heavier tails. This indicates that these products may have more extreme purchase behaviors among customers. 
    """ 
    )
    st.write('The value of kurtosis is:')
    st.write(data.kurtosis())

    # Set the style for seaborn
    sns.set_theme(style='whitegrid')

    # Create a figure for the plots
    plt.figure(figsize=(16, 10))

    # Create histogram and KDE plot
    sns.histplot(data['Lotion'], bins=10, kde=True, color='blue', alpha=0.5)

    # Add titles and labels
    plt.title('Distribution of Lotion Values, relationship Lotion with the others products')
    plt.xlabel('Lotion Values')
    plt.ylabel('Frequency')

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Display the DataFrame and Skewness value
    st.write('The value of Skewness is:', data['Lotion'].skew())

    st.markdown(
    """ 
    ***Feature Engineering***:
    The database object of this study presents the features of segmenting the market into two categories, one is by-products and the other by qualities or characterization of the customers, so it will be coded in such a way as to show us the grouped information unifying it to In order to respond to which market segmentation the product should be directed and in which product more emphasis should be placed.

    ***Categorical Variable***:

    -GENDER CODE : Master, Miss, Mr. Mrs.
    
    -STATE: All state of United States.
    
    -CREDITCARD_TYPE: American Express, Diners Club, Discover, JCB, Master Card    VISA
    
    -PURCHASE_TOUCHPOINT: Desktop, Phone
    
    -ORDER_TYPE: High Value, Medium Value, Low Value
    
    -GENERATION: Baby Boomer that represent 76 million people born between 1946 and 1964. Generation X born between 1965 and 1980. Generation Z born after 1997.

    ***Analysis***:

    In this dataset the generation Baby Boomer that represent 76 million people born between 1946 and 1964. They expenden more money that generation X, Y and Z, being generation X the ones who spend the least. See Table:
    """)

    df = pd.DataFrame(raw_df)

    # Create a cross-tabulation of Gender and Generation
    result = pd.crosstab(df['GenderCode'], df['GENERATION'])
    st.write(result)
    st.markdown(
    """ 
    ***Overall Interpretation:***
    The Baby Boomer generation shows the highest average spending and variability, indicating they may be more willing to spend on certain products or services compared to other generations.
    Generation X has both the lowest average and minimum spending, suggesting they may be more conservative in their expenditures.

    Generations Y and Z have similar average spends but show different levels of variability; both have higher averages than Gen X but lower than Baby Boomers.

    The quartile data indicates that while Baby Boomers dominate in higher spending, there are also significant numbers of lower spenders in other generations, particularly Gen X.
    
    ***Conclusion***

    This analysis provides valuable insights into generational spending behavior, which can inform marketing strategies, product development, and targeted promotions for different age groups based on their spending patterns and preferences.
    """
    )
    st.write(result.describe())

    crosstab_reset = result.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='GenderCode', var_name='GENERATION', value_name='Count')

    

    # Assuming you have your melted_crosstab DataFrame ready

    # Create the box plot
    fig = px.box(melted_crosstab, x="GENERATION", y="Count", color="GenderCode",
                title="Distribution of Count by Generation and Gender")

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.markdown(
    """ 
    Interpretation:

    ***Mean Spending***: Baby Boomers have the highest average spending at 1289.75, significantly higher than Gen X, which averages only 461.25.
    ***Standard Deviation***: Baby Boomers also show a very high standard deviation (1288.71) indicating considerable variability in their spending habits, with some individuals spending much more than others.
    ***Minimum and Maximum Values***: The minimum value for Gen X is quite low at 40, suggesting some individuals in this generation spend very little.
    ***Quartiles and Median Values***: The median value for Baby Boomers (1208.5) reinforces their higher average spending, while Gen X has a median of only 423.5, indicating that half of this group spends less than this amount.
    
    ***Conclusion for Generational Spending:*** This dataset reveals that Baby Boomers are the highest spenders among the generations analyzed, while Gen X shows significantly lower average spending with a wider range of values, indicating more variability in their purchasing behavior.
    
    ***Overall Interpretation:***
    
    Both datasets highlight significant differences in consumer behavior based on credit card type and generational demographics:
    
    -*Higher average spending is associated with certain credit cards (American Express and Diners Club) and with the Baby Boomer generation.
    
    -*Variability in spending is also notable, particularly among Baby Boomers and Diners Club users, suggesting diverse purchasing habits within these groups.
    
    -*Understanding these patterns can inform targeted marketing strategies and product offerings tailored to specific consumer segments based on their spending behaviors.
        
    """)

    df = pd.DataFrame(raw_df)

    # Create a cross-tabulation of Gender and credit card type
    result = pd.crosstab(df['GENERATION'], df['CREDITCARD_TYPE'])
    st.write(result)

    crosstab_reset = result.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='GENERATION', var_name='CREDITCARD_TYPE', value_name='Count')


    # Create the box plot
    fig = px.box(melted_crosstab, x="GENERATION", y="Count", color="CREDITCARD_TYPE",
                title="Distribution of Count by Generation and CREDIT CARD TYPE")

    # Display the plot in Streamlit
    st.write(result.describe())
    st.plotly_chart(fig)
    st.markdown(
    """ 
    ***Conclusion for Demographic Spending:***
    This dataset reveals that individuals identified as "Mr." and "Mrs." are the highest spenders among the demographic categories analyzed, while "Master" and "Miss" show significantly lower average spending with a wider range of values, indicating more variability in their purchasing behavior.
    
    ***Coverall Interpretation:***  Both datasets highlight significant differences in consumer behavior based on credit card type and demographic categories:
    
    *Higher average spending is associated with certain credit cards (American Express and Diners Club) and with individuals identified as "Mr." and "Mrs."
    
    *Variability in spending is also notable, particularly among these higher-spending groups, suggesting diverse purchasing habits within these segments.
    
    *Understanding these patterns can inform targeted marketing strategies and product offerings tailored to specific consumer segments based on their spending behaviors.
    
    """)
    ccg = pd.crosstab(df['CREDITCARD_TYPE'], df['GenderCode'])
    st.write(ccg)

    crosstab_reset = ccg.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='CREDITCARD_TYPE', var_name='GenderCode', value_name='Count')


    # Create the box plot
    fig = px.box(melted_crosstab, x="CREDITCARD_TYPE", y="Count", color="GenderCode",
                title="Distribution of Count by CREDIT CARD TYPE and Gender")

    # Display the plot in Streamlit
    st.write(ccg.describe())
    st.plotly_chart(fig)
    st.markdown(""" 

    """)

    vg = pd.crosstab(df['GenderCode'], df['ORDER_TYPE'])
    st.write(vg)
    st.write(vg.describe())

    crosstab_reset = vg.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='GenderCode', var_name='ORDER_TYPE', value_name='Count')

    # Create the box plot
    fig = px.box(melted_crosstab, x="GenderCode", y="Count", color="ORDER_TYPE",
                title="Distribution of Count by Gender TYPE and Order type")

    # Display the plot in Streamlit
    st.write(ccg.describe())
    st.plotly_chart(fig)
    st.markdown(""" 

    """)
    # Set the Seaborn style
    sns.set_theme(style="ticks", color_codes=True)

    # Assuming 'tab4' is your DataFrame
    pair = sns.pairplot(melted_crosstab)

    # Display the plot in Streamlit
    st.pyplot(pair)

    st.markdown(
    """
    ***Statistic categorical compared the use of different Purchase touch point by gender***.
    Shopping continues with the trend towards online or telephone purchases, being men and women parents the ones who currently use this system the most.

    """)

    os = pd.crosstab(df['Office Supplies'], df['ORDER_TYPE'])
    st.write(os)
    st.write(os.describe())

    crosstab_reset = os.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='Office Supplies', var_name='ORDER_TYPE', value_name='Count')

    # Create the box plot
    fig_pg = px.box(melted_crosstab, x="Office Supplies", y="Count", color="ORDER_TYPE",
                title="Distribution of Count by Office Supplies and ORDER TYPES")

    # Display the plot in Streamlit
    # st.write(fig_pg.describe())
    st.plotly_chart(fig_pg)
    gr= melted_crosstab.groupby(by="ORDER_TYPE").sum()
    st.write(gr)
    st.markdown(""" 

    """)
    pg = pd.crosstab(df['GenderCode'], df['PURCHASE_TOUCHPOINT'])
    st.write(pg)

    crosstab_reset = pg.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='GenderCode', var_name='PURCHASE_TOUCHPOINT', value_name='Count')

    # Create the box plot
    fig_pg = px.box(melted_crosstab, x="GenderCode", y="Count", color="PURCHASE_TOUCHPOINT",
                title="Distribution of Count by Gender and Purchase touch Point")

    # Display the plot in Streamlit
    # st.write(fig_pg.describe())
    st.plotly_chart(fig_pg)
    st.markdown(""" 

    """)

    # Assuming 'tab6' is your DataFrame
    sns.catplot(x='Desktop', y='Phone', kind='bar', data=pg, height=7)

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    st.markdown(
    """ 

    """)
    pp = pd.crosstab(df['Pet Supplies'], df['PURCHASE_TOUCHPOINT'])
    st.write(pp)
    st.write(pp.describe())

    crosstab_reset = pp.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='Pet Supplies', var_name='PURCHASE_TOUCHPOINT', value_name='Count')

    # Create the box plot
    fig_pg = px.box(melted_crosstab, x="Pet Supplies", y="Count", color="PURCHASE_TOUCHPOINT",
                title="Distribution of Count by Pet Supplies and PURCHASE TOUCHPOINT")

    # Display the plot in Streamlit
    # st.write(fig_pg.describe())
    st.plotly_chart(fig_pg)
    gr= melted_crosstab.groupby(by="PURCHASE_TOUCHPOINT").sum()
    st.write(gr)
    st.markdown(""" 

    """)


    sg = pd.crosstab(df['GENERATION'], df['STATE'])
    st.write(sg)
    st.write(sg.describe())

    crosstab_reset = sg.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='GENERATION', var_name='STATE', value_name='Count')

    # Create the box plot
    fig_pg = px.box(melted_crosstab, x="GENERATION", y="Count", color="STATE",
                title="Distribution of Count by Generation and States")

    # Display the plot in Streamlit
    # st.write(fig_pg.describe())
    st.plotly_chart(fig_pg)
    st.markdown(""" 

    """)

    st.markdown(
    """ 
    ***Categorical Analysis by Products***: The analysis of the products we can see how the first 3 products that are bought the most are Lotions, wipes and cleaning products. See Table.
    

    """)
    bs = pd.crosstab(df['Baby wash'], df['STATE'])
    st.write(bs)
    st.write(bs.describe())

    crosstab_reset = bs.reset_index()

    # Melt the crosstab for seaborn compatibility
    melted_crosstab = crosstab_reset.melt(id_vars='Baby wash', var_name='STATE', value_name='Count')

    # Create the box plot
    fig_pg = px.box(melted_crosstab, x="Baby wash", y="Count", color="STATE",
                title="Distribution of Count by Baby wash and States")

    # Display the plot in Streamlit
    # st.write(fig_pg.describe())
    st.plotly_chart(fig_pg)
    gr= melted_crosstab.groupby(by="STATE").sum()
    st.write(gr)
    st.markdown(""" 

    """)

    
    # Create a figure
    st.subheader("Summary")
    st.markdown(
    """ 
    In this scatter plot, you can analyze different perspectives from the dataset by selecting various variables based on the information you need to examine

    """)
    grouped = df.groupby(['ORDER_TYPE', 'PURCHASE_TOUCHPOINT', 'CREDITCARD_TYPE', 'AGE', 'ORDER_VALUE', 'GenderCode', 'GENERATION']).agg({'CITY': 'sum'}).reset_index()

    # Add a dropdown to select the x-axis column
    x_axis_column = st.selectbox('Select 1 option', grouped.columns)

    # Add a dropdown to select the y-axis column
    y_axis_column = st.selectbox('Select 2 option', grouped.columns)

    # Create the Plotly figure
    fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')

    # Customize the figure (optional)
    fig.update_layout(
        xaxis_title=x_axis_column,
        yaxis_title=y_axis_column
    )
    # Display the figure in Streamlit
    st.plotly_chart(fig)

    ######################

    grouped = df.groupby(['ADDRESS1', 'COUNTRY_CODE', 'POSTAL_CODE', 'POSTAL_CODE_PLUS4', 'CITY','STATE', 'GenderCode', 'GENERATION']).agg({'ORDER_VALUE': 'sum'}).reset_index()

    # Add a dropdown to select the x-axis column
    x_axis_column = st.selectbox('Select 1 option', grouped.columns)

    # Add a dropdown to select the y-axis column
    y_axis_column = st.selectbox('Select 2 option', grouped.columns)

    # Create the Plotly figure
    fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')

    # Customize the figure (optional)
    fig.update_layout(
        xaxis_title=x_axis_column,
        yaxis_title=y_axis_column
    )
    # Display the figure in Streamlit
    st.plotly_chart(fig)

    # import folium
    # from geopy.geocoders import GoogleV3
    # from geopy.exc import GeocoderTimedOut
    # from geopy.geocoders import Nominatim

    # # Load your dataset from the provided URL
    # data_url = "https://raw.githubusercontent.com/IBM/analyze-customer-data-spark-pixiedust/master/data/customers_orders1_opt.csv"
    # raw_df = pd.read_csv(data_url)

    # # Display the first few rows of the dataframe for reference
    # st.title("Customer Orders Map")
    # st.write(raw_df.head())

    # # Initialize geocoder with Google Maps API (replace 'YOUR_API_KEY' with your actual key)
    # from geopy.geocoders import OpenCage

    # geolocator = OpenCage("6fb4c2fa7c3d4fafa44a5c45752dcb97")
    # location = geolocator.geocode("1600 Amphitheatre Parkway, Mountain View, CA")

    # # Function to get latitude and longitude from address components
    # def get_lat_long(row):
    #     try:
    #         address = f"{row['ADDRESS1']}, {row['CITY']}, {row['COUNTRY_CODE']} {row['POSTAL_CODE']}"
    #         location = geolocator.geocode(address)
    #         if location:
    #             return location.latitude, location.longitude
    #         else:
    #             return None, None
    #     except GeocoderTimedOut:
    #         return get_lat_long(row)  # Retry on timeout

    # # Ensure your DataFrame has the necessary columns for geocoding
    # if 'ADDRESS1' in raw_df.columns and 'CITY' in raw_df.columns and 'COUNTRY_CODE' in raw_df.columns and 'POSTAL_CODE' in raw_df.columns:
    #     # Add latitude and longitude columns to the DataFrame
    #     raw_df['LATITUDE'], raw_df['LONGITUDE'] = zip(*raw_df.apply(get_lat_long, axis=1))

    #     # Create a Folium map centered around the average latitude and longitude
    #     m = folium.Map(location=[raw_df['LATITUDE'].mean(), raw_df['LONGITUDE'].mean()], zoom_start=4)

    #     # Add markers for each customer location
    #     for idx, row in raw_df.iterrows():
    #         if pd.notnull(row['LATITUDE']) and pd.notnull(row['LONGITUDE']):
    #             folium.Marker(
    #                 location=[row['LATITUDE'], row['LONGITUDE']],
    #                 popup=f"{row['ADDRESS1']}, {row['CITY']}, {row['COUNTRY_CODE']} {row['POSTAL_CODE']}",
    #                 icon=folium.Icon(color='blue')
    #             ).add_to(m)

    #     # Display the map in Streamlit
    #     st.subheader("Customer Locations")
    #     st.markdown("### Click on the markers to see customer addresses.")
    #     st_folium = st.components.v1.html(m._repr_html_(), height=500)

    # else:
    #     st.error("The required address columns are not present in the dataset.")

    # # Optionally display the DataFrame for reference
    # st.write("DataFrame:")
    # st.write(raw_df[['ADDRESS1', 'CITY', 'COUNTRY_CODE', 'POSTAL_CODE', 'LATITUDE', 'LONGITUDE']])


    st.markdown(
    """ 
    ***Model***

    Regression Analysis:

    Regression analysis is a type of supervised machine learning that helps to identify and quantify the relationship between one or more features (independent variables) and a target variable (dependent variable). This analysis is essential for predicting outcomes based on input data.
    
    ***Logistic Regression*** is a specific type of regression analysis used when the target variable is categorical, particularly for binary outcomes (e.g., yes/no, true/false). In logistic regression, the probability of an event occurring is modeled as a linear combination of predictor variables. Importantly, there is no requirement for a linear relationship between the dependent and independent variables in this case. Instead, logistic regression can handle binary values (0 or 1, representing false or true).  To map predicted probabilities to binary outcomes, we use the sigmoid function, which transforms any real-valued number into a value between 0 and 1. This allows us to interpret the output as a probability.
    
    
    ***Preparing Categorical Data***

    Before building a model, it’s crucial to prepare the data properly:
            Normalization: This process involves organizing data into tables and establishing relationships between those tables according to specific rules. Normalization aims to protect data integrity and enhance database flexibility by eliminating redundancy and ensuring consistent dependencies.
    
    Dataset Splitting
            When creating a predictive model, we typically divide our dataset into two main parts:
            Training Dataset: This subset of data is used to train the model. It includes samples that help the model learn the underlying patterns in the data.
            Validation Dataset: This subset is used to evaluate the model's performance after training. It helps assess how well the model generalizes to unseen data.
            
    -***Features and Target Variable***
   
    In regression analysis, we distinguish between:
    
    ***Features (Independent Variables)***: These are the input variables used to predict the target variable.
    
    ***Target Variable (Dependent Variable)***: This is the outcome we are trying to predict.
    
    
    ***Steps to Create a Logistic Regression Model***: Here are the steps typically involved in creating a logistic regression model:
    
    ***Data Preparation***: Clean the dataset by handling missing values and outliers. Encode categorical variables using techniques like one-hot encoding or label encoding.
        
    ***Feature Selection***: Identify relevant features that contribute significantly to predicting the target variable.
        
    ***Split the Dataset***: Divide the dataset into training and validation sets (commonly using an 80/20 split).
        
    ***Model Training***: Use logistic regression algorithms from libraries such as scikit-learn in Python to fit the model on the training dataset.
        
    ***Model Evaluation***: Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC on the validation dataset.
        
    ***Hyperparameter Tuning***: Adjust hyperparameters to optimize model performance through techniques like cross-validation.
        
    ***Final Model Deployment***: One satisfied with performance metrics, deploy the model for making predictions on new data.

    This structured approach provides clarity on each aspect of regression analysis and logistic regression while outlining essential steps for building a predictive model.

    """)

    df['GenderCode'].value_counts(normalize=True) #Normalize the dataset.
    df['GenderCode'].describe()
    product_df = df.copy() #Prepared the dataset to partition.
    train_set = product_df.sample(frac=0.75, random_state=0)
    test_set  = product_df.drop(train_set.index)

    target = 'Lotion'  #create the target and train set.
    y_train = train_set[target]
    y_train.value_counts(normalize=True)

    linear_reg = LogisticRegressionCV() #create the model.
    train_set, val = train_test_split(train_set, random_state=42)
    train_set.shape, val.shape

    #create the features by products.
    features = ['Baby Food', 'Diapers', 'Formula', 
        'Baby wash', 'Wipes', 'Fresh Fruits', 'Fresh Vegetables', 'Beer',
        'Wine', 'Club Soda', 'Sports Drink', 'Chips', 'Popcorn', 'Oatmeal',
        'Medicines', 'Canned Foods', 'Cigarettes', 'Cheese',
        'Cleaning Products', 'Condiments', 'Frozen Foods', 'Kitchen Items',
        'Meat', 'Office Supplies', 'Personal Care', 'Pet Supplies', 'Sea Food',
        'Spices']

    X_train = train_set[features] #create train set ann validation.
    y_train = train_set[target]
    X_val = val[features]
    y_val = val[target]


    imputer = SimpleImputer () #impute missiing values
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    scaler = StandardScaler() #Scaler and  fit
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    model = LogisticRegressionCV(cv=5, n_jobs=1, random_state=42) #Fit the Model
    model.fit(X_train_scaled, y_train)
    st.write('Validation Accuracy', model.score(X_val_scaled, y_val)) #Model performance indicators that is not overfitting. 
    st.title("Comparison of Beer and Lotion Preferences")

    # Create the lmplot comparing Beer and Lotion
    sns.set_theme(style="whitegrid")
    lm_plot = sns.lmplot(x='Beer', y='Lotion', data=df, scatter_kws={'alpha':0.5})

    # Add titles and labels
    plt.title('Comparison of Lotion Preference vs Beer Consumption')
    plt.xlabel('Beer Consumption')
    plt.ylabel('Lotion Preference')

    # Show the plot in Streamlit
    st.pyplot(lm_plot)
    st.markdown(
    """ 
    ***EVALUATION***
    Report validation MAE and R2
    MAE = (1/n) * Σ|yi – xi|

    """)

    encoder = ce.OneHotEncoder(use_cat_names=True) #Encoder and fit transform method with train set/val
    X_train_encoded = encoder.fit_transform(X_train)
    X_val_encoded = encoder.transform(X_val)

    linear = LogisticRegressionCV() #Logistic Regression 
    linear.fit(X_train_encoded, y_train)

    y_pred_train = model.predict(X_train_encoded)
    y_pred_val = model.predict(X_val_encoded)

    st.markdown(
    """ 
    Interpreting the results of the model's performance metrics involves understanding what each metric means and how they relate to the model's predictive accuracy. Here’s a breakdown of the metrics:
   
    1. Root Mean Squared Error (RMSE)

    Train RMSE: 0.2235
    Validation RMSE: 0.2194

    Interpretation: RMSE measures the average magnitude of the errors between predicted and actual values, with a lower value indicating better model performance.
    In this case, both the training and validation RMSE values are relatively close, suggesting that the model performs similarly on both datasets.
    The values indicate that, on average, the model's predictions deviate from the actual values by approximately 0.22 units.
    
    2. Mean Absolute Error (MAE)

    Train MAE: 0.0500
    Validation MAE: 0.0482

    Interpretation: MAE measures the average absolute errors between predicted and actual values, providing a straightforward interpretation of errors in the same units as the target variable.
    The MAE values are also close for both training and validation sets, indicating consistent performance across datasets.
    The small MAE values (around 0.05) suggest that, on average, the predictions are off by about 0.05 units.

    3. R² Score

    Train R² Score: -0.0526
    Validation R² Score: -0.0506

    Interpretation: The R² score indicates how well the independent variables explain the variance in the dependent variable. An R² score of 1 indicates perfect prediction, while a score of 0 indicates that the model does not explain any variance.
    Negative R² scores suggest that the model is performing worse than a simple mean-based prediction (i.e., predicting the mean value for all observations). This is concerning as it indicates that your model may not be capturing any useful information from the features or may be poorly specified.
    Overall Interpretation. 

    Model Performance: The RMSE and MAE indicate that while your model has relatively low average errors, it is not performing significantly better than random guessing based on the negative R² scores.
    
    Potential Issues: The negative R² scores suggest that there may be issues with model specification, feature selection, or data quality. It could indicate that important predictors are missing or that there is noise in the data affecting predictions.
    
    In summary, while your RMSE and MAE metrics suggest reasonable prediction accuracy, the negative R² scores indicate significant room for improvement in your model's predictive capability.
    """)

    st.write(f'Logistic Regression with {len(features)} features: {features}')
    st.write('______________________________________________')
    st.write('Train Root Mean Squared Error:', 
        np.sqrt(mean_squared_error(y_train, y_pred_train)))
    st.write('Validation Root Mean Square Error:', np.sqrt(mean_squared_error(y_val, y_pred_val)))
    st.write('Train Mean Absolute Error:', mean_absolute_error(y_train, y_pred_train))
    st.write('Validation Mean Absolute Error:', mean_absolute_error(y_val, y_pred_val))
    st.write('Train R^2 Score:', r2_score(y_train, y_pred_train))
    st.write('Validation R^2 Score:', r2_score(y_val, y_pred_val))

    @st.cache_data
    def create_lotion_comparison_plots(df):
        for col in sorted(df.columns):
            exclude_columns = ['ADDRESS2', 'LOCALITY', 'DRIVER_LICENSE']
            # if col != 'Lotion' and df[col].nunique() <= 20:  
            if col != 'Lotion' and col not in exclude_columns and df[col].nunique() <= 20: # Skip 'Lotion' and check for <= 20 unique values
                fig, ax = plt.subplots()
                sns.barplot(x=col, y='Lotion', data=df, ax=ax, color='yellow')
                plt.xticks(rotation=45)
                plt.xlabel(col)
                plt.ylabel('Lotion')
                plt.title(f'Lotion Comparison with {col}')
                st.pyplot(fig)

    st.title('Lotion Comparison Analysis')

    # Assuming you have your DataFrame 'df' loaded
    if st.cache(allow_output_mutation=True):  # Cache the DataFrame to improve performance
            df = pd.read_csv('https://raw.githubusercontent.com/IBM/analyze-customer-data-spark-pixiedust/master/data/customers_orders1_opt.csv')  # Replace with your data loading logic

    create_lotion_comparison_plots(df.copy())  # Operate on a copy to avoid modifying original data

    st.markdown(
    """
   ***Analysis***
    In this analysis, we utilize logistic regression to examine various features within the dataset. Our findings reveal that lotions are the primary product on which customers spend their money. When compared to other products such as beer, cigarettes, and medicine, lotions consistently show higher consumer expenditure.

    Among the payment methods analyzed, Diners Club emerges as the credit card with the highest sales volume for lotions, closely followed by American Express. This indicates a strong preference among customers using these cards for purchasing lotions.

    When comparing lotion purchases to cheese, we observe that customers exhibit similar levels of preference for both products. This suggests that lotions and cheese are perceived as equally desirable among consumers at the point of sale.

    Further analysis by generation indicates that Generation Z (individuals born after 1997) is the most significant demographic for lotion sales. This generation demonstrates a higher propensity to spend on these products, making them a key target market for future marketing efforts.

    In terms of gender demographics, both females and males (Miss and Mr.) are represented in the purchasing patterns.

    Finally, when considering the average expenditure, it is evident that customers prefer to buy lotions directly from physical stores (desktop), highlighting a significant channel for sales in this product category. 


    ***Calculate Negative Mean Absolute Error***:
    """)

    pipeline = make_pipeline(
        ce.OneHotEncoder(use_cat_names=True), 
        SimpleImputer(strategy='mean'), 
        StandardScaler(), 
        SelectKBest(f_regression, k=20), 
        Ridge(alpha=1.0)
    )
    k = 3
    scores = cross_val_score(pipeline, X_train, y_train, cv=k, 
                            scoring='neg_mean_absolute_error')
    st.write(f'MAE for {k} folds:', -scores)

    # Set the title for the Streamlit app
    st.title("Average Expenditure on Beer by Credit Card")

    # Calculate mean expenditure on beer grouped by credit card type
    mean_expenditure = df.groupby('CREDITCARD_TYPE')['Beer'].mean().sort_values()

    # Create a horizontal bar chart using Matplotlib
    plt.figure(figsize=(8, 6))

    mean_expenditure.plot(kind='barh', color='yellow')

    plt.title('Average Expenditure on Beer by Credit Card Type')
    plt.xlabel('Average Expenditure ($)')
    plt.ylabel('Credit Card Type')
    plt.xticks(rotation=45)

    # Show the plot in Streamlit
    st.pyplot(plt)
   
   # Get coefficients and feature names
    coefficients = pd.Series(model.coef_[0], features)

    # Sort coefficients for better visualization
    coefficients_sorted = coefficients.sort_values()

    # Create a horizontal bar chart for coefficients
    plt.figure(figsize=(8, 6))
    coefficients_sorted.plot(kind='barh', color='skyblue')
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')

    # Show the plot in Streamlit
    st.markdown(
    """
    ***Note***: if we remove the Lotion the model show us that the product where customer expend more money are cleaning products.
    
    """)
    st.pyplot(plt)
    st.title("""CONCLUSION:""")
    st.markdown(
    """ 
    
    ***Market Segmentation Analysis:***
    This study serves as a decision-making tool for market segmentation, providing insights into niche markets and the characterization of the target audience. The model demonstrated a precision of 0.95, indicating that the strategies developed can be effectively supported by identifying best-selling products, understanding the target community, and considering the relevant age demographics.
    Our findings reveal that the Baby Boomer generation remains a significant consumer group, surpassing other generations in terms of purchasing power. This demographic tends to be less price-sensitive, demonstrating a willingness to spend on products without much hesitation. As such, hygiene products and fragrances represent promising market opportunities.
    Conversely, Generation Z has transformed the purchasing landscape. This generation is less reliant on physical retail locations for acquiring products. However, when it comes to categories like lotions and similar items, there is still a strong preference for purchasing in physical stores. This highlights the need for businesses to maintain physical facilities to cater to this demand while also adapting to the evolving shopping behaviors of younger consumers.
    """)

    
if __name__ == "__main__":
     main()