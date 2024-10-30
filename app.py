import streamlit as st
import streamlit.components.v1 as components 
from datetime import datetime
import datetime
import requests
from requests import get
import pandas as pd
import numpy as np


# Model
import sklearn
sklearn.set_config(transform_output="pandas")
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

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
    grouped = df.groupby(['Area', 'Item']).agg({'Total Production': 'sum'}).reset_index()

    # Add a dropdown to select the x-axis column
    x_axis_column = st.selectbox('Area', grouped.columns)

    # Add a dropdown to select the y-axis column
    y_axis_column = st.selectbox('Toal Production', grouped.columns)

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
    import seaborn as sns
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

    # Correlation matrix
    # correlation_matrix = df.corr()
    # fig3 = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # st.subheader("Correlation Matrix")
    # st.pyplot(fig3)

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
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor

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
    st.markdown("<h1 style='text-align: center; color: #002966;'>Customer demographics and sales in United States</h1>", unsafe_allow_html=True)

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
    
    








if __name__ == "__main__":
     main()