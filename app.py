import streamlit as st
import streamlit.components.v1 as components 
from datetime import datetime
import datetime
import requests
from requests import get
import pandas as pd
import numpy as np

import sklearn
sklearn.set_config(transform_output="pandas")
import itertools

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go


st.markdown("<h1 style='text-align: center; color: #002967;'>Business Analysis with AI</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #002967;'>Agriculture by Countries</h1>", unsafe_allow_html=True)
def main():
    # Register pages
    pages = {
        "Home": Home,
        # "AgroBusiness": AgroBusiness,
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
    
# def AgroBusiness():
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg&ga=GA1.2.635976572.1603931911");
#     background-size: cover;
#     }
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     st.markdown("<h1 style='text-align: center; color: #002966;'>Production statistics</h1>", unsafe_allow_html=True)
    
#     df = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)
#     df.replace(np.nan, 0, inplace=True)
#     df.dropna(inplace=True)
#     # Method 2: Using boolean indexing to filter rows
#     df = pd.DataFrame(df)
#     # Create a new column 'Sum_AB' by summing values in columns 'A' and 'B' for each row
#     df['Total Production'] = df[['Y1961', 'Y1962', 'Y1963',
#        'Y1964', 'Y1965', 'Y1966', 'Y1967', 'Y1968', 'Y1969', 'Y1970',
#        'Y1971', 'Y1972', 'Y1973', 'Y1974', 'Y1975', 'Y1976', 'Y1977',
#        'Y1978', 'Y1979', 'Y1980', 'Y1981', 'Y1982', 'Y1983', 'Y1984',
#        'Y1985', 'Y1986', 'Y1987', 'Y1988', 'Y1989', 'Y1990', 'Y1991',
#        'Y1992', 'Y1993', 'Y1994', 'Y1995', 'Y1996', 'Y1997', 'Y1998',
#        'Y1999', 'Y2000', 'Y2001', 'Y2002', 'Y2003', 'Y2004', 'Y2005',
#        'Y2006', 'Y2007', 'Y2008', 'Y2009', 'Y2010', 'Y2011', 'Y2012',
#        'Y2013', 'Y2014']].sum(axis=1)

#     # info = st.sidebar.selectbox('Data Set Info', (df)) 
#     st.markdown("<h1 style='text-align: center; color: #002967;'>from 1961 to 2014</h1>", unsafe_allow_html=True)
    
#     st.markdown("""Global production, that is, agricultural production by each country for 53 years. The aim is to analyze the performance of the Agro-Business industry from 1961 to 2014. Historical data reveals that while the banana industry once faced significant challenges, it has defied initial predictions. Despite forecasts suggesting a bleak future, scientific advancements have played a crucial role in ensuring bananas remain a staple in the global food basket. Although the banana industry has experienced slower growth compared to other sectors within Agribusiness, its resilience over the past six decades is a testament to the power of innovation and adaptation.""")
#     st.write("Production by Countries:", df)
    
#     items = df.groupby(['Item', 'Element', 'Unit', 'Total Production' ]).size().reset_index(name='Products')

#     grouped_data = df.groupby(['Area', 'Item']).agg({'Total Production': 'sum'}).reset_index()
#     st.subheader("Crop by Product")
#     st.write(grouped_data)
     

#     # Create a bar chart to visualize the quantity of each product by country
#     fig = plt.figure(figsize=(20, 20))
#     plt.barh(grouped_data['Area'], grouped_data['Item'], color='skyblue')
#     plt.xlabel('Products')
#     plt.ylabel('Country')
#     plt.title('Agro Products by Country')
#     plt.xticks(rotation=40, ha='right')
#     plt.tight_layout()  # Adjust layout for better spacing
#     st.subheader("Agro Products by Country")
#     st.markdown("""The graph illustrates a concerning trend in global agricultural production. While output remained consistently high in the 1960s, it has steadily declined over the decades. This decline has led to a situation where only a limited number of countries are major agricultural producers, posing a significant food security risk on a global scale. """)
#     st.pyplot(fig)

#     # Create a figure
#     st.subheader("Agro Products by Country")
#     grouped = df.groupby(['Area', 'Item']).agg({'Total Production': 'sum'}).reset_index()

#     # Add a dropdown to select the x-axis column
#     x_axis_column = st.selectbox('Area', grouped.columns)

#     # Add a dropdown to select the y-axis column
#     y_axis_column = st.selectbox('Toal Production', grouped.columns)

#     # Create the Plotly figure
#     fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')

#     # Customize the figure (optional)
#     fig.update_layout(
#         xaxis_title=x_axis_column,
#         yaxis_title=y_axis_column
#     )
#     # Display the figure in Streamlit
#     st.plotly_chart(fig)

# #############################################
#     #Statistic
#     st.subheader("Statistical Distribution:")
#     st.write(grouped.describe())
#     st.write(grouped['Area'].value_counts())
#     st.markdown("""Summary:

# The dataset contains 7,134 observations with a mean value of 66,665,819.6385. The data is highly skewed to the right, with a standard deviation of 653,695,815.3245, indicating a wide range of values. The minimum value is 183, while the maximum is 23,108,809.118. This suggests that the majority of values are relatively small, with a few outliers pulling the mean to the right.

# Detailed Breakdown:

# ***Count***: 7,134 observations are present in the dataset.
                
# ***Mean***: The average value of the observations is 66,665,819.6385.
                
# ***Standard Deviation (SD)***: The SD is 653,695,815.3245, indicating a significant spread of values around the mean. A high SD suggests that the data points are widely dispersed.
                
# ***Min and Max***: The minimum value is 183, and the maximum is 23,108,809.118. This shows the range of values in the dataset.

# ***Percentiles***:
                
# 25% (Q1): 25% of the observations fall below 936,321.5.
                
# 50% (Median): 50% of the observations fall below 4,025,630.5. This is also the middle value when the data is sorted.
                
# 75% (Q3): 75% of the observations fall below 10,450,083.75.

# ***Interpretation of Skewness***:

# The large difference between the mean and the median (Q2) suggests that the data is skewed to the right. This means that there are a few very large values (outliers) that pull the mean to the right, while the majority of values are smaller.

# The main agricultural producer has been China and the smallest has been British Virgin Island.                

#  """)
#     import seaborn as sns
#     st.title("Box Plot and Histogram:")
#     # Box plot
#     st.header("Box Plot")
#     fig1, ax1 = plt.subplots()
#     sns.boxplot(data=grouped, ax=ax1)
#     st.pyplot(fig1)

#     # Histogram
#     st.header("Histogram")
#     fig2, ax2 = plt.subplots()
#     sns.histplot(data=grouped['Area'], bins=10, ax=ax2)
#     st.pyplot(fig2)

#     # Correlation matrix
#     # correlation_matrix = df.corr()
#     # fig3 = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#     # st.subheader("Correlation Matrix")
#     # st.pyplot(fig3)

#     st.subheader("""Model""")
#     items= df['Item'].describe()
#     st.write("Statistic Describe: Targe Products:", items)

#     y = df['Item']
#     y.nunique()
#     items = y.value_counts(normalize=True)
#     st.markdown("""The table showcases the percentage of various fruits and vegetables cultivated globally. A downward trend is evident for many of these crops. This presents a significant opportunity for market expansion and diversification, benefiting both global society and the company. By identifying regions with untapped potential, we can contribute to food security, reduce reliance on a few major producers, and foster sustainable agricultural practices. """)
#     st.write("Binary Classification: if the classes are imbalanced: ", items)

#     items = df['Item'].value_counts().nlargest(100)
#     st.write("Main agricultural products cultivated:", items)

#     df['Item'] = df['Item'].str.lower()
#     blueberries = df['Item'].str.contains('blueberries')
#     gums = df['Item'].str.contains('gums')
#     peppermint = df['Item'].str.contains('peppermint')
#     tallowtree = df['Item'].str.contains('tallowtree')
#     roots = df['Item'].str.contains('roots')
#     vegetablesmelons = df['Item'].str.contains('vegetablesmelons')
#     vegetables = df['Item'].str.contains('vegetables')
#     kola = df['Item'].str.contains('kola')
#     nuts = df['Item'].str.contains('nuts')
#     melons = df['Item'].str.contains('melons')
#     cereals = df['Item'].str.contains('cereals')
#     bananas = df['Item'].str.contains('bananas')
#     agave = df['Item'].str.contains('agave fibres nes')
#     apples = df['Item'].str.contains('apples')
#     apricots = df['Item'].str.contains('apricots')
#     sparagus = df['Item'].str.contains('sparagus')


#     df.loc[vegetables, 'Item'] = 'vegetables'
#     df.loc[roots, 'Item'] = 'roots'
#     df.loc[kola, 'Item'] = 'kola'
#     df.loc[nuts, 'Item'] = 'nuts'
#     df.loc[melons, 'Item'] = 'melons' 
#     df.loc[cereals, 'Item'] = 'cereals'
#     df.loc[bananas, 'Item'] = 'bananas'
#     df.loc[agave, 'Item'] = 'agave'
#     df.loc[apples, 'Item'] = 'apples'
#     df.loc[apricots, 'Item'] = 'apricots'
#     df.loc[sparagus, 'Item'] = 'sparagus'

#     df.loc[~blueberries & ~gums & ~roots & ~kola & ~nuts & ~melons & ~cereals &
#        ~peppermint & ~tallowtree & ~vegetables & ~vegetablesmelons & ~bananas & ~agave  & ~apples & ~apricots & ~sparagus,'Item'] = 'Other'
#     items1 = df['Item'].value_counts()
#     st.write("Subclassification:", items1)


   #model analysiis
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestClassifier 

    # from pdpbox import pdp

    # data = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)
    
    # # Split data into features and target variable
    # X = data.drop('Area', axis=1)
    # y = data['Area']

    # # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = RandomForestClassifier()
    # model.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # st.write("Accuracy:", accuracy)

    # # Export the first three decision trees from the forest

    # for i in range(3):
    #     tree = rf.estimators_[i]
    #     dot_data = export_graphviz(tree,
    #                             feature_names=X_train.columns,  
    #                             filled=True,  
    #                             max_depth=2, 
    #                             impurity=False, 
    #                             proportion=True)
    #     graph = graphviz.Source(dot_data)
    #     st.pyplot(graph)


    # Select the features for interaction
    # features = ['feature1', 'feature2']

    # # Create the PDP interact plot
    # pdp_interact_plot = pdp.pdp_interact(model, dataset=X_train, model_features=X.columns, feature_names=features)

    # # Display the plot in Streamlit
    # st.pyplot(pdp_interact_plot)








    # test = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1) 
    # val = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)  
    # train = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)
    # train['Item'].mode()   

    # test.replace(np.nan, 0, inplace=True)
    # val.replace(np.nan, 0, inplace=True)
    # train.replace(np.nan, 0, inplace=True)
    # train.describe(exclude='number')

    # val['Item'].value_counts()
    # bananas = val[val.Item=='Bananas']

    # target = 'Element'
    # train[target].value_counts(normalize=True)
    # features = train.select_dtypes('number').columns.drop(target)
    # X_train = train[features]
    # y_train = train[target]
    # X_val = val[features]
    # y_val = val[target]
    # X_test = test[features]
    # y_test = test[target]

    # pipeline = make_pipeline(
    # ce.OrdinalEncoder(), 
    # SimpleImputer(strategy='median'), 
    # RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # )

    # # Fit on train, score on val
    # pipeline.fit(X_train, y_train)
    # st.write('Validation Accuracy', pipeline.score(X_val, y_val))

    # # Get feature importances
    # rf = pipeline.named_steps['randomforestclassifier']
    # importances = pd.Series(rf.feature_importances_, X_train.columns)

    # # Plot feature importances
    # n =  100
    # plt.figure(figsize=(20, n / 2))  # Adjust figure size based on number of features
    # plt.title(f'Top {n} features')
    # sorted_importances = importances.sort_values(ascending=False)[:n]  # Sort descending and select top n
    # fig, ax = plt.subplots()
    # sorted_importances.plot.barh(color='Cyan');
    # st.title("Permutation Plotter")
    # st.pyplot(fig)    


if __name__ == "__main__":
     main()