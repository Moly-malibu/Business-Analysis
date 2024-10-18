import streamlit as st
import streamlit.components.v1 as components 
from datetime import datetime
import datetime
import requests
from requests import get
import pandas as pd
import numpy as np
#import category_encoders as ce
#from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
 


st.markdown("<h1 style='text-align: center; color: #002967;'>Business Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #002967;'>Agriculture Crop Production by Countries: Vegetales & Fruits</h1>", unsafe_allow_html=True)
def main():
    # Register pages
    pages = {
        "Home": Home,
        "datainfo": datainfo,
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
        st.write(
        """
        Artificial Intelligence helps you to perform a successful future, taking the correct decisions by little effort! 
        ---

        The aim is to analyze the performance of the Agro-Business industry from 1961 to 2014. Historical data reveals that while the banana industry once faced significant challenges, it has defied initial predictions. Despite forecasts suggesting a bleak future, scientific advancements have played a crucial role in ensuring bananas remain a staple in the global food basket. Although the banana industry has experienced slower growth compared to other sectors within Agribusiness, its resilience over the past six decades is a testament to the power of innovation and adaptation.
                        ---
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
    
def datainfo():
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg&ga=GA1.2.635976572.1603931911");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Agricultural Production</h1>", unsafe_allow_html=True)
    
    df = pd.read_csv('https://raw.githubusercontent.com/Moly-malibu/agriculture-crop-production/master/agriculture-crop-production%20(11).csv').drop(['Unnamed: 0'], axis=1)
    df.replace(np.nan, 0, inplace=True)
    info = st.sidebar.selectbox('Data Set Info', (df))
    st.markdown("<h1 style='text-align: center; color: #002967;'>By Countries</h1>", unsafe_allow_html=True)
    info = st.sidebar.selectbox('Done', (df))
    shape=(df.shape)
    
    st.write("Production by Countries:", df)
    st.write("Dimension DataSet:", shape, "Shape")  
    info= df.describe()
    st.write("Statistics Report DataSet:", info)
    
    items = df.groupby(['Item', 'Element', 'Unit']).size().reset_index(name='Products')
    st.write("Products:", items)
    
    items= df['Item'].describe()
    st.write("Statistic Describe: Targe Products:", items)

    y = df['Item']
    y.nunique()
    items = y.value_counts(normalize=True)
    st.write("Binary Classification: if the classes are imbalanced:", items)

    items= df['Item'].unique()
    st.write("classification of agricultural products:", items)

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

    df.loc[vegetables, 'Item'] = 'vegetables'
    df.loc[roots, 'Item'] = 'roots'
    df.loc[kola, 'Item'] = 'kola'
    df.loc[nuts, 'Item'] = 'nuts'
    df.loc[melons, 'Item'] = 'melons' 
    df.loc[cereals, 'Item'] = 'cereals'
    df.loc[bananas, 'Item'] = 'bananas'

    df.loc[~blueberries & ~gums & ~roots & ~kola & ~nuts & ~melons & ~cereals &
       ~peppermint & ~tallowtree & ~vegetables & ~vegetablesmelons & ~bananas, 'Item'] = 'Other'
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
    st.write("Production:", bananas)

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



if __name__ == "__main__":
   main()

  
