import streamlit as st
import streamlit.components.v1 as components 
from datetime import datetime
import datetime
import requests
from requests import get
import pandas as pd
import numpy as np
 


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
    st.write("Dimension DataSet:", shape, "Shape")
    info= df.describe()
    st.write("Statistics Report DataSet:", info)
    st.write("Production by Countries:", df)


if __name__ == "__main__":
   main()

  
