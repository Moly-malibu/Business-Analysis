import streamlit as st
import streamlit.components.v1 as components 
from datetime import datetime
import datetime
import requests
from requests import get
import pandas as pd

st.markdown("<h1 style='text-align: center; color: #002967;'>UNHCR</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #002967;'>App String Palindrome</h1>", unsafe_allow_html=True)
def main():
    # Register pages
    pages = {
        "Home": Home,
        "Palindrome": Palindrome,
    }
    st.sidebar.title("Palindrome App")
    page = st.sidebar.selectbox("Select Menu", tuple(pages.keys()))
    pages[page]()

def Home():
    def main():
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://images.pexels.com/photos/1024613/pexels-photo-1024613.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.write(
        """
        Artificial Intelligence helps you to perform a successful future, taking the correct decisions by little effort! 
        ---

        App:

        1. Check if the input is a palindrome (reads the same both forwards and backwards.
                
        2. If the input length is odd, find and return the letter in the middle of the string.
                
        3. Send appropriate responses to the frontend:
                Example Input	: 
                        "racecar" Example Output: "The input 'racecar' is a palindrome and its middle letter is 'e'."
                Example Input:	 
                        "hello" Example Output: "The input 'hello' is not a palindrome."
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
                        <a href="http://www.monicadatascience.com/" target="#002967" class="white-text">
                        ❤<i class="❤"></i>
                    </a>
                    <a href="https://www.linkedin.com/in/monica-bustamante-b2ba3781/3" target="#002966" class="white-text">
                        <i class="fab fa-linkedin fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002967" class="white-text">
                    ❤<i class="❤"></i>
                    </a>
                    <a href="https://github.com/Moly-malibu/UNHCR.git" target="#002967" class="white-text">
                        <i class="fab fa-github-square fa-4x"></i>
                    <a href="http://www.monicadatascience.com/" target="#002967" class="white-text">
                    ❤<i class="❤"></i>
                    </a>
                    </ul>
                    </div>
                </div>
                </div>
                <div class="footer-copyright">
                <div class="container">
                <a class="white-text text-lighten-3" href="http://www.monicadatascience.com/">Made by Monica Bustamante</a><br/>
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
    
def Palindrome():
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg&ga=GA1.2.635976572.1603931911");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #002966;'>Example</h1>", unsafe_allow_html=True)
    
    palinwords = 'https://raw.githubusercontent.com/Moly-malibu/UNHCR/main/wordspalid.csv'
    df = pd.read_csv(palinwords)
    words = st.sidebar.selectbox('Palindrome', (df))
    st.markdown("<h1 style='text-align: center; color: #002967;'>String</h1>", unsafe_allow_html=True)
    word = st.sidebar.selectbox('Done', (df))
    
    if(word == word[: : -1]):
        st.write("Is Palindrome")
    else:
        st.write("Is Not a Palindrome")
    
    def is_palindrome(word): 
        if len(word) == 0: 
            return True
        left = 0
        right = len(word)-1
        while (left < right): 
            if word[left] != word[right]: 
                return False
            left += 1; right -= 1
            return True
        return is_palindrome
    
    st.write('***Word or String***', word)
    if len(word) % 2 == 0:
        st.write(word[len(word) // 2 - 1:len(word) // 2 + 1])
    else:
        st.write('middle letter is',word[len(word) // 2])


if __name__ == "__main__":
   main()

  
