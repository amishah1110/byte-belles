import streamlit as st
import home
import sleep_dav
import shum
import predict

# Set page config at the top
st.set_page_config(page_title="Byte Belles Dashboard", layout="wide")

# Routing logic
page = st.query_params.get("page", "home")

if page == "home":
    home.show_home_page()
elif page == "sleep":
    sleep_dav.show_preprocessed_sleep()
elif page == "social":
    shum.show_social_dashboard()
elif page == "predictor":
    predict.show_predictor_page()