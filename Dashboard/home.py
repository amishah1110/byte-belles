import streamlit as st

def show_home_page():
  #  st.set_page_config(page_title="Byte Belles Dashboard", layout="wide")
    st.markdown("## 🏠 Welcome to the Dashboard")
    st.markdown("Select one of the visualizations below:")

    # --- Card Styles ---
    st.markdown("""
        <style>
        .card-container {
            display: flex;
            justify-content: space-evenly;
            margin-top: 40px;
        }
        .card {
            width: 300px;
            height: 180px;
            background-color: #f8f9fa;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: 0.3s;
            cursor: pointer;
        }
        .card:hover {
            background-color: #e9ecef;
            transform: scale(1.02);
        }
        .card-title {
            font-size: 22px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }
        .card-subtitle {
            font-size: 16px;
            color: #777;
        }
        </style>

        <div class="card-container">
            <a href='?page=sleep' style='text-decoration: none;'>
                <div class='card'>
                    <div class='card-title'>😴 Sleep Data</div>
                    <div class='card-subtitle'>Explore sleep pattern insights</div>
                </div>
            </a>
            <a href='?page=social' style='text-decoration: none;'>
                <div class='card'>
                    <div class='card-title'>📱 Social Media</div>
                    <div class='card-subtitle'>Analyze usage and impact</div>
                </div>
            </a>
        </div>
    """, unsafe_allow_html=True)
