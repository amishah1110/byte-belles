import streamlit as st


# Set up Streamlit layout
st.set_page_config(page_title="Social Media Usage Dashboard", layout="wide")
st.title("📊 Social Media Usage Dashboard")

# Dictionary to store image paths and their inferences
plots = [
    {
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\standard deviation and mean of daily usage.png",
        "title": "Standard Deviation and Mean of Daily Usage",
        "inference":  """
    - **Higher standard deviation for younger users** – Their engagement fluctuates more.
    - **Stable usage in older demographics** – Less variability suggests consistent habits.
    - **Peak engagement around late evenings** – Users engage more after work/school hours.
    """
    },
    {
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\daily usage time by platform.png",
        "title": "Daily Usage Time by Platform",
        "inference": """
        - **Instagram has the highest usage** – The median daily usage time is the highest among all platforms, with a wider interquartile range (IQR) and some high outliers.
        - **Twitter, Facebook, and LinkedIn show lower usage** – Their median usage times are negative or near zero, which may indicate limited engagement in the dataset.
        - **WhatsApp and Snapchat have moderate usage** – Their medians are positive, but with a wider spread, indicating varying usage patterns among users.
        - **Telegram shows mixed usage** – It has a slightly negative median but a balanced spread.
        - **Presence of "nan" category** – This might indicate missing or undefined data, but it has a small spread suggesting a constant value.         
        """,
    },
    {
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\emotion and platform heattmap.png",
        "title": "Emotion and Platform Heatmap",
        "inference": """
    - **Positive emotions correlate with Instagram and TikTok usage** – Users experiencing joy or excitement spend more time here.
    - **Negative emotions drive engagement on Twitter** – Likely due to political discussions, debates, or venting behavior.
    - **Neutral emotions dominate LinkedIn** – Suggesting more professional and structured interactions.
    - **Snapchat and WhatsApp show mixed emotions** – Users engage in personal, close-knit communication leading to varied emotional trends.
    """
    },
    {
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\Platform preferance by age.png",
        "title": "Platform Preference by Age",
        "inference":  """
    - **Instagram and TikTok dominate among younger users** – Preferred platforms for those under 25.
    - **Facebook and LinkedIn are favored by professionals** – Preferred platforms for networking and information sharing.
    - **WhatsApp remains universal** – Used across all age groups for communication.
    """
    },
    {
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\train_KDEplot.png",
        "title": "KDE Plot of Daily Usage Time",
        "inference": """
- **Most users spend 50-100 minutes per day**, but a few spend **well over 150 minutes**—these might be influencers or highly engaged users.
- **Two distinct user groups** appear: moderate users (70-80 minutes) and heavy users (~120 minutes).
- A **small percentage of extreme users (~200 minutes daily)** suggests some people are deeply invested in social media—possibly due to work, content creation, or heavy engagement.
    - **Identifies peaks in daily usage trends** – Shows most common engagement times.
    - **Multi-modal distribution** – Suggests different user groups with varied usage behaviors.
    - **Consistent engagement pattern** – Indicates habitual platform usage.
    """
,
    },
    {
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\train-correlation-heatmap.png",
        "title": "Correlation Heatmap",
        "inference": """
        
- **Strong Positive Correlations:** All variables (Daily_Usage_Time, Posts_Per_Day, Likes_Received_Per_Day, Comments_Received_Per_Day, Messages_Sent_Per_Day) show strong positive correlations with each other (all above 0.88).
- **Self-Correlation:** Each variable has a perfect correlation (1.00) with itself, as expected.
- **Daily Usage Time Impact:** Users with higher daily usage time tend to post more, receive more likes and comments, and send more messages.
- **Posting Activity:** Users who post more frequently also receive more engagement (likes and comments) and send more messages.
- **Engagement (Likes/Comments) Link:** Users receiving more likes and comments are generally more active on the app (higher usage, posting, and messaging).
- **Messaging Activity:** Users who send more messages are also more active in other aspects of the app.
- **Highest Correlation:** The strongest correlations (0.94) are between:
    - Daily_Usage_Time and Likes_Received_Per_Day
    - Likes_Received_Per_Day and Comments_Received_Per_Day
- **Lowest (Still Strong) Correlation:** The weakest (but still strong at 0.88) correlations are between:
    - Posts_Per_Day and Messages_Sent_Per_Day
    - Comments_Received_Per_Day and Messages_Sent_Per_Day
"""
    },
{
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\Distribution of platform usage and age group.png",
        "title": "Distribution of platform usage and age group.png",
        "inference": """
    - **Young users (18-24) dominate Instagram, Snapchat, and TikTok** – Higher engagement in visually driven platforms.
    - **Facebook and LinkedIn attract older demographics** – These platforms show increasing usage from age 30+.
    - **WhatsApp has a balanced spread** – Used across all age groups consistently.
    - **Younger users have higher variation** – They experiment more with platforms, leading to greater dispersion.
    
    """},
{
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\engagement score by age group.png",
        "title": "engagement score by age group.png",
        "inference": """
    - **18-24 age group shows the highest engagement** – This is the most active demographic across platforms.
    - **Engagement gradually declines with age** – Older users participate less actively, possibly due to reduced free time.
    - **Variance is highest among younger users** – Some are heavy users, while others are less engaged, leading to dispersion.
    """
},
{
        "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\engagement score by emotion.png",
        "title": "engagement score by emotion.png",
        "inference": """
    - **Happiness and Excitement drive higher engagement** – Positive emotions correlate with longer usage sessions.
    - **Anger leads to spikes in engagement** – Users vent frustrations on platforms like Twitter.
    - **Sadness shows lower engagement** – Users experiencing sadness may reduce platform usage.
    """}
]


st.write("Use the left and right buttons below to navigate through the plots and associated inferences.")

# Session state for navigation
if "index" not in st.session_state:
    st.session_state.index = 0

# Display image and inference
current_plot = plots[st.session_state.index]
st.image(current_plot["path"], caption=current_plot["title"], use_container_width=True)
st.subheader("📌 Inference")
st.write(current_plot["inference"])

# Navigation buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state.index > 0:
        st.session_state.index -= 1
        st.rerun()
with col2:
    if st.button("Next ➡️") and st.session_state.index < len(plots) - 1:
        st.session_state.index += 1
        st.rerun()
