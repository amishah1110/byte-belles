import streamlit as st



def show_social_dashboard():


    # Set up Streamlit layout
 #   st.set_page_config(page_title="Social Media Usage Dashboard", layout="wide")
    #st.title("📊 Social Media Usage Dashboard")

    # Dictionary to store image paths and their inferences
    plots = [
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\standard deviation and mean of daily usage.png",
            "title": "Standard Deviation and Mean of Daily Usage",
            "inference": "Right-skewed distribution shows most users spend between 50-100 minutes on social media. A small number spend significantly more (>150 minutes).",

        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\daily usage time by platform.png",
            "title": "Daily Usage Time by Platform",
            "inference": "This plot highlights platform engagement levels. Some platforms may have stronger engagement strategies leading to higher daily usage.",

        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\emotion and platform heattmap.png",
            "title": "Emotion and Platform Heatmap",
            "inference": "Platforms like Twitter show higher anger-related interactions, while Instagram is linked to positive emotions. This could guide content strategies.",

        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\Platform preferance by age.png",
            "title": "Platform Preference by Age",
            "inference": "Younger users dominate visually engaging platforms like Instagram, while older users prefer text-based platforms like LinkedIn or Facebook.",

        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\train_KDEplot.png",
            "title": "KDE Plot of Daily Usage Time",
            "inference": "A bimodal trend indicates two groups: moderate users (70-80 mins) and heavy users (~120 mins). A long tail suggests a few extreme users."

        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\train-correlation-heatmap.png",
            "title": "Correlation Heatmap",
            "inference": "Daily usage time strongly correlates with posts, likes, and messages sent per day, suggesting higher engagement increases usage time.",

        },
    {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\Distribution of platform usage and age group.png",
            "title": "Distribution of platform usage and age group.png",
            "inference": "**Young users (18-24) dominate Instagram, Snapchat, and TikTok** – Higher engagement in visually driven platforms."
       },
    {
            "path": "D:\\Datathon\\byte-belles\\Data\\Train_images\\engagement score by age group.png",
            "title": "engagement score by age group.png",
            "inference": "**18-24 age group shows the highest engagement** – This is the most active demographic across platforms."
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

    # Display images and inferences in pairs
    for i in range(0, len(plots), 2):
        row_plots = plots[i:i + 2]  # Get two plots for the row

        cols = st.columns(2)  # Create two columns for each pair

        for j, plot in enumerate(row_plots):
            with cols[j]:
                st.image(plot["path"], caption=plot["title"], use_container_width=True)
                st.write(f"**{plot['title']} Inference:** {plot['inference']}")


    if st.button("🔙 Back to Home"):
        st.query_params.page = "home"
        st.rerun()
