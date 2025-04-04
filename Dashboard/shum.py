import streamlit as st


def show_social_dashboard():
    # Set up Streamlit layout
    # st.set_page_config(page_title="Social Media Usage Dashboard", layout="wide")
    st.title("📊 Social Media Usage Dashboard")

    # Dictionary to store image paths and their inferences
    plots = [
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\engagement score by emotion.png",
            "title": "Age Group Distribution of Users and Distribution of Social Media Platform Usage",
            "inference": "\n1. The majority of users (63.9%) fall within the 25-34 age group, followed by the 18-24 age group (24.2%).\n 2. Instagram (22.0%) and Twitter (20.8%) are the most frequently used social media platforms among the users in this dataset.",
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_1.png",
            "title": "Platform Preference by Age Group",
            "inference": "Instagram is the most preferred platform across all represented age groups, particularly the 25-34 demographic.",
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_2.png",
            "title": "Engagement Score by Age Group",
            "inference": "The 25-34 and 35-44 age groups exhibit higher median engagement scores compared to the 18-24 and Unknown age groups, with the 18-24 group showing more variability and outliers.",
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_4.png",
            "title": "Platform vs Emotion Correlation",
            "inference": "Instagram shows a notably high correlation with Happiness, while Twitter has a relatively higher association with Anger compared to other platforms.",
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_3.png",
            "title": "Engagement Score by Emotion",
            "inference": "Users expressing Happiness tend to have the highest engagement scores, while Boredom is associated with the lowest engagement."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_5.png",
            "title": "Daily Usage Time by Platform",
            "inference": "Instagram exhibits the highest median daily usage time, while LinkedIn shows the lowest among the platforms analyzed.",
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_6.png",
            "title": "Daily Usage Time Distribution with Mean & Standard Deviation",
            "inference": "The distribution of daily usage time is right-skewed and appears bimodal, with a mean around 96 minutes and a standard deviation of approximately 39 minutes, indicating variability in user engagement."
        },

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
