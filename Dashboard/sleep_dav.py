import streamlit as st

def show_preprocessed_sleep():

    # Set up Streamlit layout
   # st.set_page_config(page_title="Social Media Usage Dashboard", layout="wide")
    st.title("📊 Sleep Pattern Dashboard")

    # Set layout to avoid scrollbars
    # st.set_page_config(layout="wide")

    plots = [
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_7.png",
            "title": "Correlation Heatmap of Features",
            "inference": "The heatmap shows strong positive correlations between age, systolic BP, and diastolic BP, as well as between sleep duration and quality, and a strong negative correlation between stress level and both sleep duration and quality."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_8.png",
            "title": "Sleep Duration Across Age Groups",
            "inference": "Normalized sleep duration appears to increase with age up to the 45-54 group, which shows the highest median duration and considerable variability, before slightly decreasing in the 55-64 group."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_9.png",
            "title": "Average Physical Activity Level by Stress Level",
            "inference": "Average normalized physical activity level tends to increase with normalized stress level up to a certain point (around 0.4), after which it fluctuates and generally decreases at higher stress levels."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_10.png",
            "title": "Sleep Duration Distribution with Mean & Std Deviation",
            "inference": "The distribution of normalized sleep duration appears multimodal with several peaks, and while the mean is around 0.49, the standard deviation of 0.29 suggests a considerable spread in sleep durations across the dataset."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_11.png",
            "title": "Average Sleep Duration by Stress Level",
            "inference": "The majority of the dataset (58.6%) falls into sleep disorder category '-1', while categories '0' and '1' have similar, smaller proportions (20.6% and 20.9% respectively)."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_12.png",
            "title": "Sleep Disorder Distribution",
            "inference": "Normalized sleep duration appears to increase with age up to the 45-54 group, which shows the highest median duration and considerable variability, before slightly decreasing in the 55-64 group."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_13.png",
            "title": "Feature Correlation with Sleep Duration",
            "inference": "Quality of Sleep shows the strongest positive correlation with Sleep Duration, while Stress Level exhibits the strongest negative correlation."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_14.png",
            "title": "Average Physical Activity Level by Stress Level",
            "inference": "The average normalized physical activity level appears to peak at a moderate normalized stress level (around 0.4) and then generally decreases at higher stress levels, with a slight increase at the highest stress level shown."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\Figure_15.png",
            "title": " Average Sleep Duration by Stress Level",
            "inference": "The average normalized sleep duration generally decreases as normalized stress level increases, with a notable drop at higher stress levels."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\train_KDEplot.png",
            "title": "Distribution of Daily Usage Time",
            "inference": "The distribution of daily usage time appears multimodal, with peaks suggesting common usage durations around 60-80 minutes and 90-100 minutes, and a noticeable right skew indicating some users with significantly longer usage times."
        }
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
