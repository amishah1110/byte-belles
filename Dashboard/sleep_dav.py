import streamlit as st

def show_preprocessed_sleep():

    # Set up Streamlit layout
   # st.set_page_config(page_title="Social Media Usage Dashboard", layout="wide")
    #st.title("📊 Sleep Pattern Dashboard")

    # Set layout to avoid scrollbars
    # st.set_page_config(layout="wide")

    plots = [
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Sleep_train_images\\Distribution of sleep disorder.png",
            "title": "Distribution of sleep disorder.png",
            "inference":"Category 2 (Non-Binary) Dominates:** Sleep disorder category '2' has a significantly higher count compared to categories '0' and '1'. This suggests a much larger proportion of individuals in the dataset are classified under category '2'."
        },

        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Sleep_train_images\\Corelation_heatmap.png",
            "title": "Corelation_heatmap.png",
            "inference":"The heatmap reveals strong correlations between age, blood pressure, BMI, and sleep quality, while highlighting the inverse relationship between stress and sleep, and weaker connections for physical activity."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Sleep_train_images\\Heart rate vs Stress level.png",
            "title": "Heart rate vs Stress level.png",
            "inference":"**Weak Correlation:** There is no clear linear relationship observed between heart rate and stress level in this scatter plot. The data points appear to be scattered randomly."

        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Sleep_train_images\\SleepDurationHisto.png",
            "title": "SleepDurationHisto.png",
            "inference": "The distribution of normalized sleep duration shows multiple peaks, suggesting diverse sleep patterns within the dataset with clustering around specific duration ranges."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\Sleep_train_images\\sleepDurationVsAge.png",
            "title": "sleepDurationVsAge.png",
            "inference":"Sleep duration varies significantly across age groups, showing a trend of shorter normalized sleep duration in younger and older age brackets with some outliers indicating individual variability."  }
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
