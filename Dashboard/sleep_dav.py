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
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\replace fig 8.png",
            "title": "Sleep Duration Across Age Groups",
            "inference": "Normalized sleep duration appears to increase with age up to the 45-54 group, which shows the highest median duration and considerable variability, before slightly decreasing in the 55-64 group."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\replace fig 10.png",
            "title": "Sleep Duration Distribution with Mean & Std Deviation",
            "inference": "The distribution of normalized sleep duration appears multimodal with several peaks, the standard deviation of 0.29 suggests a considerable spread in sleep durations across the dataset."
        },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\replace fig 11.png",
            "title": "Average Sleep Duration by Stress Level",
             "inference": "Average Sleep Duration is inversely proportional to Stress Level. Less sleep shows more stress."

    },
        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\replace fig 12.png",
            "title": "Sleep Disorder Distribution",
            "inference": "The majority of the dataset (58.6%) falls into sleep disorder category 'No Disorder', while categories 'Sleep Apnea' and 'Insomnia' have similar, smaller proportions (20.6% and 20.9% respectively)."
        },

        {
            "path": "D:\\Datathon\\byte-belles\\Data\\train social media\\replace fig 9.png",
            "title": "Average Physical Activity Level by Stress Level",
            "inference": "Average normalized physical activity level tends to increase with normalized stress level up , after which it fluctuates and generally decreases at higher stress levels."
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
