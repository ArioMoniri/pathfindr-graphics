import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the title and description of the app
st.title("Pathway Significance Visualization")
st.write("Upload an Excel file containing 'Pathway', 'Fold Enrichment', and 'p-value' columns.")

# Function to load data from an uploaded Excel file
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    # Calculating -log10(p-value)
    data['-log10(p-value)'] = -np.log10(data['p-value'].replace(0, np.finfo(float).tiny))
    # Removing details in parentheses from 'Pathway' and sorting
    data['Pathway'] = data['Pathway'].str.replace(r"\(.*\)", "", regex=True).str.strip()
    data.sort_values(by='-log10(p-value)', ascending=False, inplace=True)
    return data

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data loaded successfully!")
    st.dataframe(df.head(10))  # Displaying the top 10 entries for review

    # Selecting the top 10 significant pathways for visualization
    top_10_pathways = df.head(10)

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=top_10_pathways['Fold Enrichment'],
        y=top_10_pathways['Pathway'],
        c=top_10_pathways['-log10(p-value)'],
        cmap='viridis',  # Color map for intensity
        s=300,  # Increased size for greater emphasis
        alpha=0.85,
        marker='o',  # Circle markers
        edgecolor='black'  # Adding edge for better visibility
    )
    plt.colorbar(scatter, label='-log10(p-value)')
    plt.xlabel('Fold Enrichment')
    plt.ylabel('Pathway')
    plt.title('Top 10 Pathways by Significance')
    plt.gca().invert_yaxis()  # Invert y-axis for significance order
    plt.yticks(fontsize=8)  # Reduce font size to de-emphasize pathway names
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.warning("Please upload an Excel file to visualize the data.")
