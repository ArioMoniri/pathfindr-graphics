import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Set the title and description of the app
st.title("Pathway Significance Visualization")
st.write("Upload an Excel file containing columns like 'Annotation Name', 'Enrichment', and 'p-value'.")

# Function to load data from an uploaded Excel file
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    
    # Rename the columns to match expected naming
    data.rename(columns={'Annotation Name': 'Pathway', 'Enrichment': 'Fold Enrichment'}, inplace=True)
    
    # Display actual column names after renaming
    st.write("Renamed columns in the uploaded file:", data.columns.tolist())

    # Check if expected columns exist in the dataframe after renaming
    expected_columns = ['Pathway', 'Fold Enrichment', 'p-value']
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_columns)}")
        return None
    
    # Calculating -log10(p-value) after ensuring the column exists
    data['-log10(p-value)'] = -np.log10(data['p-value'].replace(0, np.finfo(float).tiny))
    data['Pathway'] = data['Pathway'].str.replace(r"\(.*\)", "", regex=True).str.strip()
    data.sort_values(by='-log10(p-value)', ascending=False, inplace=True)
    return data

# Function to plot and export the chart
def plot_and_export_chart(df, min_enrichment, max_enrichment):
    # Filter the data to include only rows within the selected fold enrichment range
    filtered_data = df[(df['Fold Enrichment'] >= min_enrichment) & (df['Fold Enrichment'] <= max_enrichment)]
    
    # Identify pathways outside the selected range
    outside_range = df[(df['Fold Enrichment'] < min_enrichment) | (df['Fold Enrichment'] > max_enrichment)]
    
    # Display pathways that cannot be visualized
    if not outside_range.empty:
        st.warning(f"The following pathways are outside the selected range ({min_enrichment} to {max_enrichment}):")
        st.write(outside_range[['Pathway', 'Fold Enrichment']])

    # Selecting the top 10 significant pathways for visualization
    top_10_pathways = filtered_data.head(10)

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

    return plt.gcf()  # Get current figure

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Data loaded successfully!")
        st.dataframe(df.head(10))  # Displaying the top 10 entries for review

        # Select the fold enrichment range
        st.write("### Select Fold Enrichment Range")
        min_enrichment = st.slider("Minimum Fold Enrichment", min_value=float(df['Fold Enrichment'].min()), max_value=float(df['Fold Enrichment'].max()), value=float(df['Fold Enrichment'].min()))
        max_enrichment = st.slider("Maximum Fold Enrichment", min_value=min_enrichment, max_value=float(df['Fold Enrichment'].max()), value=float(df['Fold Enrichment'].max()))

        # Plot and display the chart within the selected range
        fig = plot_and_export_chart(df, min_enrichment, max_enrichment)
        st.pyplot(fig)

        # Export buttons
        st.write("### Export Chart")
        export_as = st.selectbox("Select format to export:", ["JPG", "PNG", "SVG", "TIFF"])

        # Function to save the plot to a buffer and download
        def save_and_download(format, dpi=600):
            buffer = BytesIO()
            fig.savefig(buffer, format=format, bbox_inches='tight', facecolor='white', dpi=dpi)
            buffer.seek(0)
            plt.close(fig)  # Close the figure after saving to prevent further modifications
            return buffer

        if export_as == "JPG":
            buffer = save_and_download("jpeg")
            st.download_button("Download JPG", buffer, file_name='chart.jpg', mime='image/jpeg')

        elif export_as == "PNG":
            buffer = save_and_download("png")
            st.download_button("Download PNG", buffer, file_name='chart.png', mime='image/png')

        elif export_as == "SVG":
            buffer = save_and_download("svg")
            st.download_button("Download SVG", buffer, file_name='chart.svg', mime='image/svg+xml')

        elif export_as == "TIFF":
            # Allow user to select DPI with a default value of 600
            dpi = st.slider("Select DPI for TIFF", min_value=100, max_value=1200, value=600, step=50)
            buffer = save_and_download("tiff", dpi=dpi)
            st.download_button("Download TIFF", buffer, file_name='chart.tiff', mime='image/tiff')

else:
    st.warning("Please upload an Excel file to visualize the data.")
