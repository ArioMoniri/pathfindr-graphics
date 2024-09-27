import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
import pygwalker as pyg
from pygwalker import st_pygwalker  # Ensure to import the Streamlit-specific PyGWalker integration

# Set the title and description of the app
st.title("Pathway Significance Visualization with PyGWalker")
st.write("Upload an Excel file containing columns like 'Annotation Name', 'Enrichment', and 'p-value'.")

# Function to load data from an uploaded Excel file
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    # Rename the columns to match expected naming
    data.rename(columns={'Annotation Name': 'Pathway', 'Enrichment': 'Fold Enrichment'}, inplace=True)
    # Calculating -log10(p-value)
    data['-log10(p-value)'] = -np.log10(data['p-value'].replace(0, np.finfo(float).tiny))
    data['Pathway'] = data['Pathway'].str.replace(r"\(.*\)", "", regex=True).str.strip()
    data.sort_values(by='-log10(p-value)', ascending=False, inplace=True)
    return data

# Function to generate a custom colormap
def generate_colormap(color1, color2):
    return LinearSegmentedColormap.from_list('custom_cmap', [color1, color2])

# Function to plot and export the chart with customizable title, x-axis, y-axis, and legend labels
def plot_and_export_chart(df, min_enrichment, max_enrichment, min_log_pval, max_log_pval, colormap, title, x_label, y_label, legend_label):
    filtered_data = df[(df['Fold Enrichment'] >= min_enrichment) & (df['Fold Enrichment'] <= max_enrichment) &
                       (df['-log10(p-value)'] >= min_log_pval) & (df['-log10(p-value)'] <= max_log_pval)]
    
    outside_range = df[(df['Fold Enrichment'] < min_enrichment) | (df['Fold Enrichment'] > max_enrichment) |
                       (df['-log10(p-value)'] < min_log_pval) | (df['-log10(p-value)'] > max_log_pval)]
    
    if not outside_range.empty:
        st.warning(f"Pathways outside the selected ranges: {outside_range[['Pathway', 'Fold Enrichment', '-log10(p-value)']]}")

    top_10_pathways = filtered_data.head(10)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x=top_10_pathways['Fold Enrichment'], y=top_10_pathways['Pathway'],
                          c=top_10_pathways['-log10(p-value)'], cmap=colormap, s=300, alpha=0.85, edgecolor='black')
    plt.colorbar(scatter, label=legend_label if legend_label else '-log10(p-value)')
    plt.xlabel(x_label if x_label else 'Fold Enrichment')
    plt.ylabel(y_label if y_label else 'Pathway')
    plt.title(title if title else 'Top 10 Pathways by Significance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return plt.gcf()

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Data loaded successfully!")
        st.dataframe(df.head(10))

        # Integrating PyGWalker for interactive data exploration
        st.write("### Explore Data Interactively with PyGWalker")
        st_pygwalker(df)  # Use st_pygwalker to render PyGWalker in Streamlit

        # Select the fold enrichment range
        st.write("### Select Fold Enrichment Range")
        min_enrichment = st.slider("Minimum Fold Enrichment", min_value=float(df['Fold Enrichment'].min()), max_value=float(df['Fold Enrichment'].max()), value=float(df['Fold Enrichment'].min()))
        max_enrichment = st.slider("Maximum Fold Enrichment", min_value=min_enrichment, max_value=float(df['Fold Enrichment'].max()), value=float(df['Fold Enrichment'].max()))

        # Select the -log10(p-value) range
        st.write("### Select -log10(p-value) Range")
        min_log_pval = st.slider("Minimum -log10(p-value)", min_value=float(df['-log10(p-value)'].min()), max_value=float(df['-log10(p-value)'].max()), value=float(df['-log10(p-value)'].min()))
        max_log_pval = st.slider("Maximum -log10(p-value)", min_value=min_log_pval, max_value=float(df['-log10(p-value)'].max()), value=float(df['-log10(p-value)'].max()))

        # Default colormap option
        st.write("### Customize Color Palette (Optional)")
        use_custom_colors = st.checkbox("Use Custom Colors", value=False)
        colormap = 'viridis'
        if use_custom_colors:
            color1 = st.color_picker("Select First Color", value='#440154')
            color2 = st.color_picker("Select Second Color", value='#FDE725')
            colormap = generate_colormap(color1, color2)

        # Allow user to set title, x-axis, y-axis, and legend labels
        st.write("### Customize Labels (Optional)")
        custom_title = st.text_input("Title", "Top 10 Pathways by Significance")
        custom_x_label = st.text_input("X-axis Label", "Fold Enrichment")
        custom_y_label = st.text_input("Y-axis Label", "Pathway")
        custom_legend_label = st.text_input("Legend Label", "-log10(p-value)")

        fig = plot_and_export_chart(df, min_enrichment, max_enrichment, min_log_pval, max_log_pval, colormap, custom_title, custom_x_label, custom_y_label, custom_legend_label)
        st.pyplot(fig)

        # Export buttons
        st.write("### Export Chart")
        export_as = st.selectbox("Select format to export:", ["JPG", "PNG", "SVG", "TIFF"])

        def save_and_download(format, dpi=600):
            buffer = BytesIO()
            fig.savefig(buffer, format=format, bbox_inches='tight', facecolor='white', dpi=dpi)
            buffer.seek(0)
            plt.close(fig)
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
            dpi = st.slider("Select DPI for TIFF", min_value=100, max_value=1200, value=600, step=50)
            buffer = save_and_download("tiff", dpi=dpi)
            st.download_button("Download TIFF", buffer, file_name='chart.tiff', mime='image/tiff')

else:
    st.warning("Please upload an Excel file to visualize the data.")
