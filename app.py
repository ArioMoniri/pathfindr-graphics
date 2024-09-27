import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO
from matplotlib.colors import LinearSegmentedColormap
from pygwalker.api.streamlit import StreamlitRenderer

# Set the title and description of the app
st.title("Pathway Significance Visualization with PyGWalker Integration")
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

# Function to generate the HTML file containing PyGWalker
def generate_pygwalker_html(df):
    pygwalker_html = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/pygwalker/dist/pyg.js"></script>
    </head>
    <body>
        <div id="pyg-container" style="width: 100%; height: 100vh;"></div>
        <script>
            pygwalker.init({{
                df: {df.to_json(orient='records')}, 
                target: '#pyg-container',
            }});
        </script>
    </body>
    </html>
    """
    return pygwalker_html

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Data loaded successfully!")
        st.dataframe(df.head(10))  # Displaying the top 10 entries for review

        # PyGWalker Integration directly in Streamlit
        st.write("### Interactive Data Exploration with PyGWalker")
        pygwalker = StreamlitRenderer(df)
        pygwalker.render_explore()

        # Create HTML for PyGWalker to open in a new tab
        st.write("### Open PyGWalker in a New Tab")
        html_content = generate_pygwalker_html(df)
        html_file = StringIO(html_content)

        # Provide a download link for the user to open PyGWalker in a new tab
        st.download_button(
            label="Download PyGWalker HTML",
            data=html_file.getvalue(),
            file_name="pygwalker_visualization.html",
            mime="text/html"
        )

        # Original Visualization
        st.write("### Original Visualization")
        
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

        if use_custom_colors:
            # Color pickers for custom colormap
            color1 = st.color_picker("Select First Color", value='#440154')  # Default dark purple from 'viridis'
            color2 = st.color_picker("Select Second Color", value='#FDE725')  # Default yellow from 'viridis'
            colormap = generate_colormap(color1, color2)
        else:
            # Default colormap as 'viridis'
            colormap = 'viridis'

        # Allow user to set title, x-axis, y-axis, and legend labels
        st.write("### Customize Labels (Optional)")
        custom_title = st.text_input("Title", "Top 10 Pathways by Significance")
        custom_x_label = st.text_input("X-axis Label", "Fold Enrichment")
        custom_y_label = st.text_input("Y-axis Label", "Pathway")
        custom_legend_label = st.text_input("Legend Label", "-log10(p-value)")

        # Check if df is not None and plot
        if df is not None:
            fig = plot_and_export_chart(df, min_enrichment, max_enrichment, min_log_pval, max_log_pval, colormap, custom_title, custom_x_label, custom_y_label, custom_legend_label)
            st.pyplot(fig)
        else:
            st.error("Data could not be loaded for plotting.")

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
