import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit.components.v1 as components

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

# Function to generate a custom colormap
def generate_colormap(color1, color2):
    return LinearSegmentedColormap.from_list('custom_cmap', [color1, color2])

# Function to plot and export the chart with customizable title, x-axis, y-axis, and legend labels
def plot_and_export_chart(df, min_enrichment, max_enrichment, min_log_pval, max_log_pval, colormap, title, x_label, y_label, legend_label):
    # ... (rest of the function remains the same)

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Data loaded successfully!")
        st.dataframe(df.head(10))  # Displaying the top 10 entries for review

        # PyGWalker Integration
        st.write("### Interactive Data Exploration with PyGWalker")
        
        # Create a resizable container for PyGWalker
        pyg_html = StreamlitRenderer(df)
        with st.expander("PyGWalker Visualization", expanded=True):
            container = st.container()
            with container:
                pyg_html.render_explore()
        
        # Add a button to open PyGWalker in a new window
        if st.button("Open PyGWalker in New Window"):
            components.html(
                f"""
                <html>
                <body>
                    <script>
                        var win = window.open("", "PyGWalker", "width=800,height=600");
                        win.document.body.innerHTML = `
                            <div id="pyg-container" style="width: 100%; height: 100%;"></div>
                            <script src="https://cdn.jsdelivr.net/npm/pygwalker/dist/pyg.js"></script>
                            <script>
                                pygwalker.init('{pyg_html.spec_json}', {{
                                    target: '#pyg-container',
                                    dataUrl: '{pyg_html.df_json_url}',
                                    hideDataSourceConfig: true,
                                    themeKey: 'vega',
                                    defaultGroupByCol: false,
                                    defaultNumericBinning: false
                                }});
                            </script>
                        `;
                    </script>
                    <p>PyGWalker opened in a new window. If it doesn't appear, please check your pop-up blocker settings.</p>
                </body>
                </html>
                """,
                height=100,
            )

        # Original Visualization
        st.write("### Original Visualization")
        
        # ... (rest of the code remains the same)

else:
    st.warning("Please upload an Excel file to visualize the data.")
