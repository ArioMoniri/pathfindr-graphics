import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from pygwalker.api.streamlit import StreamlitRenderer

# Set the title and description of the app
st.title("Pathway Significance Visualization with PyGWalker Integration")
st.write("Upload an Excel file and map the columns for 'Pathway', 'Fold Enrichment', and 'p-value'.")

# Function to load data from an uploaded Excel file
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    return data

# Function to generate a custom colormap
def generate_colormap(color1, color2):
    return LinearSegmentedColormap.from_list('custom_cmap', [color1, color2])

# Function to plot and export the chart
def plot_and_export_chart(df, min_enrichment, max_enrichment, min_log_pval, max_log_pval, colormap, title, x_label, y_label, legend_label):
    filtered_data = df[(df['Fold Enrichment'] >= min_enrichment) & (df['Fold Enrichment'] <= max_enrichment) &
                       (df['-log10(p-value)'] >= min_log_pval) & (df['-log10(p-value)'] <= max_log_pval)]
    
    outside_range = df[(df['Fold Enrichment'] < min_enrichment) | (df['Fold Enrichment'] > max_enrichment) |
                       (df['-log10(p-value)'] < min_log_pval) | (df['-log10(p-value)'] > max_log_pval)]
    
    if not outside_range.empty:
        st.warning(f"The following pathways are outside the selected ranges for fold enrichment or -log10(p-value):")
        st.write(outside_range[['Pathway', 'Fold Enrichment', '-log10(p-value)']])

    top_10_pathways = filtered_data.head(10)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=top_10_pathways['Fold Enrichment'],
        y=top_10_pathways['Pathway'],
        c=top_10_pathways['-log10(p-value)'],
        cmap=colormap,
        s=300,
        alpha=0.85,
        marker='o',
        edgecolor='black'
    )
    
    plt.colorbar(scatter, label=legend_label if legend_label else '-log10(p-value)')
    plt.xlabel(x_label if x_label else 'Fold Enrichment')
    plt.ylabel(y_label if y_label else 'Pathway')
    plt.title(title if title else 'Top 10 Pathways by Significance')
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=8)
    plt.tight_layout()

    return plt.gcf()

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write("Data loaded successfully!")
        st.dataframe(df.head(10))

        # Let user select which columns to use for Pathway, Fold Enrichment, and p-value
        columns = df.columns.tolist()
        
        pathway_col = st.selectbox("Select column for Pathway", options=columns, index=columns.index("Annotation Name") if "Annotation Name" in columns else 0)
        enrichment_col = st.selectbox("Select column for Fold Enrichment", options=columns, index=columns.index("Enrichment") if "Enrichment" in columns else 1)
        pval_col = st.selectbox("Select column for p-value", options=columns, index=columns.index("p-value") if "p-value" in columns else 2)

        # Rename the columns based on user selection
        df.rename(columns={pathway_col: 'Pathway', enrichment_col: 'Fold Enrichment', pval_col: 'p-value'}, inplace=True)

        # Add -log10(p-value) column and clean Pathway column
        df['-log10(p-value)'] = -np.log10(df['p-value'].replace(0, np.finfo(float).tiny))
        df['Pathway'] = df['Pathway'].str.replace(r"\(.*\)", "", regex=True).str.strip()
        df.sort_values(by='-log10(p-value)', ascending=False, inplace=True)

        # PyGWalker Integration directly in Streamlit
        st.write("### Interactive Data Exploration with PyGWalker")
        pygwalker = StreamlitRenderer(df)
        with st.container():
            st.write("""
                <style>
                    iframe {
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 140%;
                        height: 800px !important;
                    }
                </style>
                """, unsafe_allow_html=True)
            pygwalker.render_explore()

        # Original Visualization
        st.write("### Original Visualization")
        min_enrichment = st.slider("Minimum Fold Enrichment", min_value=float(df['Fold Enrichment'].min()), max_value=float(df['Fold Enrichment'].max()), value=float(df['Fold Enrichment'].min()))
        max_enrichment = st.slider("Maximum Fold Enrichment", min_value=min_enrichment, max_value=float(df['Fold Enrichment'].max()), value=float(df['Fold Enrichment'].max()))

        min_log_pval = st.slider("Minimum -log10(p-value)", min_value=float(df['-log10(p-value)'].min()), max_value=float(df['-log10(p-value)'].max()), value=float(df['-log10(p-value)'].min()))
        max_log_pval = st.slider("Maximum -log10(p-value)", min_value=min_log_pval, max_value=float(df['-log10(p-value)'].max()), value=float(df['-log10(p-value)'].max()))

        use_custom_colors = st.checkbox("Use Custom Colors", value=False)

        if use_custom_colors:
            color1 = st.color_picker("Select First Color", value='#440154')
            color2 = st.color_picker("Select Second Color", value='#FDE725')
            colormap = generate_colormap(color1, color2)
        else:
            colormap = 'viridis'

        custom_title = st.text_input("Title", "Top 10 Pathways by Significance")
        custom_x_label = st.text_input("X-axis Label", "Fold Enrichment")
        custom_y_label = st.text_input("Y-axis Label", "Pathway")
        custom_legend_label = st.text_input("Legend Label", "-log10(p-value)")

        fig = plot_and_export_chart(df, min_enrichment, max_enrichment, min_log_pval, max_log_pval, colormap, custom_title, custom_x_label, custom_y_label, custom_legend_label)
        st.pyplot(fig)

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
