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
def plot_and_export_chart(df, min_x, max_x, min_y, max_y, colormap, title, x_label, y_label, legend_label, sort_order, sort_variable, top_n, x_col, y_col, color_col):
    # Sort the dataframe by the selected variable and order
    if sort_order == 'Head':
        sorted_df = df.sort_values(by=sort_variable).head(top_n)
    else:
        sorted_df = df.sort_values(by=sort_variable).tail(top_n)

    # Filter the data to include only rows within the selected ranges for the x and y columns
    filtered_data = sorted_df[(sorted_df[x_col] >= min_x) & (sorted_df[x_col] <= max_x) &
                              (sorted_df[y_col] >= min_y) & (sorted_df[y_col] <= max_y)]

    outside_range = sorted_df[(sorted_df[x_col] < min_x) | (sorted_df[x_col] > max_x) |
                              (sorted_df[y_col] < min_y) | (sorted_df[y_col] > max_y)]

    if not outside_range.empty:
        st.warning(f"The following pathways are outside the selected ranges for {x_col} or {y_col}:")
        st.write(outside_range[[x_col, y_col, color_col]])

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=filtered_data[x_col],
        y=filtered_data[y_col],
        c=filtered_data[color_col],
        cmap=colormap,
        s=300,
        alpha=0.85,
        marker='o',
        edgecolor='black'
    )
    
    plt.colorbar(scatter, label=legend_label if legend_label else color_col)
    plt.xlabel(x_label if x_label else x_col)
    plt.ylabel(y_label if y_label else y_col)
    plt.title(title if title else f'Top Pathways by {sort_variable}')
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
        
        x_col = st.selectbox("Select column for X-axis", options=columns, index=columns.index("Fold Enrichment") if "Fold Enrichment" in columns else 0)
        y_col = st.selectbox("Select column for Y-axis", options=columns, index=columns.index("Pathway") if "Pathway" in columns else 1)
        color_col = st.selectbox("Select column for Color", options=columns, index=columns.index("-log10(p-value)") if "-log10(p-value)" in columns else 2)
        
        # Sliders for dynamic ranges based on the selected X and Y columns
        if pd.api.types.is_numeric_dtype(df[x_col]):
            min_x = st.slider(f"Minimum {x_col}", min_value=float(df[x_col].min()), max_value=float(df[x_col].max()), value=float(df[x_col].min()))
            max_x = st.slider(f"Maximum {x_col}", min_value=min_x, max_value=float(df[x_col].max()), value=float(df[x_col].max()))
        else:
            st.write(f"Column {x_col} is not numeric, so it cannot have a range slider.")

        if pd.api.types.is_numeric_dtype(df[y_col]):
            min_y = st.slider(f"Minimum {y_col}", min_value=float(df[y_col].min()), max_value=float(df[y_col].max()), value=float(df[y_col].min()))
            max_y = st.slider(f"Maximum {y_col}", min_value=min_y, max_value=float(df[y_col].max()), value=float(df[y_col].max()))
        else:
            st.write(f"Column {y_col} is not numeric, so it cannot have a range slider.")

        # Sort options
        st.write("### Sort Options")
        sort_variable = st.selectbox("Sort by", options=columns)
        sort_order = st.selectbox("Sort Order", options=["Head", "Tail"])
        top_n = st.slider("Number of Pathways to Display", min_value=1, max_value=50, value=10)

        use_custom_colors = st.checkbox("Use Custom Colors", value=False)

        if use_custom_colors:
            color1 = st.color_picker("Select First Color", value='#440154')
            color2 = st.color_picker("Select Second Color", value='#FDE725')
            colormap = generate_colormap(color1, color2)
        else:
            colormap = 'viridis'

        custom_title = st.text_input("Title", "Top Pathways by Significance")
        custom_x_label = st.text_input("X-axis Label", x_col)
        custom_y_label = st.text_input("Y-axis Label", y_col)
        custom_legend_label = st.text_input("Legend Label", color_col)

        fig = plot_and_export_chart(df, min_x, max_x, min_y, max_y, colormap, custom_title, custom_x_label, custom_y_label, custom_legend_label, sort_order, sort_variable, top_n, x_col, y_col, color_col)
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
