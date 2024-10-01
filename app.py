import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from pygwalker.api.streamlit import StreamlitRenderer

# Set the title and description of the app
st.title("Pathway Significance Visualization with PyGWalker Integration")
st.write("Upload an Excel file and customize your visualization.")

# Function to load data from an uploaded Excel file
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    return data

# Function to generate a custom colormap
def generate_colormap(color1, color2):
    return LinearSegmentedColormap.from_list('custom_cmap', [color1, color2])

# Function to add log10 transformed columns
def add_log10_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    transform_columns = st.multiselect(
        "Select columns to apply log10 transformation",
        options=numeric_columns,
        help="These columns will be transformed before visualization options"
    )
    
    for col in transform_columns:
        new_col_name = f'log10({col})'
        df[new_col_name] = np.log10(df[col].replace(0, np.finfo(float).tiny))
        
        neg_log_col_name = f'-log10({col})'
        df[neg_log_col_name] = -np.log10(df[col].replace(0, np.finfo(float).tiny))
    
    return df

# Function to plot and export the chart
def plot_and_export_chart(df, x_col, y_col, color_col, ranges, colormap, title, x_label, y_label, legend_label, sort_by, sort_order, num_pathways):
    # Filter data based on ranges
    filtered_data = df.copy()
    for col, (min_val, max_val) in ranges.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            filtered_data = filtered_data[(filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)]
    
    # Sort data
    filtered_data = filtered_data.sort_values(by=sort_by, ascending=(sort_order == 'ascending'))
    
    # Select top/bottom pathways
    if num_pathways > len(filtered_data):
        num_pathways = len(filtered_data)
    selected_data = filtered_data.head(num_pathways) if sort_order == 'descending' else filtered_data.tail(num_pathways)

    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(df[color_col]):
        scatter = plt.scatter(
            x=selected_data[x_col],
            y=selected_data[y_col],
            c=selected_data[color_col],
            cmap=colormap,
            s=300,
            alpha=0.85,
            marker='o',
            edgecolor='black'
        )
        plt.colorbar(scatter, label=legend_label)
    else:
        unique_categories = df[color_col].unique()
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(unique_categories)))
        for category, color in zip(unique_categories, colors):
            category_data = selected_data[selected_data[color_col] == category]
            plt.scatter(
                x=category_data[x_col],
                y=category_data[y_col],
                label=category,
                color=color,
                s=300,
                alpha=0.85,
                marker='o',
                edgecolor='black'
            )
        plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if pd.api.types.is_categorical_dtype(df[y_col]) or pd.api.types.is_object_dtype(df[y_col]):
        plt.gca().invert_yaxis()
    plt.yticks(fontsize=8)
    plt.tight_layout()

    return plt.gcf(), filtered_data

# File uploader widget
uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write("Data loaded successfully!")
        st.dataframe(df.head(10))

        # Add log10 transformations before column selection
        df = add_log10_columns(df)
        
        # Update columns list after transformations
        columns = df.columns.tolist()

        # Column selection
        st.write("### Select Visualization Columns")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("Select X-axis column", options=columns, 
                                index=columns.index("Enrichment") if "Enrichment" in columns else 0)
        with col2:
            y_col = st.selectbox("Select Y-axis column", options=columns, 
                                index=columns.index("Annotation Name") if "Annotation Name" in columns else 0)
        with col3:
            color_col = st.selectbox("Select color column", options=columns, 
                                    index=columns.index("-log10(p-value)") if "-log10(p-value)" in columns else 0)

        # Sorting options
        st.write("### Sorting Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sort_by = st.selectbox("Sort pathways by", options=columns)
        with col2:
            sort_order = st.selectbox("Sort order", options=['descending', 'ascending'])
        with col3:
            num_pathways = st.slider("Number of pathways to show", min_value=1, max_value=len(df), value=10)

        # Range sliders for numeric columns
        st.write("### Range Filters")
        ranges = {}
        numeric_cols = [col for col in [x_col, y_col, color_col] if pd.api.types.is_numeric_dtype(df[col])]
        
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(numeric_cols):
                    col = numeric_cols[i + j]
                    with cols[j]:
                        st.write(f"Range for {col}")
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        ranges[col] = st.slider(
                            f"Select range for {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val)
                        )

        # Color options
        st.write("### Visual Customization")
        use_custom_colors = st.checkbox("Use Custom Colors", value=False)
        if use_custom_colors:
            col1, col2 = st.columns(2)
            with col1:
                color1 = st.color_picker("Select First Color", value='#440154')
            with col2:
                color2 = st.color_picker("Select Second Color", value='#FDE725')
            colormap = generate_colormap(color1, color2)
        else:
            colormap = 'viridis'

        # Custom labels
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_title = st.text_input("Title", "Pathway Visualization")
        with col2:
            custom_x_label = st.text_input("X-axis Label", x_col)
        with col3:
            custom_y_label = st.text_input("Y-axis Label", y_col)
        custom_legend_label = st.text_input("Legend Label", color_col)

        # Plot chart
        st.write("### Visualization")
        fig, filtered_data = plot_and_export_chart(
            df, x_col, y_col, color_col, ranges, colormap,
            custom_title, custom_x_label, custom_y_label, custom_legend_label,
            sort_by, sort_order, num_pathways
        )
        st.pyplot(fig)

        # Show filtered data
        st.write("### Filtered and Sorted Data")
        st.dataframe(filtered_data)

        # PyGWalker Integration
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

        # Export options
        st.write("### Export Options")
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
