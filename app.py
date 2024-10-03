import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer
from functools import lru_cache

# Set the title and description of the app
st.set_page_config(layout="wide", page_title="Pathway Significance Visualization")
st.title("Pathway Significance Visualization with PyGWalker Integration")
st.write("Upload an Excel file and customize your visualization.")

# Function to load data from an uploaded Excel file
@st.cache_data
def load_data(uploaded_file):
    try:
        data = pd.read_excel(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to generate a custom colormap
@lru_cache(maxsize=None)
def generate_colormap(color1, color2):
    return LinearSegmentedColormap.from_list('custom_cmap', [color1, color2])

# Function to handle p-values and add log10 transformed columns
def transform_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    pvalue_columns = st.multiselect(
        "Select p-value columns for -log10 transformation",
        options=numeric_columns,
        help="These columns will be treated as p-values with -log10 transformation."
    )
    
    if pvalue_columns:
        st.info("Selected p-value columns will be transformed using -log10.")
        
        for col in pvalue_columns:
            neg_log_col_name = f'-log10({col})'
            df[neg_log_col_name] = -np.log10(df[col].clip(lower=1e-300))
    
    return df

# Function to normalize the data for size and opacity
def normalize_data(data, min_val, max_val, factor=1.0, increase=True):
    if isinstance(data, (int, float)):
        return data
    
    if np.all(np.isnan(data)) or len(data) == 0 or np.max(data) == np.min(data):
        return np.full(len(data), (min_val + max_val) / 2)
    
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    scaled_data = norm_data * (max_val - min_val) + min_val
    
    if not increase:
        scaled_data = max_val - (scaled_data - min_val)
    
    return scaled_data * factor

# Function to get sorted and filtered data
def get_sorted_filtered_data(df, sort_by, ranges, selection_method, num_pathways):
    filtered_data = df.copy()
    for col, (min_val, max_val) in ranges.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            filtered_data = filtered_data[(filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)]
    
    filtered_data = filtered_data.sort_values(by=sort_by)
    
    if num_pathways > len(filtered_data):
        num_pathways = len(filtered_data)
    
    if selection_method == 'Top (Highest Values)':
        selected_data = filtered_data.tail(num_pathways)
    elif selection_method == 'Bottom (Lowest Values)':
        selected_data = filtered_data.head(num_pathways)
    elif selection_method == 'Both Ends':
        half_num = num_pathways // 2
        selected_data = pd.concat([
            filtered_data.head(half_num),
            filtered_data.tail(half_num)
        ])
    else:  # Middle
        start_idx = (len(filtered_data) - num_pathways) // 2
        selected_data = filtered_data.iloc[start_idx:start_idx + num_pathways]
    
    return selected_data, filtered_data

# Function to create size and opacity legend
def create_legends(ax, sizes, opacities, size_label=None, opacity_label=None):
    legend_elements = []
    legend_labels = []
    
    # Size legend
    if not isinstance(sizes, (int, float)):
        unique_sizes = sorted(set([int(s) for s in sizes]))
        if len(unique_sizes) > 1:
            size_values = [min(unique_sizes), np.median(unique_sizes), max(unique_sizes)]
            for size in size_values:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor='gray', markersize=np.sqrt(size/10),
                                             markeredgecolor='black', linestyle='None'))
                legend_labels.append(f'{size_label}: {int(size)}' if size_label else f'Size: {int(size)}')

    # Opacity legend
    if not isinstance(opacities, (int, float)):
        unique_opacities = sorted(set([round(o, 2) for o in opacities]))
        if len(unique_opacities) > 1:
            opacity_values = [min(unique_opacities), np.median(unique_opacities), max(unique_opacities)]
            for opacity in opacity_values:
                legend_elements.append(Line2D([0], [0], marker='o', color='gray',
                                             markerfacecolor='gray', markersize=10,
                                             alpha=opacity, linestyle='None'))
                legend_labels.append(f'{opacity_label}: {opacity:.2f}' if opacity_label else f'Opacity: {opacity:.2f}')

    if legend_elements:
        ax.legend(legend_elements, legend_labels, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True, title="Size and Opacity")

# Function to plot and export the chart
def plot_and_export_chart(df, x_col, y_col, color_col, size_col, opacity_col, ranges, 
                         colormap, title, x_label, y_label, legend_label, sort_by, 
                         selection_method, num_pathways, fig_width, fig_height, 
                         min_size, max_size, min_opacity, max_opacity, 
                         size_increase, opacity_increase, size_factor, opacity_factor):
    try:
        selected_data, filtered_data = get_sorted_filtered_data(df, sort_by, ranges, 
                                                               selection_method, num_pathways)
        
        if selected_data.empty:
            st.warning("No data to display after applying filters.")
            return None, filtered_data, selected_data

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Handle size values
        if size_col != "None":
            sizes = pd.to_numeric(selected_data[size_col], errors='coerce')
            sizes = normalize_data(sizes, min_size, max_size, size_factor, size_increase)
        else:
            sizes = (min_size + max_size) / 2
        
        # Handle opacity values
        if opacity_col != "None":
            opacities = pd.to_numeric(selected_data[opacity_col], errors='coerce')
            opacities = normalize_data(opacities, min_opacity, max_opacity, opacity_factor, opacity_increase)
        else:
            opacities = (min_opacity + max_opacity) / 2

        # Ensure x_col and y_col are numeric, handle non-numeric (categorical) values explicitly
        x_values = pd.to_numeric(selected_data[x_col], errors='coerce')
        if pd.api.types.is_numeric_dtype(selected_data[y_col]):
            y_values = pd.to_numeric(selected_data[y_col], errors='coerce')
        else:
            # Convert non-numeric y_col to categorical and assign numeric labels
            y_values = pd.Categorical(selected_data[y_col]).codes

        # Plot the data
        if pd.api.types.is_numeric_dtype(selected_data[color_col]):
            scatter = ax.scatter(x_values, y_values,
                                c=selected_data[color_col],
                                cmap=colormap,
                                s=sizes,
                                alpha=opacities,
                                edgecolors='black')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(legend_label)
        else:
            unique_categories = selected_data[color_col].unique()
            colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(unique_categories)))

            for category, color in zip(unique_categories, colors):
                mask = selected_data[color_col] == category
                ax.scatter(x_values[mask], y_values[mask],
                          label=category,
                          color=color,
                          s=sizes[mask] if isinstance(sizes, np.ndarray) else sizes,
                          alpha=opacities[mask] if isinstance(opacities, np.ndarray) else opacities,
                          edgecolors='black')
            ax.legend(title=legend_label, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if isinstance(y_values, pd.Series) and (y_values.dtype == object or isinstance(y_values.dtype, pd.CategoricalDtype)):
            ax.invert_yaxis()

        # Create size and opacity legends
        create_legends(ax, sizes, opacities, 
                      size_label=size_col if size_col != "None" else None,
                      opacity_label=opacity_col if opacity_col != "None" else None)

        plt.tight_layout()
        return fig, filtered_data, selected_data
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None, None, None

# Function to display the plot in Streamlit
def display_plot(fig):
    if fig is not None:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        buf.seek(0)
        st.image(buf)

# Main execution
if __name__ == "__main__":
    uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data loaded successfully!")
            
            # Initialize variables
            fig = None
            filtered_data = None
            selected_data = None
            
            # Automatically apply p-value transformation
            df = transform_columns(df)
            columns = df.columns.tolist()

            # Use tabs to organize the UI
            tab1, tab2, tab3 = st.tabs(["Data Preview", "Visualization Settings", "Results"])
            
            with tab1:
                st.dataframe(df.head(10))

            with tab2:
                # Move all the setting widgets here
                with st.form("visualization_settings"):
                    # Column selection
                    st.write("### Select Visualization Columns")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        x_col = st.selectbox("Select X-axis column", options=columns)
                    with col2:
                        y_col = st.selectbox("Select Y-axis column", options=columns)
                    with col3:
                        color_col = st.selectbox("Select color column", options=columns)

                    # Size and opacity options
                    st.write("### Additional Circle Customization Options")
                    col1, col2 = st.columns(2)
                    with col1:
                        size_col = st.selectbox("Select size column (optional)", options=["None"] + columns)
                    with col2:
                        opacity_col = st.selectbox("Select opacity column (optional)", options=["None"] + columns)

                    # Min and max size and opacity
                    st.write("### Size and Opacity Adjustments")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        min_size = st.slider("Min size", min_value=100, max_value=1000, value=300)
                    with col2:
                        max_size = st.slider("Max size", min_value=100, max_value=1000, value=600)
                    with col3:
                        min_opacity = st.slider("Min opacity", min_value=0.0, max_value=1.0, value=0.5)
                    with col4:
                        max_opacity = st.slider("Max opacity", min_value=0.0, max_value=1.0, value=1.0)

                    # Sensitivity for size and opacity
                    st.write("### Sensitivity for Size and Opacity Changes")
                    col1, col2 = st.columns(2)
                    with col1:
                        size_factor = st.slider("Size change factor", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                    with col2:
                        opacity_factor = st.slider("Opacity change factor", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

                    # Options to increase or decrease size and opacity
                    st.write("### Size and Opacity Scaling Options")
                    col1, col2 = st.columns(2)
                    with col1:
                        size_increase = st.radio("Size increases with values", options=[True, False], index=0)
                    with col2:
                        opacity_increase = st.radio("Opacity increases with values", options=[True, False], index=0)

                    # Sorting options
                    st.write("### Sorting and Selection Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sort_by = st.selectbox("Sort pathways by", options=columns)
                    with col2:
                        selection_method = st.selectbox(
                            "Selection method",
                            options=['Top (Highest Values)', 'Bottom (Lowest Values)', 'Both Ends', 'Middle']
                        )
                    with col3:
                        num_pathways = st.slider("Number of pathways to show", min_value=1, max_value=len(df), value=10)

                    # Figure size options
                    st.write("### Figure Size")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_width = st.slider("Figure width", min_value=6, max_value=20, value=12)
                    with col2:
                        fig_height = st.slider("Figure height", min_value=4, max_value=16, value=8)

                    # Range sliders for numeric columns
                    st.write("### Range Filters")
                    ranges = {}
                    numeric_cols = [col for col in [x_col, y_col, color_col, size_col, opacity_col] 
                                   if col != "None" and pd.api.types.is_numeric_dtype(df[col])]

                    # Create a unique list of columns to avoid duplicate keys
                    unique_numeric_cols = list(dict.fromkeys(numeric_cols))

                    for i, col in enumerate(unique_numeric_cols):
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        
                        # Create a unique key for each column using its index
                        unique_key = f"range_slider_{i}_{col}"
                        
                        st.write(f"Range for {col}")
                        ranges[col] = st.slider(
                            f"Select range for {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=unique_key
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
                    st.write("### Custom Labels")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        custom_title = st.text_input("Title", "Pathway Visualization")
                    with col2:
                        custom_x_label = st.text_input("X-axis Label", x_col)
                    with col3:
                        custom_y_label = st.text_input("Y-axis Label", y_col)
                    custom_legend_label = st.text_input("Legend Label", color_col)

                    # Submit button for form
                    submit_button = st.form_submit_button("Generate Visualization")

                # Plot and export chart upon form submission
                if submit_button:
                    fig, filtered_data, selected_data = plot_and_export_chart(
                        df, x_col, y_col, color_col, size_col, opacity_col, ranges, colormap,
                        custom_title, custom_x_label, custom_y_label, custom_legend_label,
                        sort_by, selection_method, num_pathways, fig_width, fig_height, 
                        min_size, max_size, min_opacity, max_opacity, 
                        size_increase, opacity_increase, size_factor, opacity_factor
                    )
                    display_plot(fig)

            with tab3:
                # Show filtered and selected data
                if selected_data is not None and filtered_data is not None:
                    st.write("### Selected Data for Visualization")
                    st.dataframe(selected_data)
                    
                    st.write("### All Filtered Data")
                    st.dataframe(filtered_data)

                # Export options
                if fig is not None:
                    st.write("### Export Options")
                    export_as = st.selectbox("Select format to export:", ["JPG", "PNG", "SVG", "TIFF"])

                    def save_and_download(format, dpi=600):
                        buffer = BytesIO()
                        fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
                        buffer.seek(0)
                        plt.close()
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

                # PyGWalker Integration
                st.write("### Interactive Data Exploration with PyGWalker")
                init_streamlit_comm()
                pygwalker = StreamlitRenderer(df)
                with st.container():
                    st.write("""
                        <style>
                            iframe {
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                width: 100%;
                                height: 800px !important;
                            }
                        </style>
                        """, unsafe_allow_html=True)
                    pygwalker.explorer()

        else:
            st.warning("Please upload an Excel file to visualize the data.")
