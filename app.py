import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer

# Set the title and description of the app
st.title("Pathway Significance Visualization with PyGWalker Integration")
st.write("Upload an Excel file and customize your visualization.")

# Function to load data from an uploaded Excel file
def load_data(uploaded_file):
    try:
        data = pd.read_excel(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to generate a custom colormap
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
def create_legend(ax, min_size, max_size, min_opacity, max_opacity):
    # Create a separate legend for size
    size_legend = [Line2D([0], [0], marker='o', color='w', label=f'Size: {int(size)}',
                          markerfacecolor='gray', markersize=size, markeredgecolor='black')
                   for size in np.linspace(min_size, max_size, 3)]

    size_labels = [f"Size: {int(size)}" for size in np.linspace(min_size, max_size, 3)]

    # Create a separate legend for opacity
    opacity_legend = [Line2D([0], [0], marker='o', color='gray', label=f'Opacity: {opacity:.2f}',
                             markerfacecolor='gray', markersize=10, alpha=opacity)
                      for opacity in np.linspace(min_opacity, max_opacity, 3)]

    opacity_labels = [f"Opacity: {opacity:.2f}" for opacity in np.linspace(min_opacity, max_opacity, 3)]

    # Combine legends
    size_legend = ax.legend(size_legend, size_labels, loc='upper right', title="Size Legend", frameon=True, fontsize='small', title_fontsize='small')
    ax.add_artist(size_legend)  # Add the size legend separately to avoid overlap

    ax.legend(opacity_legend, opacity_labels, loc='upper left', title="Opacity Legend", frameon=True, fontsize='small', title_fontsize='small')

# Function to plot and export the chart
def plot_and_export_chart(df, x_col, y_col, color_col, size_col, opacity_col, ranges, colormap, title, x_label, y_label, legend_label, sort_by, selection_method, num_pathways, fig_width, fig_height, min_size, max_size, min_opacity, max_opacity, size_increase, opacity_increase, size_factor, opacity_factor):
    selected_data, filtered_data = get_sorted_filtered_data(df, sort_by, ranges, selection_method, num_pathways)

    # Ensure minimum figure dimensions
    min_width, min_height = 6, 4
    fig_width = max(fig_width, min_width)
    fig_height = max(fig_height, min_height)

    # Create the figure
    plt.clf()  # Clear any existing plots
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Handle size values
    if size_col != "None":
        size_data = pd.to_numeric(selected_data[size_col], errors='coerce').fillna(300)
        size_data = np.clip(size_data * size_factor, min_size, max_size)  # Apply size factor
        if not size_increase:
            size_data = max_size - (size_data - min_size)  # Invert size scaling
    else:
        size_data = 300

    # Handle opacity values
    if opacity_col != "None":
        opacity_data = pd.to_numeric(selected_data[opacity_col], errors='coerce').fillna(0.85)
        opacity_data = np.clip(opacity_data * opacity_factor, min_opacity, max_opacity)  # Apply opacity factor
        if not opacity_increase:
            opacity_data = max_opacity - (opacity_data - min_opacity)  # Invert opacity scaling
    else:
        opacity_data = 0.85

    if pd.api.types.is_numeric_dtype(df[color_col]):
        scatter = ax.scatter(
            x=selected_data[x_col],
            y=selected_data[y_col],
            c=selected_data[color_col],
            cmap=colormap,
            s=size_data,
            alpha=opacity_data,
            marker='o',
            edgecolor='black'
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(legend_label)
    else:
        unique_categories = df[color_col].unique()
        colors = plt.colormaps[colormap](np.linspace(0, 1, len(unique_categories)))
        for category, color in zip(unique_categories, colors):
            category_data = selected_data[selected_data[color_col] == category]
            
            if opacity_col != "None":
                category_opacity = pd.to_numeric(category_data[opacity_col], errors='coerce').fillna(0.85)
                category_opacity = np.clip(category_opacity * opacity_factor, min_opacity, max_opacity)
                if not opacity_increase:
                    category_opacity = max_opacity - (category_opacity - min_opacity)
            else:
                category_opacity = 0.85
                
            if size_col != "None":
                category_size = pd.to_numeric(category_data[size_col], errors='coerce').fillna(300)
                category_size = np.clip(category_size * size_factor, min_size, max_size)
                if not size_increase:
                    category_size = max_size - (category_size - min_size)
            else:
                category_size = 300
                
            ax.scatter(
                x=category_data[x_col],
                y=category_data[y_col],
                label=category,
                color=color,
                s=category_size,
                alpha=category_opacity,
                marker='o',
                edgecolor='black'
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if isinstance(df[y_col].dtype, pd.CategoricalDtype) or df[y_col].dtype == object:
        ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8)

    # Use tight_layout to adjust layout
    fig.tight_layout()

    # Add combined size and opacity legend
    create_legend(ax, min_size, max_size, min_opacity, max_opacity)

    return fig, filtered_data, selected_data

# Function to display the plot in Streamlit
def display_plot(fig):
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
            st.dataframe(df.head(10))

            # Transform columns
            df = transform_columns(df)
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
                default_color_col = next((col for col in columns if col.startswith('-log10(')), columns[0])
                color_col = st.selectbox("Select color column", options=columns, index=columns.index(default_color_col))

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
                                value=(min_val, max_val),
                                key=f"slider_{col}"  # Add a unique key for each slider
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
            fig, filtered_data, selected_data = plot_and_export_chart(
                df, x_col, y_col, color_col, size_col, opacity_col, ranges, colormap,
                custom_title, custom_x_label, custom_y_label, custom_legend_label,
                sort_by, selection_method, num_pathways, fig_width, fig_height, min_size, max_size, min_opacity, max_opacity, size_increase, opacity_increase, size_factor, opacity_factor
            )
            display_plot(fig)

            # Show filtered and selected data
            st.write("### Selected Data for Visualization")
            st.dataframe(selected_data)
            
            st.write("### All Filtered Data")
            st.dataframe(filtered_data)

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
                            width: 140%;
                            height: 800px !important;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                pygwalker.explorer()

            # Export options
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

    else:
        st.warning("Please upload an Excel file to visualize the data.")
