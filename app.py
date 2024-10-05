import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pygwalker.api.streamlit import init_streamlit_comm, StreamlitRenderer
from functools import lru_cache
import time
import re




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

# Optimized normalize_data function
@np.vectorize
def normalize_data_vectorized(value, min_val, max_val, factor=1.0, increase=True):
    if np.isnan(value):
        return (min_val + max_val) / 2
    norm_value = (value - min_val) / (max_val - min_val)
    if not increase:
        norm_value = 1 - norm_value
    return np.clip(norm_value * factor, 0, 1) * (max_val - min_val) + min_val

# Optimized get_sorted_filtered_data function
@st.cache_data
# Updated get_sorted_filtered_data to handle "Allow more rows" correctly
def get_sorted_filtered_data(df, sort_by, ranges, selection_method, num_pathways, allow_more_rows):
    filtered_data = df.copy()
    discarded_data = {}

    # Apply filters to the data
    for col, (min_val, max_val) in ranges.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            discarded = filtered_data[(filtered_data[col] < min_val) | (filtered_data[col] > max_val)]
            filtered_data = filtered_data[(filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)]
            discarded_data[col] = discarded

    filtered_data = filtered_data.sort_values(by=sort_by, ascending=False)

    # Select data based on the selection method
    if len(filtered_data) >= num_pathways:
        if selection_method == 'Top (Highest Values)':
            selected_data = filtered_data.head(num_pathways)
        elif selection_method == 'Bottom (Lowest Values)':
            selected_data = filtered_data.tail(num_pathways)
        elif selection_method == 'Both Ends':
            half_num = num_pathways // 2
            selected_data = pd.concat([
                filtered_data.head(half_num),
                filtered_data.tail(half_num)
            ])
        else:  # Middle
            start_idx = (len(filtered_data) - num_pathways) // 2
            selected_data = filtered_data.iloc[start_idx:start_idx + num_pathways]
    else:
        selected_data = filtered_data.copy()

    # If 'allow_more_rows' is True, attempt to compensate by adding rows from discarded data
    if allow_more_rows and len(selected_data) < num_pathways:
        remaining_num = num_pathways - len(selected_data)
        extra_data = pd.concat(discarded_data.values(), axis=0).sort_values(by=sort_by, ascending=False)
        extra_data = extra_data.head(remaining_num)
        selected_data = pd.concat([selected_data, extra_data])

    return selected_data, filtered_data, discarded_data


def display_plot(fig):
    if fig is not None:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        buf.seek(0)
        st.image(buf)


# Updated create_legends function
def create_legends(ax, sizes, opacities, size_col, opacity_col, legend_fontsize):
    legend_elements = []
    legend_labels = []
    
    if size_col != "None":
        size_values = [np.min(sizes), np.median(sizes), np.max(sizes)]
        for size in size_values:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='gray', markersize=np.sqrt(size),
                                         markeredgecolor='black', linestyle='None'))
            legend_labels.append(f'{size_col}: {int(size)}')

    if opacity_col != "None":
        opacity_values = [np.min(opacities), np.median(opacities), np.max(opacities)]
        for opacity in opacity_values:
            legend_elements.append(Line2D([0], [0], marker='o', color='gray',
                                         markerfacecolor='gray', markersize=10,
                                         alpha=opacity, linestyle='None'))
            legend_labels.append(f'{opacity_col}: {opacity:.2f}')

    if legend_elements:
        leg = ax.legend(legend_elements, legend_labels, loc='center left', 
                  bbox_to_anchor=(1.05, 0.5), frameon=True, title="Size and Opacity",
                  fontsize=legend_fontsize)
        plt.setp(leg.get_title(), multialignment='center', fontsize=legend_fontsize)

# Updated plot_and_export_chart function
def plot_and_export_chart(df, x_col, y_col, color_col, size_col, opacity_col, ranges, 
                         colormap, title, x_label, y_label, legend_label, sort_by, 
                         selection_method, num_pathways, fig_width, fig_height, 
                         min_size, max_size, min_opacity, max_opacity, 
                         size_increase, opacity_increase, size_factor, opacity_factor,
                         show_annotation_id, annotation_sort, annotation_font, annotation_size,
                         annotation_alignment, legend_fontsize, allow_more_rows, sort_order_ascending=True):
    try:
        # Prepare the filtered and sorted data
        selected_data, filtered_data, discarded_data = get_sorted_filtered_data(
            df, sort_by, ranges, selection_method, num_pathways, allow_more_rows
        )

        if selected_data.empty:
            st.warning("No data to display after applying filters.")
            return None, filtered_data, selected_data, discarded_data

        # Ensure y_col exists in selected_data
        if y_col not in selected_data.columns:
            st.error(f"Column '{y_col}' not found in the filtered data.")
            return None, filtered_data, selected_data, discarded_data

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Apply clean_pathway_name to y_col if show_annotation_id is False
        def clean_pathway_name(name):
            name = re.sub(r'\(R-HSA-\d+\)', '', name)  # Remove R-HSA IDs
            name = re.sub(r'\(DOID:\d+\)', '', name)   # Remove DOID IDs
            return name.strip()

        if not show_annotation_id:
            selected_data[y_col] = selected_data[y_col].apply(clean_pathway_name)

        # Prepare annotations
        annotations = selected_data[y_col].tolist()

        # Sorting annotations and corresponding data
        if annotation_sort == 'p-value':
            sort_order = selected_data[color_col].sort_values(ascending=sort_order_ascending).index
        elif annotation_sort == 'name_length':
            sort_order = selected_data[y_col].str.len().sort_values(ascending=sort_order_ascending).index
        else:
            sort_order = selected_data.index

        # Safely reorder annotations and other data
        annotations = selected_data.loc[sort_order, y_col].tolist()
        x_values = selected_data.loc[sort_order, x_col].values
        y_values = np.arange(len(annotations))

        # Initialize sizes and opacities
        sizes = np.full(len(selected_data), (min_size + max_size) / 2) if size_col == "None" else None
        opacities = np.full(len(selected_data), (min_opacity + max_opacity) / 2) if opacity_col == "None" else None

        # Handle size values
        if size_col != "None":
            sizes = pd.to_numeric(selected_data.loc[sort_order, size_col], errors='coerce')
            sizes = normalize_data_vectorized(sizes, min_size, max_size, size_factor, size_increase)

        # Handle opacity values
        if opacity_col != "None":
            opacities = pd.to_numeric(selected_data.loc[sort_order, opacity_col], errors='coerce')
            opacities = normalize_data_vectorized(opacities, min_opacity, max_opacity, opacity_factor, opacity_increase)

        # Plot the data
        scatter = ax.scatter(x_values, y_values,
                            c=selected_data.loc[sort_order, color_col],
                            cmap=colormap,
                            s=sizes,
                            alpha=opacities,
                            edgecolors='black')

        # Set y-axis ticks and labels with proper alignment and font
        ax.set_yticks(y_values)
        ax.set_yticklabels(annotations, fontsize=annotation_size, fontfamily=annotation_font, ha=annotation_alignment)

        # Adjust layout to make room for labels outside the plot area
        plt.subplots_adjust(left=0.45, right=0.8)  # Increased left margin for pathway names

        # Set X and Y axis labels
        ax.set_xlabel(x_label, fontsize=legend_fontsize)
        ax.set_ylabel(y_label, fontsize=legend_fontsize)
        ax.set_title(title, fontsize=legend_fontsize)

        # Invert Y-axis to align with the expected order
        ax.invert_yaxis()

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(legend_label, fontsize=legend_fontsize)

        # Adjust axis limits to ensure circles are not too close to the Y-axis
        ax.set_xlim([min(x_values) - 20, max(x_values) + 20])  # Adding padding to the x-axis
        ax.set_ylim([min(y_values) - 1, max(y_values) + 1])

        # Add size and opacity legends
        create_legends(fig, sizes, opacities, size_col, opacity_col, legend_fontsize)

        plt.tight_layout()
        return fig, filtered_data, selected_data, discarded_data
    except Exception as e:
        st.error(f"Error in plot_and_export_chart: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, {}

        
        
def create_legends(ax, sizes, opacities, size_col, opacity_col, legend_fontsize):
    # Create legend for size and opacity by plotting invisible reference points
    legend_elements = []
    if size_col != "None":
        for size in [np.min(sizes), np.median(sizes), np.max(sizes)]:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor='gray', markersize=np.sqrt(size), 
                                              linestyle='None', label=f'{size_col}: {int(size)}'))
    
    if opacity_col != "None":
        for opacity in [np.min(opacities), np.median(opacities), np.max(opacities)]:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray',
                                              markerfacecolor='gray', markersize=10, 
                                              alpha=opacity, linestyle='None', label=f'{opacity_col}: {opacity:.2f}'))
    
    # Add the legends to the plot
    if legend_elements:
        ax.legend(handles=legend_elements, loc='best', fontsize=legend_fontsize, title="Size and Opacity")
        
def create_legends(fig, sizes, opacities, size_col, opacity_col, legend_fontsize):
    legend_elements = []
    legend_labels = []

    if size_col != "None":
        size_values = [np.min(sizes), np.median(sizes), np.max(sizes)]
        for size in size_values:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor='gray', markersize=np.sqrt(size),
                                          markeredgecolor='black', linestyle='None'))
            legend_labels.append(f'{size_col}: {int(size)}')

    if opacity_col != "None":
        opacity_values = [np.min(opacities), np.median(opacities), np.max(opacities)]
        for opacity in opacity_values:
            legend_elements.append(Line2D([0], [0], marker='o', color='gray',
                                          markerfacecolor='gray', markersize=10,
                                          alpha=opacity, linestyle='None'))
            legend_labels.append(f'{opacity_col}: {opacity:.2f}')

    # Add legends to the plot
    if legend_elements:
        leg = fig.legend(legend_elements, legend_labels, loc='center left',
                         bbox_to_anchor=(1.05, 0.5), frameon=True, title="Size and Opacity",
                         fontsize=legend_fontsize)
        plt.setp(leg.get_title(), multialignment='center', fontsize=legend_fontsize)


def get_sorted_filtered_data(df, sort_by, ranges, selection_method, num_pathways, allow_more_rows):
    filtered_data = df.copy()
    discarded_data = {}

    for col, (min_val, max_val) in ranges.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            discarded = filtered_data[(filtered_data[col] < min_val) | (filtered_data[col] > max_val)]
            filtered_data = filtered_data[(filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)]
            discarded_data[col] = discarded

    filtered_data = filtered_data.sort_values(by=sort_by, ascending=False)

    if allow_more_rows:
        num_pathways = max(num_pathways, len(filtered_data))  # Set num_pathways to the total length if allowed

    # Selection based on method
    if selection_method == 'Top (Highest Values)':
        selected_data = filtered_data.head(num_pathways)
    elif selection_method == 'Bottom (Lowest Values)':
        selected_data = filtered_data.tail(num_pathways)
    elif selection_method == 'Both Ends':
        half_num = num_pathways // 2
        selected_data = pd.concat([
            filtered_data.head(half_num),
            filtered_data.tail(half_num)
        ])
    else:  # Middle
        start_idx = (len(filtered_data) - num_pathways) // 2
        selected_data = filtered_data.iloc[start_idx:start_idx + num_pathways]

    return selected_data, filtered_data, discarded_data
        
# Main execution
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Pathway Significance Visualization")
    st.title("Pathway Significance Visualization with PyGWalker Integration")
    st.write("Upload an Excel file and customize your visualization.")

    uploaded_file = st.file_uploader("Upload your data file", type=["xlsx"])

    if uploaded_file is not None:
        start_time = time.time()
        df = load_data(uploaded_file)
        load_time = time.time() - start_time
        st.write(f"Data loading time: {load_time:.2f} seconds")
        
        if df is not None:
            st.write("Data loaded successfully!")

            # Initialize variables
            fig = None
            filtered_data = None
            selected_data = None
            
            # Apply -log10 transformation
            tab1, tab2, tab3 = st.tabs(["Data Preview", "Visualization Settings", "Interactive Options"])
            
            with tab1:
                st.dataframe(df.head(10))
                
                # Move p-value column selection to tab 1
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

with tab2:
    allow_more_rows = st.checkbox("Allow more rows if filters reduce selection below specified number")
    # Move all the setting widgets here
    with st.form("visualization_settings"):
        columns = df.columns.tolist()

        # Column selection
        st.write("### Select Visualization Columns")
        col1, col2, col3 = st.columns(3)

        with col1:
            x_col = st.selectbox("Select X-axis column", options=columns, index=columns.index("Enrichment") if "Enrichment" in columns else 0)
        with col2:
            y_col = st.selectbox("Select Y-axis column", options=columns, index=columns.index("Annotation Name") if "Annotation Name" in columns else 0)
        with col3:
            color_col = st.selectbox("Select color column", options=columns, index=columns.index("-log10(p-value)") if "-log10(p-value)" in columns else 0)

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
            min_size = st.slider("Min size", min_value=10, max_value=1000, value=50)
        with col2:
            max_size = st.slider("Max size", min_value=10, max_value=1000, value=500)
        with col3:
            min_opacity = st.slider("Min opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        with col4:
            max_opacity = st.slider("Max opacity", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

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
            num_pathways = st.slider("Number of pathways to show", min_value=1, max_value=len(df), value=min(10, len(df)))

        # New: Sorting Order (Ascending/Descending)
        col1, col2 = st.columns(2)
        with col1:
            sort_options = st.radio("Sort Order", options=["Ascending", "Descending"], index=0)
            sort_order_ascending = True if sort_options == "Ascending" else False

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

        # Annotation options
        st.write("### Annotation Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_annotation_id = st.checkbox("Show Annotation IDs", value=False)
        with col2:
            annotation_sort = st.selectbox("Sort annotations by", ["p-value", "name_length", "none"])
        with col3:
            annotation_alignment = st.selectbox("Annotation alignment", ["left", "right", "center"])

        col1, col2, col3 = st.columns(3)
        with col1:
            annotation_font = st.selectbox("Annotation font", ["Arial", "Times New Roman", "Courier"])
        with col2:
            annotation_size = st.slider("Annotation font size", 6, 20, 10)
        with col3:
            legend_fontsize = st.slider("Legend font size", 6, 20, 10)

        # Submit button for form
        submit_button = st.form_submit_button("Generate Visualization")

        # Generate the visualization if the form is submitted
        if submit_button:
            try:
                result = plot_and_export_chart(
                    df, x_col, y_col, color_col, size_col, opacity_col, ranges, colormap,
                    custom_title, custom_x_label, custom_y_label, custom_legend_label,
                    sort_by, selection_method, num_pathways, fig_width, fig_height,
                    min_size, max_size, min_opacity, max_opacity,
                    size_increase, opacity_increase, size_factor, opacity_factor,
                    show_annotation_id, annotation_sort, annotation_font, annotation_size,
                    annotation_alignment, legend_fontsize, allow_more_rows, sort_order_ascending
                )

                if isinstance(result, tuple) and len(result) == 4:
                    fig, filtered_data, selected_data, discarded_data = result
                    if fig is not None:
                        st.pyplot(fig)

                        # Display discarded rows information
                        st.write("### Rows Discarded Due to Filtering")
                        if discarded_data:
                            for col, discarded in discarded_data.items():
                                st.write(f"Discarded by {col} filter:")
                                st.dataframe(discarded)
                        else:
                            st.write("No rows were discarded by filtering.")
                    else:
                        st.warning("No visualization could be generated with the current settings.")
                else:
                    st.error("Unexpected result from plot_and_export_chart function.")
            except Exception as e:
                st.error(f"An error occurred while generating the visualization: {str(e)}")
                st.error("Please check your inputs and try again.")        
        
                        
                # Show selected data in tab 2
                if selected_data is not None:
                    st.write("### Selected Data for Visualization")
                    st.dataframe(selected_data)

                # Export options in tab 2
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

                # Show discarded rows due to filtering
                if filtered_data is not None:
                    discarded_data = df[~df.index.isin(filtered_data.index)]
                    if not discarded_data.empty:
                        st.write("### Rows Discarded Due to Filtering")
                        st.dataframe(discarded_data)

            with tab3:
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
