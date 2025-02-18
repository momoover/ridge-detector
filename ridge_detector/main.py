from ridge_detector import RidgeDetector
import pandas as pd
import os

def main():
    # 1. Define your input image and where to save outputs
    image_path = '/Users/M/Desktop/smal_test.jpg'
    output_dir = "/Users/M/Desktop/test_1"
    
    # 2. Instantiate the RidgeDetector with desired parameters
    rd = RidgeDetector(low_contrast=150,  # Lower bound of intensity contrast
                       high_contrast=255,  # Higher bound of intensity contrast
                       min_len=50, # Ignore ridges shorter than this length
                       max_len=0, # Ignore ridges longer than this length, set to 0 for no limit
                       dark_line=True, # Set to True if detecting black ridges in white background, False otherwise
                       estimate_width=True, # Estimate width for each detected ridge point
                       extend_line=True, # Tend to preserve ridges near junctions if set to True
                       correct_pos=True,  # Correct ridge positions with asymmetric widths if set to True
                       )
    
    # 3. Detect lines on the specified image
    rd.detect_lines(image_path)
    
    # 4. Save detailed point-by-point results (e.g., all contour points)
    detailed_df = rd.save_detailed_results(save_dir=output_dir, prefix="example")
    print("Detailed results saved at:", os.path.join(output_dir, "example_detailed_analysis.csv"))
    
    # 5. Plot labeled contours and save figure
    rd.plot_labeled_contours(save_dir=output_dir, prefix="example")
    
    # 6. Compute additional analysis
    #stats_df = rd.compute_statistical_analysis()
    #shape_df = rd.compute_shape_characteristics()
    network_summaries, network_nodes = rd.compute_network_analysis()
    
    # 7. (Optional) Create advanced interactive visualizations (Plotly)
    fig = rd.create_advanced_visualizations(save_dir=output_dir, image_path=image_path,prefix="example")
    
    # 8. Save results (images + summary CSV)
    final_df = rd.save_results(
        save_dir=output_dir, 
        prefix="example", 
        make_binary=True, 
        draw_junc=True, 
        draw_width=True
    )
    
    # 9. Print or combine analysis dataframes as needed
    #print("Statistical Analysis:\n", stats_df.head())
    #print("Shape Characteristics:\n", shape_df.head())
    print("Network Summaries:\n", network_summaries)
    print("Node-level Network Metrics:\n", network_nodes.head())
    print("Final Summary Data:\n", final_df.head())

if __name__ == "__main__":
    main()

