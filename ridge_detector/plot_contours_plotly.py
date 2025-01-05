
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np

# 1) Load your CSV data
#    Make sure your CSV has columns like:
#    Contour_ID, Point_ID, X_Coord, Y_Coord, Angle, Total_Width, ...
df = pd.read_csv("/Users/matthiasoverberg/Desktop/test_ridge/example_detailed_analysis.csv")

def take_every_xth(g):
    # sort by Point_ID so we keep ascending order,
    # then take every 10th row
    g_sorted = g.sort_values('Point_ID')
    return g_sorted.iloc[::5]

# Group by Contour_ID, apply the above function, reassemble
df_downsampled = df.groupby("Contour_ID", group_keys=False).apply(take_every_xth)

# 2) Load the background image
img_path = "screenshot_test.jpg"
img = Image.open(img_path)

# Extract the pixel size of the loaded image
width, height = img.size  # width (x-dim), height (y-dim)

# Convert PIL image to a Plotly-compatible image source
# (Plotly can often take PIL Image objects directly, but
#  if needed, you can do e.g. np.array(img) or a base64-encoded string.)
img_source = img  # or np.array(img), etc.

# 3) Create Plotly figure
fig = go.Figure()

# 4) Add the image as background (layer='below')
#    We assume the bottom-left corner of the image is at (x=0, y=0)
#    and the top-right corner is at (x=width, y=height).
fig.add_layout_image(
    dict(
        source=img_source,
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=width,
        sizey=height,
        sizing="stretch",
        opacity=0.5,      # transparency
        layer="below"     # so the image is behind the data
    )
)

# 5) Plot each contour as a separate Scatter trace
#    Group by Contour_ID, then connect points in ascending Point_ID
for contour_id, grp in df_downsampled.groupby("Contour_ID"):
    grp_sorted = grp.sort_values("Point_ID")
    x_vals = grp_sorted["X_Coord"]
    y_vals = grp_sorted["Y_Coord"]
    
    # Add a line+markers trace for the contour
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=-y_vals,
            mode='lines+markers',
            name=f"Contour {contour_id}",
            line=dict(width=1)
        )
    )
    
    # 6) (Optional) Draw perpendicular lines for width
    #    We'll assume 'Angle' is in radians and 'Total_Width' is the length
    for row in grp_sorted.itertuples():
        px = row.X_Coord
        py = row.Y_Coord
        w = row.Total_Width / 2.0  # half width for each direction
        angle = row.Angle         # in radians (adjust if in degrees)
        
        # Perpendicular direction: angle + π/2
        perp_angle = angle 
        
        dx = w * np.cos(perp_angle)
        dy = w * np.sin(perp_angle)
        
        x0 = px - dx
        y0 = py - dy
        x1 = px + dx
        y1 = py + dy
        
        # Add a line shape for each perpendicular segment
        fig.add_shape(
            type="line",
            x0=x0,
            y0=-y0,
            x1=x1,
            y1=-y1,
            line=dict(color="red", width=1),
            opacity=0.5,
            layer="above"  # above the image
        )
'''
fig.update_xaxes(
    range=[0, width],   # covers entire image width
)
fig.update_yaxes(
    range=[0, height],  # covers entire image height
    # If you want the origin at bottom-left in standard Cartesian coordinates:
    scaleanchor="x",    # lock the aspect ratio to the x-axis
    scaleratio=1
)
'''
# If your image’s (0,0) is top-left, you may want:
# fig.update_yaxes(autorange="reversed")

fig.update_layout(
    width=800,          # final figure width in pixels (for display)
    height=600,         # final figure height in pixels (for display)
    margin=dict(r=10, l=10, t=30, b=10),
    title="Plotly Contours with Background Image"
)
# 8) Save to HTML
fig.write_html("output.html")

# 9) Save to SVG (requires kaleido: pip install kaleido)
#fig.write_image("output.svg")

# Optionally, display the figure in an interactive window (e.g., in Jupyter)
# fig.show()
