from sailency import get_saliency_maps,saliency_score_part
import numpy as np
import matplotlib.pyplot as plt

def plot_sailency(img,sailency_map,concept_ids,concept_names,coordinates=None):
    """
    Plot the sailency map and the coordinates on the image
    args: img: torch.tensor: The image
    sailency_map: list of np.array: The sailency maps
    concept_ids: list of int: List of concepts that need to be plotted (somewhere between 1 and 5)
    concept_names: list of str: The names of the concepts
    coordinates: list of list of tuples: The coordinates to plot
    """

    num_pictures = len(concept_ids)+1
    fig, ax = plt.subplots(1, num_pictures, figsize=(10*num_pictures, 10))
    
    #Plot original image
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title("Original image",fontsize=30)

    for i,concept_id in enumerate(concept_ids):
        #Plot sailency map
        ax[i+1].imshow(sailency_map[i],cmap='viridis')
        ax[i+1].set_title(concept_names[concept_id],fontsize=30)
        ax[i+1].axis('off')

        #Plot coordinates
        if coordinates is not None:
            for x,y in coordinates[concept_id]:
                ax[i+1].plot(x,y,'ro')
            if len(coordinates[concept_id]) > 0:
                score = saliency_score_part(sailency_map[i],coordinates[concept_id])
                ax[i+1].set_title(f"{concept_names[concept_id]} \n Saliency score: {score:.2f}",fontsize=30)

    plt.show()


#Function to display a DataFrame as a scrollable element in Jupyter Notebook
from IPython.display import display, HTML
# Display DataFrame as a scrollable element
# Display DataFrame as a scrollable element with a smaller max height
def display_scrollable_dataframe(df, max_height=500):  # Set max_height as needed
    display(HTML(f"""
        <div style="max-height: {max_height}px; overflow-y: auto; border: 1px solid #ccc; padding: 5px;">
            {df.to_html(index=True)}
        </div>
    """))

