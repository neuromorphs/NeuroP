from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def connect_coordinates(coord1, coord2, map):
    x1, y1 = coord1
    x2, y2 = coord2
    print(type(coord2))
    map.drawgreatcircle(x1, y1, x2, y2, linewidth=2, color='b')

def create_world_map():
    # Create a Basemap instance with desired projection and map boundaries
    map = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)

    # Draw coastlines and country borders
    map.drawcoastlines()
    map.drawcountries()

    # Add your coordinate connections
    connect_coordinates((0.1, 0.1), (40.1, 30.1), map)
    connect_coordinates((40, 30), (-100, 50), map)
    connect_coordinates((-100, 50), (0, 0), map)

    # Display the plot
    plt.show()

# Call the function to create the world map
create_world_map()