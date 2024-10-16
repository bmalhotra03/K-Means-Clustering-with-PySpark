# Author: Brij Malhotra
# Filename: hw4.py
# Version: 1
# Purpose: K means clustering spark application

from pyspark import SparkContext
import sys

# Initialize Spark Context
sc = SparkContext(appName="KMeansClustering")

# Parameters
K = 5
convergeDist = 0.1
seed = 34

# Helper function to calculate the squared distance between two points
def squared_distance(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

# Parse input file
def parse_line(line):
    fields = line.split(',')
    latitude, longitude = float(fields[3]), float(fields[4])
    if (latitude, longitude) != (0, 0):
        return (latitude, longitude)
    else:
        return None

def kmeans_spark(data, k, converge_dist):
    # Take a random sample of K points as the initial center points
    initial_centers = data.takeSample(False, k, seed)
    centers = sc.parallelize(initial_centers)

    tempDist = float("inf")
    while tempDist > converge_dist:
        # Assign points to the nearest center
        closest = data.map(lambda point: (min([(i, squared_distance(point, center)) for i, center in enumerate(centers.collect())], key=lambda x: x[1])[0], point))
        
        # Calculate the new centers
        # Sum the coordinates of all points assigned to each center and count the num of points per center
        # Calculates the new centers by averaging the summed coords
        point_stats = closest.map(lambda x: (x[0], (x[1], 1))) \
                            .reduceByKey(lambda a, b: ([a[0][0] + b[0][0], a[0][1] + b[0][1]], a[1] + b[1])) \
                            .mapValues(lambda val: [val[0][0] / val[1], val[0][1] / val[1]])

        new_centers = point_stats.map(lambda x: x[1]).collect()
        
        # Calculate the total movement distance of the centers
        tempDist = sum([squared_distance(center, new_centers[i]) for i, center in enumerate(centers.collect())])
        
        # Update the centers
        centers = sc.parallelize(new_centers)
    
    return centers

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: hw4.py <input> <output>", file=sys.stderr)
        exit(-1)
    
    # Get input and output file paths from command-line arguments
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load and filter data
    textfile = sc.textFile(input_path) # RDD of lines 

    # Get the 3rd and 4th values in an RDD (latitude, longitude)
    data = textfile.map(parse_line).filter(lambda pair: pair is not None)

    # Perform K-means clustering
    final_centers = kmeans_spark(data, K, convergeDist)

    # Save the final center points to the output path
    final_centers.saveAsTextFile(output_path)

    sc.stop()