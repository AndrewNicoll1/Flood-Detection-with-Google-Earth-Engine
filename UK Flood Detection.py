import ee
ee.Initialize()
import os
import geemap
import pandas as pd
import numpy as np
from termcolor import colored
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split

#### CONTROL PANEL ####

start =  '2019-10-1'                                             # Search start date 
finish = '2020-2-1'                                              # Search end date 

region_1 = 45                                                    # Start region  
region_2 = 45                                                    # End region 

percentage = 0.01                                                # Percentage threshold for flood pixels:
                                                                 # 0.01 is sufficient for small regions
                                                  
cloud_perc = 20                                                  # Cloud pixel percentage: <= 20    
    
num_cols = 5                                                     # Number of columns
num_rows = 13                                                    # Number of rows
    
subject = [(-2.4702, 54.4442), (-0.4927, 54.4442),               # Area of interest: England (rectangle)
           (-0.4927, 50.9165), (-2.4702, 50.9165)]  

# Example parameters: 

# start = 2019-10-1    num_cols = 5      percentage = 0.01   
# finish = 2020-2-1    num_rows = 13     cloud_perc = 20

# (1) subject = England          (2) subject = England           (3) subject = England
#     idx_1 = 45                     idx_1 = 17                      idx_1 = 17
#     idx_2 = 45                     idx_2 = 17                      idx_2 = 23


#### FUNCTIONS ####

Map = geemap.Map()     # Create Map

datelist = []          # List of dates of flood optical images
imageList = []         # List of flood optical images

predateList = []       # List of dates of pre flood optical images
pre_imageList = []     # List of pre flood optical images
maskList = []          # List of flood masks

SAR_imageList = []     # List of flood SAR images
preSAR_imageList = []  # List of pre flood SAR images

# Function to split UK (or other) into smaller regions 
def grid(nx, ny, subject):

    subject = Polygon(subject)

    minx, miny, maxx, maxy = subject.bounds
    dx = (maxx - minx) / nx  
    dy = (maxy - miny) / ny  

    longitude_split = [LineString([(minx, miny + i*dy), (maxx, miny + i*dy)]) for i in range(ny)]
    latitude_split = [LineString([(minx + i*dx, miny), (minx + i*dx, maxy)]) for i in range(nx)]
    regions = longitude_split + latitude_split

    result = subject
    for s in regions:
        result = MultiPolygon(split(result, s))

    parts = [list(part.exterior.coords) for part in result.geoms]

    geometries = []

    for i in parts:

        lon_1 = i[0][0]
        lat_1 = i[0][1]

        lon_3 = i[2][0]
        lat_3 = i[2][1]

        geometry = ee.Geometry.Rectangle([[lon_1, lat_1], [lon_3, lat_3]])
        geometries.append(geometry) 
        
    return geometries

# Function to remove images that do not cover the full region of interest
def complete(Image, geometry, band):

    totPixels = ee.Number(ee.Image(1).reduceRegion(**{
        'reducer': ee.Reducer.count(),
        'geometry': geometry,
        'scale': 30}).values().get(0))

    actPixels = ee.Number(Image.select(band).reduceRegion(**{
        'reducer': ee.Reducer.count(),
        'scale': 30,
        'geometry': geometry}).values().get(0))

    pcPix = actPixels.divide(totPixels).multiply(100).getInfo()

    return(round(pcPix, 0))

def ndwi(image):
        ndwi = image.normalizedDifference(['B3', 'B8A'])\
        .rename('NDWI')\
        .copyProperties(image, ['system:time_start'])
        return image.addBands(ndwi)
    
    # Reomve images with same dates in collections
    
def removeDuplicate(img_col):
    img_col = img_col.sort('system:time_start')
    img_col = img_col.toList(collection.size())

    length = img_col.size().getInfo()

    filtered_images = []

    for dd in range(0, length-1):

        image = ee.Image(img_col.get(dd))
        image2 = ee.Image(img_col.get(dd+1))

        dateTime = ee.Date(image.get('system:time_start'))
        date = dateTime.format().getInfo()[0:10]

        dateTime2 = ee.Date(image2.get('system:time_start'))
        date2 = dateTime2.format().getInfo()[0:10]

        if date2 != date:
            filtered_images.append(image)

    img_col = ee.ImageCollection.fromImages(filtered_images)
        
    return img_col

# Function iterates over the NDWI collection and computes the difference in consecutive images. 

def NDWIdiff(date):

    #get the image corresponding to the date
    currentImage = NDWI_collection.filter(ee.Filter.eq('system:time_start', date)).first();

    #Now we have to get the 'previous' image in the collection
    indexCurrent = dateList.indexOf(date);
    indexPrevious = indexCurrent.subtract(1);
    datePrevious = dateList.get(indexPrevious);
    previousImage = NDWI_collection.filter(ee.Filter.eq('system:time_start', datePrevious)).first(); 

    #Subtract the current image from the previous
    diffImage = currentImage.subtract(previousImage).select(['NDWI'], ['NDWIdiff']);

    return currentImage.addBands(diffImage).set('system:index_previous', previousImage.get('system:index'))

def floodPixels(image):
    def floodPixels(value):
        return image.select('NDWIdiff').gt(ee.Number(value))
    
    # Threshold for NDWI difference 
    threshold = [0.8] 

    water = ee.ImageCollection.fromImages(ee.List(threshold).sort()\
        .map(floodPixels)).reduce('sum').rename('water')\
        .reproject(image.projection(), None, image.projection().nominalScale()); 

    return image.addBands(water)


#### MAIN ####

# Create regions

geometries = grid(num_cols, num_rows ,subject)
num = len(geometries) 

num_select = region_2-region_1+1
region_area = geometries[0].area().divide(1000*1000).getInfo()

print("Number of regions:", num)
print("Number of regions selected: {}".format(num_select))
print("")
print("Area of single region: {}km\u00b2".format(round(region_area, 0)))
print("Total area of regions selected: {}km\u00b2".format(round(region_area * num_select, 0)))
print("")
print("Start search:", start)
print("Stop search:", finish)
print("")

total_flood = 0
number = 1

for geom in range(region_1-1, region_2): 
    
    print("({}) REGION".format(number), geom+1)
    print("")
    
    aoi = geometries[geom]
    
    # Creating SAR image collection over region
    
    SAR = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
        .filterBounds(aoi)
    
    # Creating optical image collection over region
    
    start = ee.Date(start)
    finish = ee.Date(finish)
    
    collection = ee.ImageCollection("COPERNICUS/S2")\
        .filterBounds(aoi)\
        .filterDate(start, finish)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))\

    # Remove images with duplicate dates
    collection = removeDuplicate(collection)
    
    # Create the NDWI images 
    collection = collection.map(ndwi)

    # NDWI image collection
    NDWI_collection = collection.select('NDWI').sort('system:time_start')

    # Remove first data
    dateList = ee.List(NDWI_collection.aggregate_array('system:time_start')).slice(1)

    # Collection of NDWI difference images
    difference_collection = ee.ImageCollection.fromImages(dateList.map(NDWIdiff)) 

    # Thresholding the NDWI difference images to classify flooded pixels
    threshold_collection = difference_collection.map(floodPixels) 

    # Thresholded collection to list of images
    threshold_collection = threshold_collection.toList(threshold_collection.size())

    #Convert orinal optical collection to list and sort by date
    collection = collection.sort('system:time_start')
    collection = collection.toList(collection.size()) 

    length = collection.size().getInfo() - 1   

    print("Number of optical images:", length)

    currentDate = []
    flood_count = 0
    
    for i in range(0, length):

        optical = ee.Image(collection.get(i+1)).clip(aoi)
        pre_optical = ee.Image(collection.get(i)).clip(aoi)
        flood_mask = ee.Image(threshold_collection.get(i)).clip(aoi).select('water')
        
        ee_date = ee.Date(optical.get('system:time_start'))
        date = ee_date.format().getInfo()

        pre_ee_date = ee.Date(pre_optical.get('system:time_start'))
        pre_date = pre_ee_date.format().getInfo()

        meanDict = flood_mask.reduceRegion(**{
            'reducer': ee.Reducer.mean(),
            'geometry': aoi,
            'scale': 30, 
            'bestEffort': True 
            
        })

        try:
            mean = meanDict.get('water')
            flood_perc = mean.getInfo()*100

            print("")
            print("IMAGE:", i+1)
            print("DATE:", date[0:10])
            print("Flood percentage:", round(flood_perc, 3))

            if flood_perc >= percentage: #Image is a flood image if True  
                
                print("")
                print(colored("FLOODS DETECTED", 'blue'))
                
                # Collecting flood and preflood SAR images 
                
                date = date[0:10]
                pre_date = pre_date[0:10]
                
                flood_images = SAR\
                    .filterDate(ee.Date(date), ee.Date(date).advance(2, 'day'))
                
                preflood_images = SAR\
                    .filterDate(ee.Date(pre_date).advance(-2, 'day'), ee.Date(pre_date))
                
                f_image = ee.Image(flood_images.mosaic()).clip(aoi)
                p_image = ee.Image(preflood_images.mosaic()).clip(aoi)
                
                # Ensure images are useable for training and/or classification
                
                perc_1 = complete(optical, aoi, 'NDWI')
                perc_2 = complete(pre_optical, aoi, 'NDWI')
                perc_3 = complete(f_image, aoi, 'VH')
                perc_4 = complete(p_image, aoi, 'VH')
                
                # Check that optical images are the same size, and SAR images
                    
                if (perc_1 == perc_2) and perc_3 == 100 and perc_4 == 100:

                    print(colored("IMAGES ACCEPTED", 'green'))

                    datelist.append(date) 
                    predateList.append(pre_date)

                    currentDate.append(date)

                    imageList.append(optical)
                    pre_imageList.append(pre_optical)
                    
                    flood_mask = flood_mask.mask(flood_mask)
                    maskList.append(flood_mask)

                    SAR_imageList.append(f_image)
                    preSAR_imageList.append(p_image)

                    flood_count += 1
                    total_flood += 1  

                else:

                    print(colored("IMAGES DECLINED", 'red'))
        except:
            
            print(colored("ERROR", 'red'))
            pass


    print("")
    print("Flood dates:", currentDate)
    print("Number of floods:", flood_count)
    print("")
    
    number += 1
    
print("FINISHED ALL REGIONS")
print("Total floods:", total_flood)

# Create optical and mask collections

flood_collection = ee.ImageCollection.fromImages(imageList)
preflood_collection = ee.ImageCollection.fromImages(pre_imageList)
mask_collection = ee.ImageCollection.fromImages(maskList)

# Create SAR collections

SAR_flood_collection = ee.ImageCollection.fromImages(SAR_imageList)
SAR_preflood_collection = ee.ImageCollection.fromImages(preSAR_imageList)

#Map flood images & masks

for im in range(0, len(datelist)):
    
    # Flood optical
    Map.addLayer(ee.Image(imageList[im]), {'bands': ['B8', 'B11', 'B4'], 'min':0, 'max':3000}, '(Img {}) Optical'.format(im+1), True)
    # Preflood optical
    Map.addLayer(ee.Image(pre_imageList[im]), {'bands': ['B8', 'B11', 'B4'], 'min':0, 'max':3000}, '(Img {}) Pre optical'.format(im+1), True)
    # Flooded pixels
    Map.addLayer(ee.Image(maskList[im]), {"palette": 'blue'}, '(Img {}) Flooded pixels'.format(im+1), True)
    # Flood SAR
    Map.addLayer(ee.Image(SAR_imageList[im]), {'bands': 'VH', 'min': -25, 'max': 0}, '(Img {}) Flood SAR'.format(im+1), True)
    # Preflood SAR
    Map.addLayer(ee.Image(preSAR_imageList[im]), {'bands': 'VH', 'min': -25, 'max': 0}, '(Img {}) Preflood SAR'.format(im+1), True)
    
subject_aoi = ee.Geometry.Rectangle([[subject[0][0], subject[0][1]], [subject[2][0], subject[2][1]]])
Map.centerObject(subject_aoi, zoom=6)
Map   
