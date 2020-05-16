A high level model of the workflow of our clothes reccomendation engine can be broken down roughly into the following parts:

- User input interface
- A microservice which constantly webscrapes new clothes as a background process
- An object detection service
- A shared api to connect the different teams
- A recommendation engine

These parts can be mapped together like below:

![Service workflow](https://github.com/DanielJohnHarty/DSTI_shazzam_for_clothes_image_processor/blob/master/Documents/imgs/user_shazzam_process.png)

Team three were responsible to deliver two main objectives:

- generate recommendations
- create an api to simplify the repetetive, technical tasks for all teams in the pipeline

Let's talk through our path to delivering on each of these two objectives.

## A shared api to connect the different teams

The API covers 2 areas: database api and image preprocessing

**Database Api**
To allow different team members to share the same data store, we created straightforward relational database hosted on AWS. The stored data are simply urls, coordinates and labels. We retain no images after the processing is done as we can perform any task on the stored data and lazily download images when needed (it's funny to think that all those image servers are working for us for free in this context).

    ```python
    # Import the api
    import ShazzamForClothes.database_api as db_api

    # Instantiate the Database class as an object
    db = db_api.Database()

    # Open a conection to the database
    db.open_connection()

    # Use the Database.run_query method to query the database.
    sql="SELECT * FROM RAW_IMAGES;"
    results = db.run_query(sql)

    # Iterate through the results
    for result in results:
        print(result)
    ```

**Image preprocessing API**

Image classification models are *fussy eaters*.

The image needs to be of a particular object, to be a particular size, resolution, compression standard and file type. The raw images provided as input for the model come from a web scraping microservice running on the web where it's rare to find the rxact picture you want in the size and format you want. So we built a python module acting as a wrapper around the  ***Pillow*** library. The suite includes individual fucntions to convert images, resize, change the resolution and colour mode, renale andother things. The main purpose of the module however are the functions **get_subsection_of_image and standardise_img**, and **standardise_img**, which together can transform any scraped image into a valid input for our image classification system.



## A recommendation engine

We built a recommendation system for images of clothing, considering each image as a data point in high dimensional space. This requires the ability to produce what we call an embedding or encoding for each image. An embedding of an image is a small vector representation which carries the maximum possible information from the image. Once each image is properly represented as data point we can use the K-Nearest Neighbours algorithm to find closest data points, each of which represent an image from the set. That is to say, that the closer two points are, the more similar they are.

Here is a brief summary of the steps we followed:

 - Part 1: Building an embedding model 
 - Part 2: Storing images as embeddings
 - Part 3: Searching for closest images among the embeddings database using KNN


*Building an embedding model*

First we used a pre-trained convolutional neural network (CNN) called VGG16 and available on keras. We removed the last 2 dense layers so that this model output 4096 dimensional vector. When we feed an image of cloth into this network we get an embedding of size 4096.

![image  of architechture](https://github.com/DanielJohnHarty/DSTI_shazzam_for_clothes_image_processor/blob/master/Documents/imgs/archivgg.png)

To assess the quality of our embedding we use a visualization tool called t-SNE. 

![t-SNE](https://github.com/DanielJohnHarty/DSTI_shazzam_for_clothes_image_processor/blob/master/Documents/imgs/fashion_tsne.png)

So far our neural network is a general model and is not specialized into discriminating images of clothes. In order to produce better embeddings we fine-tuned our model, that is to say that we trained VGG16 starting from the pretrained version. Actually there are 2 steps:
- first we removed ouput layer and replace it by a new one adapted to our task a softmax layer of size 11. We froze every layers except the last layer and perform training.
- Then we perform once again training step but with every layers.

We used the visualization tool called t-SNE again to reasses the quality of our embedding.

![t-SNE](https://github.com/DanielJohnHarty/DSTI_shazzam_for_clothes_image_processor/blob/master/Documents/imgs/fashion_tsne_ft11111.png)

*Storing images as embeddings*

We use our embedding model over our whole databse of images. We store embeddings as a hdf5 file.

*Searching for closest images among the embeddings database using KNN*

On receiving an image of some item of clothing from a user, we compute its embedding. Here there are several ways to proceed. Either the user provides a specific category of clothes and and we only have to search among clothes of that specific category. Or there is no category provided, in which case we then need to use our fine tuned VGG16 classifer in order to infer the category. We used the Manahattan distance (L1 norm) to measure the distance between two points. It has been shown to perform quite well in "Effects of Distance Measure Choice on KNN Classifier Performance - A Review" for KNN in high dimension. For each candidate point from our stored embeddings database we compute the Manhattan distance to the embedding of user image. It should be noted that this operation can be parallelized to increase the processing speed. Once we get the closest points in the embeddings database, we search for their corresponding image in image database.

![demo results](https://github.com/DanielJohnHarty/DSTI_shazzam_for_clothes_image_processor/blob/master/Documents/imgs/demo.png)