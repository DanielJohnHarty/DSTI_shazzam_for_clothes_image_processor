The work of team 3 was split in to two main requirements:

- generate recommendations
- create an api to simplify the repetetive, technical tasks for all teams in the pipeline

**Recommendation Generator**



**API**

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
