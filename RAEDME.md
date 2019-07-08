# Project: ISSUE Detection from Text

## IssueModelService
This is a web service created with django Framework to consume a issue model classification and a rasa nlu model
   * The sentiment model classification classify short texts into ISSUE or NOT_ISSUE
   * The rasa nlu model classify short texts into many intents
   
#### How do I get set up?
The following section describes how to run the service locally.
* virtualenv venv
* source venv/bin/activate (Under windows run $ venv/Scripts/activate.bat)
* pip install -r requirements.txt
* python manage.py runserver
* navigate to localhost

### Install dependencies
 To get a development environment running you should :
 * Create a new virtual environment and easily install all libraries by running the following command :
	```
	python -m venv venv_name
	```
* To activate the new environment:
	```
	(venv_name) $ source activate  venv_name
	 
	```
 * Create a new virtual environment and easily install all libraries by running the following command :
	```
	(venv_name) $ pip install -r requirements.txt
	```
 In the file requirements.txt you find all necessary dependencies for this project.
 
 To generate the requirements file :
	```
	 (venv_name) $ pip freeze > requirements.txt
	```


### Running the tests
 
 * To run this project use this command:
```
python manage.py runserver
```
* To test this web service input should be a json object and contains 2 key: "id" and "message" and the 2 values must be string

### Building Docker image and running a container
* To build an image from docker file:
```
 docker build --tag=imagetag .
```
* To run a container from docker image ,mapping your machine’s port 88 to the container’s published port 80 using -p :
```
 docker run -p 88:80 image_tag
```
* docker is configured to use the default machine with IP@:
 ```
 192.168.99.100
```
     
To find the IP address, use the command 
 ```
 docker-machine ip
```

Go to that URL in a web browser to see the display content served up on a web page:
 ```
http://IP@:80/prediction
```




