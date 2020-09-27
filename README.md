# HDPS

Run this command to install django and python packages.
````
pip install -r requirements.txt
````

Then run this three command to initiate database.
````
python manage.py makemigrations
python manage.py migrate
````

After all of this finally run this command to run project
````
python manage.py runserver
````