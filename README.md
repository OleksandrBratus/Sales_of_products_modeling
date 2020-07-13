## Init project(dev)
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
## Run project
flask db upgrade

flask run

goto http://localhost:5000/api/
