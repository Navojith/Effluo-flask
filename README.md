### Server - Flask

### Setting Up a Virtual Environment

It's recommended to create a virtual environment to isolate project dependencies. Open a terminal navigate to the root directory of the project and run the following commands:

```
# Create a virtual environment (replace 'myenv' with your preferred name)
python -m venv myenv
```

```
# Activate the virtual environment
# On Windows
myenv\Scripts\activate
```

### Add your venv to the gitignore file

Once you have activated the virtual environment, you can install the project dependencies using pip

```
pip install -r requirements.txt
```

If you install anyother dependencies make sure to update the requirements.txt by running the following command

```
pip freeze > requirements.txt
```

### Run app

```
python run.py
```
