# hackathon-kin

## Hackathon Kin: Customers churn (Finance and Risk).

Classification project based on Supervised Machine Learning using Logistic
Regression, Random Forest, K-Neighbours, SVC, Decision Tree, Gradient Boosting,
XGB, LGBM, Cat Boost, Ada Boost and ANN.\

![Churn](https://www.diplomadosonline.com/wp-content/uploads/2020/09/customer-churn-imagen.jpeg)


### Requirements

Python 3.10+

### Git

- First, clone repository:

```
git clone https://github.com/jpcadena/hackathon-kin.git
```

- Change directory to root project with:

```
  cd hackathon-kin

```

- Create your git branch with the following:

```
git checkout -b <new_branch>
```

For _<new_branch>_ use some convention as following:

```
yourgithubusername
```

Or if some work in progress (WIP) or bug shows up, you can use:

```
yourgithubusername_feature
```

- Switch to your branch:

```
git checkout <new_branch>
```

- **Before** you start working on some section, retrieve the latest changes
  with:

```
git pull
```

- Add your new files and changes:

```
git add .
```

- Make your commit with a reference message about the fix/changes.

```
git commit -m "Commit message"
```

- First push for remote branch:

```
git push --set-upstream origin <new_branch>
```

- Latter pushes:

```
git push origin
```

### Environment

- Create a **virtual environment** 'sample_venv' with:

```
python3 -m venv sample_venv
```

- Activate environment in Windows with:

```
.\sample_venv\Scripts\activate
```

- Or with Unix or Mac:

```
source sample_venv/bin/activate
```

### Installation of libraries and dependencies

```
pip install -r requirements.txt
```

### Execution

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Environment credentials

Rename **sample.env** to **.env** and replace your SMTP e-mail credentials.

### Documentation

Use docstrings with **reStructuredText** format by adding triple double quotes
**"""** after function definition.\
Add a brief function description, also for the parameters including the return
value and its corresponding data type.

### Additional information

If you want to give more style and a better format to this README.md file,
check documentation
at [GitHub Docs](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).\
Please use **linting** to check your code quality
following [PEP 8](https://peps.python.org/pep-0008/). Check documentation
for [Visual Studio Code](https://code.visualstudio.com/docs/python/linting#_run-linting)
or
for [Jetbrains Pycharm](https://github.com/leinardi/pylint-pycharm/blob/master/README.md).\
Recommended plugin for
autocompletion: [Tabnine](https://www.tabnine.com/install)
