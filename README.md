# AutoBugTriage: Automatically Predicting Bug Severity

## Description

Bug severity is an important factor in prioritizing which bugs to fix first. The process of triaging bug reports and assigning a severity requires developer expertise and knowledge of the underlying software. The Automatic Bug Traige (AutoBugTriage) tool allows for the prediction of bug severity at the beginning of the project by using an organization’s historical data, in the form of bug reports from past projects, to train the prediction classifier.

## Environment Setup
We recommend running our application in a python virtual environment to ensure the packages required do not conflict with your exisitng python distribution.

### Example in Linux

Install virtualenv
```bash
$ sudo apt install virtualenv
```

Create virtual environment in home directory
```bash
$ python3 -m venv thesis-env
```

Activate virtual environment
```bash
$ source ~/thesis-env/bin/activate
```

Install required packages
```bash
pip install -r requirements.txt
```

When done you can use deactivate
```bash
$ deactivate
```

### Other Platforms

Please see the following reference site for more information on python virtual environments.

https://docs.python.org/3/tutorial/venv.html
