# Named Entity Recognition(NER)

A named entity is a “real-world object” that’s assigned a name – for example, a person, a country, a product or a book title. 

For example, for a text: `Apple is looking at buying U.K. startup for $1 billion.`
|Text   |Start| End |Label  |Description|
| ----------- | ----------- | ----------- | ----------- |----------- |
Apple   |0  |5  |ORG    |Companies, agencies, institutions.|
U.K.    |27 |31 |   GPE |Geopolitical entity, i.e. countries, cities, states.|
$1 billion  |44 |54 |MONEY| Monetary values, including unit.|

Here, the word Apple is the organization, the word the UK is a geopolitical entity or country, and $1 billion is money. 

# Custom Named Entity Recognition model to identify disease name form clinical data using spaCy V3.2 Transformers

The project develops a custom Named Entity Recognition model with spaCy. The custom model can recognize the disease name from the clinical text.
# Dataset Description
To develop the custom NER model, we will use clinical text form [Biobert](https://github.com/dmis-lab/biobert) data on Github. 

They have provided a pre-processed version of datasets on NER:
* [Named Entity Recognition:](https://drive.google.com/file/d/1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh/view) (17.3 MB), eight datasets on biomedical named entity recognition.


If you want to download the data manually:
1. Open the following google drive link
   *  https://drive.google.com/file/d/1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh/view
  
2. Download the zip file and then extract the data inside the NERdata folder.

We are programmers; therefore, we will download the dataset by writing python code. Before that, let's install the required package on a virtual environment and then we will download the data through code.

### Install Virtual Environment

1. Install `virtualenv` package.

    ```sh
    pip3 install virtualenv
    ```
    Now let's clone the repository using the terminal.
    ```sh
    git clone git@github.com:kshitizregmi/Custom-NER-on-biobert-clinical-data.git
    ```

2. Goto NER-on-biomedical-disease-data folder and open the terminal and type 

    ```sh
    virtualenv disenv
    ``` 
    to initialize the virtual environment on the project folder.


3. After this, you will need to activate the virtual environment to start working on the project through the terminal. Use the following command:

    ```sh
    source disenv/bin/activate
    ``` 

    Once you do this, your prompt will change and it will show (disenv on the beginning of the absolute path of your project folder)

4. Install all required packages listed on requiremnts.txt

    ```sh
    pip3 install -r requirements.txt
    ``` 
    The project setup is complete now, let's download the data using code.


### Download data using code

You can download data through code using the following command:
```sh
python3 download_data.py
```
The above command will download all data in [Named Entity Recognition:](https://drive.google.com/file/d/1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh/view).

Here, we will only use the `BC5CDR-disease dataset.` In the dataset, `train_dev.tsv`, `test. tsv`, `train.tsv` are the data for validation, testing and training, respectively.

### Inside train.tsv

<center>
<img src = "https://drive.google.com/uc?export=view&id=184BsKaYyecW5CPTqO2OkipADebIOpUNw"/>
<figure caption >Figure 1: Clip of training data </figure caption>
</center>

Here,
*  B: Begin entity, 
*  I: Inside entity and,
*  O: Outside entity

spaCy doesn't support this format of data. Therefore we have to convert data to a spaCy readable format. To convert data into spacy format:
* Load the  train data and validation data using `load_data_spacy()`method inside `format_to_spacy.py`
* The loaded data is then converted to trainnable custom spaCy V3 format. 

You can do all this in one step by running the command:

```python
python3 load_and_reformat.py
```
The command will generate two files `train.spacy` and `valid.spacy` for training and validation of custom spacy model.


# How to train a custom NER model?


We will train the custom NER model on GPU by using the following configuration:


<img src = "https://drive.google.com/uc?export=view&id=1iu1m8UX_U49QBSCX63s5RVWyDx-M5UQ-"/>
<figure caption >Figure 2: Configuration </figure caption>

Selecting the shown configuration on the given [link](https://spacy.io/usage/training#config) generates a `base_config.cfg` file as shown below:


<img src = "https://drive.google.com/uc?export=view&id=10qMeeuRQu7myhr6m4reaCYpqLh6S1s-o"/>
<figure caption >Figure 3: Configuration File base_config.cfg</figure caption>

You can download the file using the icon on the bottom right corner and copy the file to your project directory.

After you download and add the configuration file you have to fill the values in the configuration file. You can autofill all the necessary details using the following command:

```sh
python3 -m spacy init fill-config base_config.cfg config.cfg
```

It will generate a complete auto-filled `config. cfg` file. To limit the number of training epochs, you can open this file and set the `max_epochs = 10` and don't forget to save the file. I will only use ten epochs because it takes a long time to train a model.

```config
[training]
.
.
max_epochs = 10
.
.
```

Now it is time to train a custom NER model. You can train the model using the following command:

```sh 
bash train_model
```
At the end of the training, i.e. after 10 epochs, the custom NER model is saved on folder `ner_demo/training` as `model-best`. The training image looks like this:


<center>
<img src = "https://drive.google.com/uc?export=view&id=1xO5wMNFqI21S5R9sUXmQSMncUP2Bj09u"/>
<figure caption >Figure 4: Training and scores </figure caption>
</center>



# How to Infer the output

To visualize the output on our custom data we will use the following code. The same code is written in the `disp_output.py` file. You can run the `python3 disp_output.py` command to visualize the result.


```python
import spacy
from spacy import displacy
MODEL_PATH = r'ner_demo/training/model-best'
ner = spacy.load(MODEL_PATH) #load the best model
doc = ner("Selegiline - induced postural hypotension in Parkinson ' s disease : a longitudinal study on the effects of drug withdrawal.The aims of this study were to confirm our previous findings in a separate cohort of patients and to determine the time course of the cardiovascular consequences of stopping selegiline in the expectation that this might shed light on the mechanisms by which the drug causes orthostatic hypotension")
# displacy.render(doc, style = "ent", jupyter = True)
displacy.serve(doc, style="ent")
```

For the given input text, the output from the custom NER model looks like the following image:

<img src = "https://drive.google.com/uc?export=view&id=1mW1gZjUlwjxQoF8jHsxpHi0mrr5L_Ter"/>
<figure caption >Figure 5: NER to detect disease form input text </figure caption>

You can see the image like above by opening `http://0.0.0.0:5000` on your browser after you run the command `python3 disp_output.py`.

# Model Performance on test data

* Precision  = 0.9594695319536979
* recall = 0.9594507835975112
* f1-score = 0.9513224382154714

The result can be seen by running

```python3
python3 metric.py
```

# References
* [Named Entity Recognition using Spacy and Tensorflow](https://aihub.cloud.google.com/u/0/p/products%2F2290fc65-0041-4c87-a898-0289f59aa8ba)
* [Biobert](https://github.com/dmis-lab/biobert)
* [💥 Out now: spaCy v3.2 ](https://spacy.io)
* [Custom Named Entity (Disease) Recognition in clinical text with spaCy 2.0 in Python](https://github.com/rsreetech/CustomNERwithspaCy)
