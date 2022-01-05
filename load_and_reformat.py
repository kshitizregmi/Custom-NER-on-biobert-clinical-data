from format_to_spacy import load_data_spacy
TRAIN_DATA, LABELS = load_data_spacy("NERdata/BC5CDR-disease/train.tsv")
print("Length of Training data is {}".format(len(TRAIN_DATA)))
TEST_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/test.tsv")
print("Length of Training data is {}".format(len(TEST_DATA)))
VALID_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/train_dev.tsv")
print("Length of Training data is {}".format(len(VALID_DATA)))
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("./train.spacy") # save the docbin object

db = DocBin()
for text, annot in tqdm(VALID_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("./valid.spacy") # save the docbin object

