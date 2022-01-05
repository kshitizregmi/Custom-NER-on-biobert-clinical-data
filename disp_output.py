import spacy
from spacy import displacy
MODEL_PATH = r'ner_demo/training/model-best'
ner = spacy.load(MODEL_PATH) #load the best model
doc = ner("Selegiline - induced postural hypotension in Parkinson ' s disease : a longitudinal study on the effects of drug withdrawal.The aims of this study were to confirm our previous findings in a separate cohort of patients and to determine the time course of the cardiovascular consequences of stopping selegiline in the expectation that this might shed light on the mechanisms by which the drug causes orthostatic hypotension")
# displacy.render(doc, style = "ent", jupyter = True)
displacy.serve(doc, style="ent")

