import stanfordnlp
import pandas as pd
import os

stanfordnlp.download('en', force=True)
nlp = stanfordnlp.Pipeline(lang='en', processors="tokenize,lemma,pos,depparse")

df = pd.read_csv(os.path.join(os.getcwd(), ".cache", "raw_data", "e_snli", "cleaned_data", "dev.csv"),
                 sep=",")
sent_df = df.premise + df.hypothesis
for s in sent_df:
    doc = nlp(s)

    print(*[f"index: {word.index.rjust(2)}\
    \tword: {word.text.ljust(11)}\tgovernor index: {word.governor}\
    \tgovernor: {(doc.sentences[0].words[word.governor-1].text if word.governor > 0 else 'root').ljust(11)}\
    \tdeprel: {word.dependency_relation}" for word in doc.sentences[0].words], sep='\n')

    break
