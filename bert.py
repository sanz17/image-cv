from transformers import BertTokenizer, BertModel

tok = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tok("Transformers are powerful", return_tensors="pt")
emb = model(**inputs).last_hidden_state
