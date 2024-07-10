# Import libraries
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
import nltk
nltk.download('punkt')
from datapreparation import prepareData
tagged_data = prepareData()
# Model initialization
#model = Doc2Vec(vector_size = 50,
#min_count = 5,
#epochs = 100,
#alpha = 0.001
#)


# Vocabulary building
model.build_vocab(tagged_data)
# Get the vocabulary keys
keys = model.wv.key_to_index.keys()
# Print the length of the vocabulary keys
print(len(keys))

# Train the model
for epoch in range(model.epochs):
    print(f"Training epoch {epoch+1}/{model.epochs}")
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

model.save('resumeeval_doc2vec.model')
print("Model saved")

