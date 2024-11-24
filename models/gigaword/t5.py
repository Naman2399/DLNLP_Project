from transformers import pipeline

# Initialize the pipeline for text-to-text generation
pipe = pipeline("text2text-generation", model="soumagok/flan-t5-base-gigaword")

# Input text to summarize
ARTICLE_TO_SUMMARIZE = ("japan 's nec corp. and unk computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .")

# Generate summary using the pipeline
result = pipe(ARTICLE_TO_SUMMARIZE, max_length=50, clean_up_tokenization_spaces=True)

# Print the generated summary
print(result[0]['generated_text'])
