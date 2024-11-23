from transformers import AutoTokenizer, PegasusForConditionalGeneration

# Load model directly
from transformers import pipeline

summarizer = pipeline("summarization", model="lidiya/bart-large-xsum-samsum")

ARTICLE_TO_SUMMARIZE = ("japan 's nec corp. and unk computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .")

print(summarizer(ARTICLE_TO_SUMMARIZE))
