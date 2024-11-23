# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")

ARTICLE_TO_SUMMARIZE = ("japan 's nec corp. and unk computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .")
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(outputs)
