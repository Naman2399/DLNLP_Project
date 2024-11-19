from rouge import Rouge

r = Rouge()
pred_text = "Hello how are u"
target_text = "Who are u"
scores = r.get_scores(pred_text, target_text)

print(scores)
print(scores[0]["rouge-1"]['f'])
print(scores[0]["rouge-2"]['f'])
print(scores[0]["rouge-l"]['f'])