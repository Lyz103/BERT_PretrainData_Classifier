def text2token(text,tokenizer,max_length=512):
    text2id = tokenizer(
        text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt"
    )
    input_ids=text2id["input_ids"].tolist()
    attention_mask=text2id["attention_mask"].tolist()
    return input_ids,attention_mask