from tokenizers import BasicTokenizer, RegexTokenizer

with open("GPT/tokenizer/taylorswift.txt", "r", encoding = "utf-8") as f:
    text = f.read()

tokenizer1 = BasicTokenizer()
tokenizer1.train(text, 512, verbose = True)

tokenizer2 = RegexTokenizer()
tokenizer2.train(text, 512, verbose = True)